"""
dlg_text.py - DLG on masked language modelling with BERT (paper Section 4.2).

The official repo does not include this experiment, so it is reimplemented here
from the paper's description:
  - the victim computes a masked-language-model gradient on one sentence
    (15% of tokens replaced by [MASK]);
  - the attacker knows the model, its weights and the sequence length, and
    optimises continuous dummy *embeddings* plus dummy soft labels with L-BFGS
    so that the dummy gradient matches the shared one (100 iterations);
  - each recovered embedding is mapped to its nearest entry in BERT's
    word-embedding matrix to read off a token.

Design choices the paper does not specify (state these in the report):
  - The word-embedding matrix is frozen (as often in fine-tuning). Otherwise its
    gradient would differ structurally between the real input (token lookups)
    and the dummy input (continuous embeddings) and could never be matched.
    BERT ties this matrix to the output layer, so that is frozen too.
  - --label-mode all (default): the loss covers every position, so the label at
    a masked position reveals the hidden word. --label-mode masked: standard
    MLM, loss only at masked positions, attacker assumed to know where they are.
  - Default model is a small randomly initialised BERT (CPU-friendly);
    --size base --pretrained uses the real bert-base-uncased (much slower).

Metrics (on real tokens only, excluding [CLS]/[SEP]):
  input_token_acc      embedding -> nearest token vs. the (masked) input
  label_token_acc      recovered labels vs. true labels
  recovered_token_acc  combined: embedding tokens, with masked slots filled from
                       the recovered labels, vs. the ORIGINAL sentence
  norm_edit_distance   token-level Levenshtein distance / length
  exact_match          whole sentence recovered

Examples:
  python dlg_text.py                                   # 3 built-in sentences x 3 seeds
  python dlg_text.py --sentences paper_sentences.txt   # the paper's sentences
  python dlg_text.py --wikitext 20 --seeds 0           # 20 WikiText-2 sentences
  pip install transformers datasets                    # needed first
"""

import argparse
import csv
import json
import math
import os
import random
import statistics
import time

import torch
import torch.nn.functional as F

SIZES = {
    "tiny": dict(num_hidden_layers=2, hidden_size=128, num_attention_heads=2, intermediate_size=512),
    "small": dict(num_hidden_layers=4, hidden_size=256, num_attention_heads=4, intermediate_size=1024),
    "base": dict(),  # bert-base-uncased: 12 layers, hidden 768
}

# Placeholder sentences. For a faithful replication, put the three sentences
# from the paper's text figure into a file and pass --sentences.
DEFAULT_SENTENCES = [
    "The conference brings together researchers working on machine learning and neural computation.",
    "Participants keep their training data on local devices and only share model updates.",
    "Gradients computed from private examples may reveal the examples themselves.",
]


def load_sentences(args):
    if args.sentences:
        with open(args.sentences) as f:
            return [l.strip() for l in f if l.strip()]
    if args.wikitext:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        pool = []
        for para in ds["text"]:
            para = para.strip()
            if not para or para.startswith("="):
                continue
            para = (para.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")
                        .replace(" ,", ",").replace(" ;", ";").replace(" 's", "'s"))
            for sent in para.split(" . "):
                n = len(sent.split())
                if 8 <= n <= 20:
                    pool.append(sent.strip().rstrip(" .") + ".")
        rng = random.Random(args.data_seed)
        return rng.sample(pool, min(args.wikitext, len(pool)))
    return DEFAULT_SENTENCES


def build_model(args, device):
    from transformers import BertConfig, BertForMaskedLM, BertTokenizerFast
    tok = BertTokenizerFast.from_pretrained("bert-base-uncased")
    if args.pretrained:
        model = BertForMaskedLM.from_pretrained("bert-base-uncased", attn_implementation="eager")
    else:
        cfg = BertConfig.from_pretrained("bert-base-uncased", **SIZES[args.size],
                                         hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0)
        torch.manual_seed(args.model_seed)
        try:
            # eager attention: guaranteed second-order derivatives
            model = BertForMaskedLM._from_config(cfg, attn_implementation="eager")
        except TypeError:
            model = BertForMaskedLM(cfg)
    model.to(device).eval()  # eval = no dropout; gradients still flow
    model.bert.embeddings.word_embeddings.weight.requires_grad_(False)  # tied to decoder
    return tok, model


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def attack_sentence(sentence, tok, model, args, seed, device, log):
    emb = model.bert.embeddings.word_embeddings
    V, H = emb.weight.shape
    params = [p for p in model.parameters() if p.requires_grad]

    enc = tok(sentence, return_tensors="pt", truncation=True, max_length=args.max_len)
    ids = enc["input_ids"].to(device)
    L = ids.size(1)
    special = set(tok.all_special_ids)
    content = [i for i in range(L) if ids[0, i].item() not in special]

    # Mask positions depend only on the sentence, so seeds vary just the attacker's start
    mrng = random.Random(f"{args.data_seed}:{sentence}")
    n_mask = max(1, round(args.mask_prob * len(content)))
    mpos = sorted(mrng.sample(content, n_mask))
    masked = ids.clone()
    masked[0, mpos] = tok.mask_token_id
    label_pos = list(range(L)) if args.label_mode == "all" else mpos
    targets = ids[0, label_pos]

    def loss_fn(inputs_embeds, label_probs):
        logits = model(inputs_embeds=inputs_embeds).logits[0, label_pos]
        return torch.mean(torch.sum(-label_probs * F.log_softmax(logits, dim=-1), dim=-1))

    def grads(loss, create_graph):
        return torch.autograd.grad(loss, params, create_graph=create_graph, allow_unused=True)

    # 1. Victim's shared gradient
    true_embeds = emb(masked).detach()
    orig = [None if g is None else g.detach().clone()
            for g in grads(loss_fn(true_embeds, F.one_hot(targets, V).float()), False)]

    # 2. Attacker
    torch.manual_seed(seed)
    dummy_emb = (torch.randn(1, L, H, device=device) * emb.weight.std()).requires_grad_(True)
    dummy_label = torch.randn(len(label_pos), V, device=device).requires_grad_(True)
    opt = torch.optim.LBFGS([dummy_emb, dummy_label], lr=args.lr)

    def closure():
        opt.zero_grad()
        dg = grads(loss_fn(dummy_emb, F.softmax(dummy_label, dim=-1)), True)
        diff = 0
        for gx, gy in zip(dg, orig):
            if gx is not None and gy is not None:
                diff = diff + ((gx - gy) ** 2).sum()
        diff.backward()
        return diff

    def recover():
        with torch.no_grad():
            emb_tok = torch.cdist(dummy_emb[0], emb.weight).argmin(dim=-1)
            lab_tok = dummy_label.argmax(dim=-1)
            combined = emb_tok.clone()
            for k, pos in enumerate(label_pos):
                if pos in mpos:
                    combined[pos] = lab_tok[k]
        return emb_tok, lab_tok, combined

    start, history, diverged = time.time(), [], False
    for it in range(args.iters):
        opt.step(closure)
        if it % 10 == 0 or it == args.iters - 1:
            cur = closure().item()
            history.append((it, cur))
            _, _, comb = recover()
            log.write(f"  iter {it:>4}  loss {cur:.4g}  | {tok.decode(comb[content])}\n")
            if not math.isfinite(cur):
                diverged = True
                break
    runtime = time.time() - start

    emb_tok, lab_tok, combined = recover()
    c = torch.tensor(content, device=device)
    orig_toks = ids[0, c].tolist()
    rec_toks = combined[c].tolist()
    label_content = [k for k, pos in enumerate(label_pos) if pos in content]
    return dict(
        original=tok.decode(ids[0, c]), recovered=tok.decode(combined[c]),
        n_tokens=len(content), n_masked=len(mpos),
        input_token_acc=(emb_tok[c] == masked[0, c]).float().mean().item(),
        label_token_acc=(lab_tok[label_content] == targets[label_content]).float().mean().item(),
        recovered_token_acc=sum(a == b for a, b in zip(rec_toks, orig_toks)) / len(orig_toks),
        norm_edit_distance=edit_distance(rec_toks, orig_toks) / len(orig_toks),
        exact_match=int(rec_toks == orig_toks),
        final_grad_loss=history[-1][1], runtime_s=runtime, diverged=int(diverged),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sentences", help="text file, one sentence per line")
    p.add_argument("--wikitext", type=int, default=0, help="sample N sentences from WikiText-2")
    p.add_argument("--size", default="tiny", choices=list(SIZES))
    p.add_argument("--pretrained", action="store_true", help="real bert-base-uncased weights")
    p.add_argument("--label-mode", default="all", choices=["all", "masked"])
    p.add_argument("--mask-prob", type=float, default=0.15)
    p.add_argument("--max-len", type=int, default=32)
    p.add_argument("--iters", type=int, default=100, help="paper: 100")
    p.add_argument("--lr", type=float, default=1.0)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--model-seed", type=int, default=1234)
    p.add_argument("--data-seed", type=int, default=0)
    p.add_argument("--out", default="results")
    p.add_argument("--tag", default="")
    args = p.parse_args()
    if args.pretrained:
        args.size = "base"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sentences = load_sentences(args)
    tok, model = build_model(args, device)
    source = "wikitext" if args.wikitext else ("file" if args.sentences else "builtin")
    run = f"text_{source}_{args.size}" + ("_pretrained" if args.pretrained else "") + \
          (f"_{args.tag}" if args.tag else "")
    out_dir = os.path.join(args.out, run)
    os.makedirs(out_dir, exist_ok=True)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"BERT-{args.size}{' (pretrained)' if args.pretrained else ''}, {n_params:,} shared params, "
          f"{len(sentences)} sentences x {len(args.seeds)} seeds, device {device}\n")

    fields = ["sentence_id", "seed", "original", "recovered", "n_tokens", "n_masked",
              "input_token_acc", "label_token_acc", "recovered_token_acc", "norm_edit_distance",
              "exact_match", "final_grad_loss", "runtime_s", "diverged"]
    rows = []
    with open(os.path.join(out_dir, "results.csv"), "w", newline="") as f, \
            open(os.path.join(out_dir, "progress.txt"), "w") as log:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for sid, sent in enumerate(sentences):
            for seed in args.seeds:
                log.write(f"\nSentence {sid}, seed {seed}\n  original: {sent}\n")
                r = attack_sentence(sent, tok, model, args, seed, device, log)
                r.update(sentence_id=sid, seed=seed)
                w.writerow({k: r[k] for k in fields})
                f.flush()
                log.flush()
                rows.append(r)
                print(f"[{sid + 1}/{len(sentences)}] seed {seed}: token acc {r['recovered_token_acc']:.0%}  "
                      f"edit {r['norm_edit_distance']:.2f}  {r['runtime_s']:.0f}s\n"
                      f"   orig: {r['original']}\n   rec:  {r['recovered']}")

    def mean(k):
        return statistics.mean(r[k] for r in rows)
    summary = dict(type="text", run=run, model=f"bert-{args.size}", pretrained=args.pretrained,
                   label_mode=args.label_mode, iters=args.iters, seeds=args.seeds,
                   n_sentences=len(sentences), n_trials=len(rows),
                   input_token_acc=mean("input_token_acc"), label_token_acc=mean("label_token_acc"),
                   recovered_token_acc=mean("recovered_token_acc"),
                   norm_edit_distance=mean("norm_edit_distance"),
                   exact_match_rate=mean("exact_match"), runtime_s=mean("runtime_s"),
                   n_diverged=sum(r["diverged"] for r in rows))
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\n" + json.dumps(summary, indent=2) + f"\nSaved to {out_dir}/ (progress.txt shows text over iterations)")


if __name__ == "__main__":
    main()

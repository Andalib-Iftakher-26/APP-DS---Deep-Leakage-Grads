"""
dlg_lib.py - Core Deep Leakage from Gradients (DLG) components.

Mirrors the official implementation (github.com/mit-han-lab/dlg):
  - LeNet with Sigmoid activations (models/vision.py)
  - weights_init: uniform(-0.5, 0.5) on weights and biases
  - label_to_onehot / cross_entropy_for_onehot (utils.py)
  - L-BFGS optimisation of dummy image + dummy (soft) label, model seed 1234

Extensions for the reproduction study:
  - LeNet generalised to any channel count / image size / class count
    (identical to the original for CIFAR-100: 3x32x32, 100 classes, fc 768->100)
  - Loaders for CIFAR-100, CIFAR-10, MNIST, Fashion-MNIST, SVHN, LFW, and an
    image-folder dataset (the Smartphone Natural Objects dataset)
  - A Sigmoid ResNet family (resnet20/32/56) approximating the paper's
    "ResNet-56 with ReLU replaced by Sigmoid and strides removed"
  - Evaluation metrics: MSE, PSNR, SSIM, label recovery
"""

import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms


# --------------------------------------------------------------------------
# Model (same architecture as the official LeNet)
# --------------------------------------------------------------------------
class LeNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=100, image_size=32):
        super().__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
            act(),
        )
        # Work out the flattened size instead of hard-coding 768,
        # so the same model works for 28x28, 32x32 and 64x64 inputs.
        with torch.no_grad():
            n_flat = self.body(torch.zeros(1, in_channels, image_size, image_size)).numel()
        self.fc = nn.Sequential(nn.Linear(n_flat, num_classes))

    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        return self.fc(out)


def weights_init(m):
    if hasattr(m, "weight") and m.weight is not None:
        m.weight.data.uniform_(-0.5, 0.5)
    if hasattr(m, "bias") and m.bias is not None:
        m.bias.data.uniform_(-0.5, 0.5)


class _SigmoidBlock(nn.Module):
    """CIFAR-style basic residual block with Sigmoid instead of ReLU, no BatchNorm
    (BatchNorm with a single image is ill-defined, and the paper needs the model
    to be twice differentiable)."""

    def __init__(self, cin, cout, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1)
        self.shortcut = None
        if stride != 1 or cin != cout:
            self.shortcut = nn.Conv2d(cin, cout, 1, stride)

    def forward(self, x):
        out = torch.sigmoid(self.conv1(x))
        out = self.conv2(out)
        sc = x if self.shortcut is None else self.shortcut(x)
        return torch.sigmoid(out + sc)


class ResNetSigmoid(nn.Module):
    """ResNet-(6n+2) for small images (He et al., 2016, CIFAR variant), with the
    paper's two modifications: Sigmoid activations and no strides (strides=False).
    depth 56 = the paper's model. The official repo does not ship a working version."""

    def __init__(self, depth=20, in_channels=3, num_classes=100, image_size=32, strides=False):
        super().__init__()
        assert (depth - 2) % 6 == 0, "depth must be 6n+2 (20, 32, 44, 56)"
        n = (depth - 2) // 6
        self.stem = nn.Conv2d(in_channels, 16, 3, 1, 1)
        layers, cin = [], 16
        for stage, cout in enumerate([16, 32, 64]):
            for b in range(n):
                stride = 2 if (strides and stage > 0 and b == 0) else 1
                layers.append(_SigmoidBlock(cin, cout, stride))
                cin = cout
        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        out = torch.sigmoid(self.stem(x))
        out = self.layers(out)
        out = out.mean(dim=(2, 3))  # global average pooling
        return self.fc(out)


ARCHS = ("lenet", "resnet20", "resnet32", "resnet56")


def label_to_onehot(target, num_classes=100):
    target = torch.unsqueeze(target, 1)
    onehot = torch.zeros(target.size(0), num_classes, device=target.device)
    onehot.scatter_(1, target, 1)
    return onehot


def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(-target * F.log_softmax(pred, dim=-1), 1))


# --------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------
# Every image is converted to a tensor in [0, 1] with ToTensor() only,
# exactly as in the official code (no mean/std normalisation).
DATASET_INFO = {
    "cifar100":   dict(channels=3, size=32, classes=100),
    "cifar10":    dict(channels=3, size=32, classes=10),
    "mnist":      dict(channels=1, size=28, classes=10),
    "fmnist":     dict(channels=1, size=28, classes=10),
    "svhn":       dict(channels=3, size=32, classes=10),
    "lfw":        dict(channels=3, size=32, classes=None),  # classes set on load
    "smartphone": dict(channels=3, size=32, classes=10),  # size overridable
}


class ListDataset(torch.utils.data.Dataset):
    """Minimal dataset: items[i] -> PIL image via `loader`, plus integer targets."""

    def __init__(self, items, targets, classes, loader):
        self.items, self.targets, self.classes, self._loader = items, list(targets), classes, loader

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self._loader(self.items[i]), self.targets[i]


def _square_centre(img, frac=1.0):
    w, h = img.size
    s = int(min(w, h) * frac)
    left, top = (w - s) // 2, (h - s) // 2
    return img.crop((left, top, left + s, top + s))


def load_lfw(root, min_faces=20):
    """
    Labeled Faces in the Wild, identities with >= min_faces images
    (20 -> 62 people, ~3,000 images). The paper does not say how LFW was
    preprocessed; here: the standard face-region crop, centre square, then resize.
    Tries scikit-learn's copy first (more reliable download), then torchvision's.
    """
    from PIL import Image
    try:
        import numpy as np
        from sklearn.datasets import fetch_lfw_people
        lfw = fetch_lfw_people(data_home=root, min_faces_per_person=min_faces,
                               color=True, resize=1.0, funneled=True)
        imgs = lfw.images
        if imgs.max() <= 1.0:
            imgs = imgs * 255.0
        imgs = imgs.clip(0, 255).astype(np.uint8)

        def loader(i):
            return _square_centre(Image.fromarray(imgs[i]))
        return ListDataset(list(range(len(imgs))), lfw.target.tolist(),
                           [str(n) for n in lfw.target_names], loader)
    except ImportError:
        pass
    base = datasets.LFWPeople(root, split="10fold", image_set="funneled", download=True)
    counts = {}
    for t in base.targets:
        counts[t] = counts.get(t, 0) + 1
    keep = sorted(t for t, c in counts.items() if c >= min_faces)
    remap = {t: i for i, t in enumerate(keep)}
    idx_to_name = {v: k for k, v in base.class_to_idx.items()}
    items = [i for i, t in enumerate(base.targets) if t in remap]

    def loader(i):
        # 250x250 funneled image; the central 50% roughly matches sklearn's face crop
        return _square_centre(base[i][0], frac=0.5)
    return ListDataset(items, [remap[base.targets[i]] for i in items],
                       [idx_to_name[t] for t in keep], loader)


def load_dataset(name, root="~/.torch", smartphone_root=None, image_size=None):
    """Return (dataset, info). dataset[i] -> (PIL image, int label)."""
    root = os.path.expanduser(root)
    info = dict(DATASET_INFO[name])
    if name == "cifar100":
        dst = datasets.CIFAR100(root, download=True)
    elif name == "cifar10":
        dst = datasets.CIFAR10(root, download=True)
    elif name == "mnist":
        dst = datasets.MNIST(root, download=True)
    elif name == "fmnist":
        dst = datasets.FashionMNIST(root, download=True)
    elif name == "svhn":
        dst = datasets.SVHN(root, split="test", download=True)  # 26k images, smaller download
    elif name == "lfw":
        dst = load_lfw(root)
        info["classes"] = len(dst.classes)
    elif name == "smartphone":
        if smartphone_root is None:
            raise ValueError("--smartphone-root is required for the smartphone dataset")
        # Expects <root>/<class_name>/<image>.png (output of prepare_smartphone.py)
        dst = datasets.ImageFolder(os.path.expanduser(smartphone_root))
        info["classes"] = len(dst.classes)
        first_img, _ = dst[0]
        info["size"] = first_img.size[0]
    else:
        raise ValueError(f"Unknown dataset: {name}")
    if image_size is not None:
        info["size"] = image_size
    return dst, info


def get_targets(dst):
    """Class label for every item, without loading the images."""
    for attr in ("targets", "labels"):
        if hasattr(dst, attr):
            t = getattr(dst, attr)
            return [int(x) for x in (t.tolist() if torch.is_tensor(t) else t)]
    if hasattr(dst, "samples"):  # ImageFolder
        return [s[1] for s in dst.samples]
    return [int(dst[i][1]) for i in range(len(dst))]


def class_names(dst):
    if hasattr(dst, "classes"):
        return list(dst.classes)
    return None


def stratified_indices(dst, n_images, seed=0):
    """Pick n_images indices, cycling through classes so classes are balanced."""
    targets = get_targets(dst)
    g = torch.Generator().manual_seed(seed)
    by_class = {}
    for idx in torch.randperm(len(targets), generator=g).tolist():
        by_class.setdefault(targets[idx], []).append(idx)
    order = sorted(by_class)
    chosen, round_i = [], 0
    while len(chosen) < n_images:
        added = False
        for c in order:
            if round_i < len(by_class[c]) and len(chosen) < n_images:
                chosen.append(by_class[c][round_i])
                added = True
        if not added:
            break
        round_i += 1
    return chosen


def get_image_tensor(dst, index, info):
    img, label = dst[index]
    img = img.convert("RGB" if info["channels"] == 3 else "L")
    if img.size != (info["size"], info["size"]):
        img = img.resize((info["size"], info["size"]), resample=3)  # bicubic
    return transforms.ToTensor()(img), int(label)


# --------------------------------------------------------------------------
# Metrics (computed by the researcher only - the attacker never sees x)
# --------------------------------------------------------------------------
def mse(x, x_hat):
    return F.mse_loss(x_hat.clamp(0, 1), x).item()


def psnr(x, x_hat):
    m = mse(x, x_hat)
    return float("inf") if m == 0 else 10 * math.log10(1.0 / m)


def _gaussian_window(size=11, sigma=1.5, channels=3, device="cpu"):
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).unsqueeze(0)
    window = (g.t() @ g).unsqueeze(0).unsqueeze(0)
    return window.expand(channels, 1, size, size).contiguous()


def ssim(x, x_hat, window_size=11):
    """Standard SSIM (Wang et al., 2004), Gaussian window, data range [0, 1]."""
    x = x.unsqueeze(0) if x.dim() == 3 else x
    y = x_hat.clamp(0, 1)
    y = y.unsqueeze(0) if y.dim() == 3 else y
    c = x.size(1)
    window_size = min(window_size, x.size(-1))
    w = _gaussian_window(window_size, 1.5, c, x.device)
    pad = window_size // 2
    mu_x = F.conv2d(x, w, padding=pad, groups=c)
    mu_y = F.conv2d(y, w, padding=pad, groups=c)
    sxx = F.conv2d(x * x, w, padding=pad, groups=c) - mu_x ** 2
    syy = F.conv2d(y * y, w, padding=pad, groups=c) - mu_y ** 2
    sxy = F.conv2d(x * y, w, padding=pad, groups=c) - mu_x * mu_y
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mu_x * mu_y + C1) * (2 * sxy + C2)) / ((mu_x ** 2 + mu_y ** 2 + C1) * (sxx + syy + C2))
    return s.mean().item()


# --------------------------------------------------------------------------
# The attack
# --------------------------------------------------------------------------
def build_model(info, model_seed=1234, device="cpu", arch="lenet", init="official",
                resnet_strides=False):
    """
    arch: lenet (official demo) | resnet20 | resnet32 | resnet56 (paper's model).
    init: "official" = uniform(-0.5, 0.5) as in the repo; "default" = PyTorch default.
    The model's initialiser is kept as net.reinit so the official random stream
    can be reproduced (see run_dlg).
    """
    if arch == "lenet":
        net = LeNet(info["channels"], info["classes"], info["size"])
    elif arch.startswith("resnet"):
        net = ResNetSigmoid(int(arch[len("resnet"):]), info["channels"], info["classes"],
                            info["size"], strides=resnet_strides)
    else:
        raise ValueError(f"Unknown arch {arch}; choose from {ARCHS}")
    net = net.to(device)

    def reinit():
        if init == "official":
            net.apply(weights_init)
        else:
            for m in net.modules():
                if m is not net and hasattr(m, "reset_parameters"):
                    m.reset_parameters()

    net.reinit = reinit
    torch.manual_seed(model_seed)  # same as the official code
    reinit()
    return net


def run_dlg(net, gt_data, gt_label, num_classes, iters=300, dummy_seed=0,
            snapshot_every=10, device="cpu", model_seed=1234):
    """
    Run one DLG attack on a single image.

    gt_data: tensor (C, H, W) in [0, 1];  gt_label: int.
    dummy_seed: seed for the attacker's random starting noise. Use -1 for
      "official" mode: like main.py, the noise is drawn from the same random
      stream straight after torch.manual_seed(1234) + weights_init, so the
      result matches main.py exactly.
    Returns a dict with the reconstruction, recovered label, loss history and
    snapshots (for animations), plus runtime.
    """
    criterion = cross_entropy_for_onehot
    gt_data = gt_data.unsqueeze(0).to(device)
    gt_label_t = torch.tensor([gt_label], device=device).long()
    gt_onehot = label_to_onehot(gt_label_t, num_classes=num_classes)

    # 1. The victim computes and shares its gradient.
    pred = net(gt_data)
    loss = criterion(pred, gt_onehot)
    dy_dx = torch.autograd.grad(loss, net.parameters())
    original_dy_dx = [g.detach().clone() for g in dy_dx]

    # 2. The attacker starts from random noise (seeded so trials are repeatable).
    if dummy_seed is None or dummy_seed < 0:
        # Official main.py order: seed 1234 -> weights_init -> randn (no reseed).
        # Re-applying weights_init sets identical weights and advances the RNG
        # exactly as main.py does.
        torch.manual_seed(model_seed)
        net.reinit() if hasattr(net, "reinit") else net.apply(weights_init)
    else:
        torch.manual_seed(dummy_seed)
    dummy_data = torch.randn(gt_data.size(), device=device).requires_grad_(True)
    dummy_label = torch.randn(gt_onehot.size(), device=device).requires_grad_(True)
    optimizer = torch.optim.LBFGS([dummy_data, dummy_label])  # lr=1, history=100, max_iter=20 (defaults)

    loss_history, snapshots = [], []
    start = time.time()
    diverged = False
    for it in range(iters):
        def closure():
            optimizer.zero_grad()
            dummy_pred = net(dummy_data)
            dummy_onehot = F.softmax(dummy_label, dim=-1)
            dummy_loss = criterion(dummy_pred, dummy_onehot)
            dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)
            grad_diff = 0
            for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                grad_diff += ((gx - gy) ** 2).sum()
            grad_diff.backward()
            return grad_diff

        optimizer.step(closure)
        if it % snapshot_every == 0 or it == iters - 1:
            current = closure().item()
            loss_history.append((it, current))
            snapshots.append((it, dummy_data.detach()[0].cpu().clone()))
            if not math.isfinite(current):
                diverged = True
                break
    runtime = time.time() - start

    recon = dummy_data.detach()[0].cpu()
    return dict(
        recon=recon,
        pred_label=int(dummy_label.detach().argmax(dim=-1).item()),
        loss_history=loss_history,
        snapshots=snapshots,
        runtime_s=runtime,
        diverged=diverged,
    )


def iters_to_converge(loss_history, threshold=1e-3):
    """First logged iteration where the gradient-matching loss drops below threshold."""
    for it, l in loss_history:
        if math.isfinite(l) and l < threshold:
            return it
    return None

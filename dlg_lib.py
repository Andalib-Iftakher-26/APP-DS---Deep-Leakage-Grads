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
  - Loaders for CIFAR-100, CIFAR-10, MNIST, Fashion-MNIST, SVHN, and an
    image-folder dataset (the Smartphone Natural Objects dataset)
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
    "smartphone": dict(channels=3, size=32, classes=10),  # size overridable
}


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
        dst = datasets.SVHN(root, split="train", download=True)
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
def build_model(info, model_seed=1234, device="cpu"):
    net = LeNet(info["channels"], info["classes"], info["size"]).to(device)
    torch.manual_seed(model_seed)  # same as the official code
    net.apply(weights_init)
    return net


def run_dlg(net, gt_data, gt_label, num_classes, iters=300, dummy_seed=0,
            snapshot_every=10, device="cpu"):
    """
    Run one DLG attack on a single image.

    gt_data: tensor (C, H, W) in [0, 1];  gt_label: int.
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

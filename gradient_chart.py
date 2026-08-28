import random
import argparse

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

from models.vision import LeNet, weights_init
from utils import cross_entropy_for_onehot, label_to_onehot


ITERATIONS = 300


def main():
    parser = argparse.ArgumentParser(
        description="Plot gradient difference for a CIFAR-100 image."
    )
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="CIFAR-100 image index; choose randomly when omitted",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = datasets.CIFAR100("~/.torch", download=True)
    if args.index is None:
        image_index = random.randrange(len(dataset))
    elif 0 <= args.index < len(dataset):
        image_index = args.index
    else:
        parser.error("--index must be between 0 and %d" % (len(dataset) - 1))

    to_tensor = transforms.ToTensor()
    image, label = dataset[image_index]
    gt_data = to_tensor(image).to(device).view(1, 3, 32, 32)
    gt_label = torch.tensor([label], dtype=torch.long, device=device)
    gt_onehot_label = label_to_onehot(gt_label)

    torch.manual_seed(1234)
    net = LeNet().to(device)
    net.apply(weights_init)
    criterion = cross_entropy_for_onehot

    pred = net(gt_data)
    original_loss = criterion(pred, gt_onehot_label)
    original_gradients = torch.autograd.grad(original_loss, net.parameters())
    original_gradients = [gradient.detach().clone() for gradient in original_gradients]

    dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
    dummy_label = torch.randn(gt_onehot_label.size()).to(device).requires_grad_(True)
    optimizer = torch.optim.LBFGS([dummy_data, dummy_label])
    losses = []

    def gradient_difference(create_graph):
        dummy_pred = net(dummy_data)
        dummy_onehot_label = F.softmax(dummy_label, dim=-1)
        dummy_loss = criterion(dummy_pred, dummy_onehot_label)
        dummy_gradients = torch.autograd.grad(
            dummy_loss, net.parameters(), create_graph=create_graph
        )
        return sum(
            ((dummy_gradient - original_gradient) ** 2).sum()
            for dummy_gradient, original_gradient in zip(
                dummy_gradients, original_gradients
            )
        )

    def closure():
        optimizer.zero_grad()
        difference = gradient_difference(create_graph=True)
        difference.backward()
        return difference

    for iteration in range(ITERATIONS):
        optimizer.step(closure)
        losses.append(gradient_difference(create_graph=False).item())

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, ITERATIONS + 1), losses)
    plt.xlabel("Iteration")
    plt.ylabel("Gradient difference")
    plt.title("Gradient difference over 300 iterations (index %d)" % image_index)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("gradient_chart.png")
    print("Randomly selected CIFAR-100 index: %d" % image_index)
    print("Initial gradient difference: %.6f" % losses[0])
    print("Final gradient difference: %.6f" % losses[-1])
    print("Chart saved to gradient_chart.png")
    plt.show()


if __name__ == "__main__":
    main()

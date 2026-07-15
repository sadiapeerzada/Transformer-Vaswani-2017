"""
Training script for the from-scratch DDPM.

Usage:
    python3 train.py --epochs 400 --batch_size 128

Saves:
    outputs/loss_curve.png                 - training loss vs. step
    outputs/samples/epoch_XXXX.png          - grid of generated samples at checkpoint epochs
    outputs/samples/denoising_trajectory.png- single sample shown across reverse timesteps
    outputs/model.pt                        - final trained weights
"""
import argparse
import time
import os
import torch
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import UNet
from diffusion import GaussianDiffusion
from data import get_dataset


def save_grid(imgs, path, nrow=8, upscale=8):
    """imgs: (N, 1, H, W) tensor in [-1, 1]. Saves a matplotlib grid image."""
    imgs = ((imgs.clamp(-1, 1) + 1) / 2).cpu().numpy()  # -> [0, 1]
    n = imgs.shape[0]
    ncol = nrow
    nrow_actual = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow_actual, ncol, figsize=(ncol * 1.2, nrow_actual * 1.2))
    axes = np.array(axes).reshape(-1)
    for i, ax in enumerate(axes):
        ax.axis("off")
        if i < n:
            ax.imshow(imgs[i, 0], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    plt.tight_layout(pad=0.3)
    plt.savefig(path, dpi=120)
    plt.close(fig)


def save_trajectory(trajectory, path):
    """trajectory: list of (N,1,H,W) tensors across reverse diffusion steps, N>=1. Show sample 0."""
    imgs = [((t[0, 0].clamp(-1, 1) + 1) / 2).cpu().numpy() for t in trajectory]
    n = len(imgs)
    fig, axes = plt.subplots(1, n, figsize=(n * 1.2, 1.4))
    for i, ax in enumerate(axes):
        ax.axis("off")
        ax.imshow(imgs[i], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(f"{i}", fontsize=7)
    plt.tight_layout(pad=0.3)
    plt.savefig(path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--timesteps", type=int, default=200)
    parser.add_argument("--img_size", type=int, default=16)
    parser.add_argument("--out_dir", type=str, default="outputs")
    parser.add_argument("--checkpoint_every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "samples"), exist_ok=True)

    device = "cpu"
    data = get_dataset(img_size=args.img_size).to(device)
    n = data.shape[0]
    print(f"Dataset: {n} images, shape {tuple(data.shape[1:])}")

    model = UNet(img_channels=1, base_ch=32, time_dim=128).to(device)
    diffusion = GaussianDiffusion(timesteps=args.timesteps, device=device)
    opt = optim.Adam(model.parameters(), lr=args.lr)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    losses = []
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        perm = torch.randperm(n)
        epoch_losses = []
        for i in range(0, n, args.batch_size):
            idx = perm[i:i + args.batch_size]
            x0 = data[idx]
            loss = diffusion.training_loss(model, x0)

            opt.zero_grad()
            loss.backward()
            opt.step()

            epoch_losses.append(loss.item())
        mean_loss = float(np.mean(epoch_losses))
        losses.append(mean_loss)

        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.time() - t_start
            print(f"epoch {epoch:4d}/{args.epochs}  loss {mean_loss:.4f}  elapsed {elapsed:.1f}s")

        if epoch % args.checkpoint_every == 0 or epoch == args.epochs:
            model.eval()
            samples = diffusion.sample(model, (16, 1, args.img_size, args.img_size))
            save_grid(samples, os.path.join(args.out_dir, "samples", f"epoch_{epoch:04d}.png"), nrow=4)
            model.train()

    # Final: denoising trajectory visualization + loss curve + checkpoint
    model.eval()
    _, traj = diffusion.sample(model, (1, 1, args.img_size, args.img_size), return_trajectory=True)
    save_trajectory(traj, os.path.join(args.out_dir, "samples", "denoising_trajectory.png"))

    plt.figure(figsize=(6, 4))
    plt.plot(losses)
    plt.xlabel("epoch")
    plt.ylabel("training loss (simplified ELBO, MSE on noise)")
    plt.title("DDPM training loss")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "loss_curve.png"), dpi=120)
    plt.close()

    torch.save(model.state_dict(), os.path.join(args.out_dir, "model.pt"))
    np.save(os.path.join(args.out_dir, "losses.npy"), np.array(losses))

    total_time = time.time() - t_start
    print(f"Done. Total training time: {total_time:.1f}s. Final loss: {losses[-1]:.4f}")


if __name__ == "__main__":
    main()

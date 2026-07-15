"""
Small UNet noise-prediction network for DDPM (Ho et al. 2020), Section 3 / Appendix B.
Architecture: sinusoidal timestep embedding -> MLP, injected into GroupNorm/SiLU/Conv
residual blocks, with a downsample path, a bottleneck (with self-attention), and a
symmetric upsample path with skip connections -- the same recipe as the paper's UNet,
just narrower (fewer channels, fewer resolution levels) so it trains on a single CPU core.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_embedding(timesteps, dim):
    """Transformer-style sinusoidal position embedding, applied to diffusion timestep t."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class TimeMLP(nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        self.dim = dim
        self.net = nn.Sequential(
            nn.Linear(dim, out_dim), nn.SiLU(), nn.Linear(out_dim, out_dim)
        )

    def forward(self, t):
        return self.net(sinusoidal_embedding(t, self.dim))


class ResBlock(nn.Module):
    """GroupNorm -> SiLU -> Conv, twice, with the timestep embedding added in between."""

    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        groups = min(8, in_ch)
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    """Single-head spatial self-attention, used at the bottleneck (small feature map)."""

    def __init__(self, ch):
        super().__init__()
        self.norm = nn.GroupNorm(min(8, ch), ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = q.reshape(b, c, h * w).permute(0, 2, 1)
        k = k.reshape(b, c, h * w)
        v = v.reshape(b, c, h * w).permute(0, 2, 1)
        attn = torch.softmax(q @ k / math.sqrt(c), dim=-1)
        out = (attn @ v).permute(0, 2, 1).reshape(b, c, h, w)
        return x + self.proj(out)


class UNet(nn.Module):
    def __init__(self, img_channels=1, base_ch=32, time_dim=128):
        super().__init__()
        self.time_mlp = TimeMLP(time_dim, time_dim)

        self.in_conv = nn.Conv2d(img_channels, base_ch, 3, padding=1)

        # Down path: 16x16 -> 8x8 -> 4x4
        self.down1 = ResBlock(base_ch, base_ch, time_dim)
        self.down1_pool = nn.Conv2d(base_ch, base_ch, 4, stride=2, padding=1)  # 16->8
        self.down2 = ResBlock(base_ch, base_ch * 2, time_dim)
        self.down2_pool = nn.Conv2d(base_ch * 2, base_ch * 2, 4, stride=2, padding=1)  # 8->4

        # Bottleneck at 4x4
        self.mid1 = ResBlock(base_ch * 2, base_ch * 2, time_dim)
        self.mid_attn = SelfAttention(base_ch * 2)
        self.mid2 = ResBlock(base_ch * 2, base_ch * 2, time_dim)

        # Up path: 4x4 -> 8x8 -> 16x16 (with skip connections)
        self.up2_upsample = nn.ConvTranspose2d(base_ch * 2, base_ch * 2, 4, stride=2, padding=1)  # 4->8
        self.up2 = ResBlock(base_ch * 2 + base_ch * 2, base_ch, time_dim)
        self.up1_upsample = nn.ConvTranspose2d(base_ch, base_ch, 4, stride=2, padding=1)  # 8->16
        self.up1 = ResBlock(base_ch + base_ch, base_ch, time_dim)

        self.out_norm = nn.GroupNorm(min(8, base_ch), base_ch)
        self.out_conv = nn.Conv2d(base_ch, img_channels, 3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)

        x0 = self.in_conv(x)              # 16x16, base_ch
        d1 = self.down1(x0, t_emb)         # 16x16, base_ch
        d1p = self.down1_pool(d1)          # 8x8,  base_ch
        d2 = self.down2(d1p, t_emb)        # 8x8,  base_ch*2
        d2p = self.down2_pool(d2)          # 4x4,  base_ch*2

        m = self.mid1(d2p, t_emb)
        m = self.mid_attn(m)
        m = self.mid2(m, t_emb)            # 4x4, base_ch*2

        u2 = self.up2_upsample(m)          # 8x8, base_ch*2
        u2 = self.up2(torch.cat([u2, d2], dim=1), t_emb)  # 8x8, base_ch
        u1 = self.up1_upsample(u2)         # 16x16, base_ch
        u1 = self.up1(torch.cat([u1, d1], dim=1), t_emb)  # 16x16, base_ch

        out = self.out_conv(F.silu(self.out_norm(u1)))
        return out

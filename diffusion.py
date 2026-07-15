"""
Core DDPM math (Ho, Jain, Abbeel 2020).

Forward process:      q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) x_0, (1 - alpha_bar_t) I)
Training objective:    L_simple = E_{t,x_0,eps}[ || eps - eps_theta(x_t, t) ||^2 ]   (Eq. 14)
Reverse process step:  x_{t-1} = 1/sqrt(alpha_t) * (x_t - beta_t/sqrt(1-alpha_bar_t) * eps_theta)
                                  + sigma_t * z                                       (Eq. 11)
with sigma_t^2 = beta_t (the paper's simpler choice, matching sigma_t^2 = beta~_t at t=1).
"""
import torch


class GaussianDiffusion:
    def __init__(self, timesteps=200, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.T = timesteps
        self.device = device

        betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars
        self.sqrt_alpha_bars = torch.sqrt(alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)

    def q_sample(self, x0, t, noise=None):
        """Sample x_t ~ q(x_t | x_0) in closed form (the reparameterization trick)."""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab = self.sqrt_alpha_bars[t][:, None, None, None]
        sqrt_1m_ab = self.sqrt_one_minus_alpha_bars[t][:, None, None, None]
        return sqrt_ab * x0 + sqrt_1m_ab * noise, noise

    def training_loss(self, model, x0):
        """One Monte-Carlo estimate of L_simple for a batch of clean images x0."""
        b = x0.shape[0]
        t = torch.randint(0, self.T, (b,), device=x0.device).long()
        x_t, noise = self.q_sample(x0, t)
        pred_noise = model(x_t, t)
        return torch.nn.functional.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def sample(self, model, shape, return_trajectory=False):
        """Ancestral sampling: start from pure noise, iteratively denoise to x_0."""
        device = self.device
        x = torch.randn(shape, device=device)
        trajectory = [x.clone()] if return_trajectory else None

        for t_idx in reversed(range(self.T)):
            t = torch.full((shape[0],), t_idx, device=device, dtype=torch.long)
            pred_noise = model(x, t)

            alpha_t = self.alphas[t_idx]
            alpha_bar_t = self.alpha_bars[t_idx]
            beta_t = self.betas[t_idx]

            mean = (1.0 / torch.sqrt(alpha_t)) * (
                x - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * pred_noise
            )
            if t_idx > 0:
                noise = torch.randn_like(x)
                x = mean + torch.sqrt(beta_t) * noise
            else:
                x = mean  # no noise added on the final step

            if return_trajectory and t_idx % max(1, self.T // 10) == 0:
                trajectory.append(x.clone())

        if return_trajectory:
            return x, trajectory
        return x

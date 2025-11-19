"""
Volume Rendering Engine

Implements differentiable ray marching for X-ray CT forward model.

Based on the Beer-Lambert law:
    I = I₀ exp(-∫ μ(s) ds)

Which in expectation form (volume rendering equation):
    C(r) = ∫ T(t) · σ(r(t)) dt
    T(t) = exp(-∫₀ᵗ σ(s) ds)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict
import numpy as np


class VolumeRenderer(nn.Module):
    """
    Differentiable volume renderer for X-ray CT.

    Implements ray marching with Beer-Lambert attenuation.

    Args:
        num_samples: Number of samples per ray (default: 128)
        near: Near clipping plane (default: 0.0)
        far: Far clipping plane (default: 1.0)
        perturb: Add random perturbation to sample points (default: True)
        raw_noise_std: Std of noise added to raw density (for regularization, default: 0.0)
        white_background: Use white background (default: False)
    """

    def __init__(
        self,
        num_samples: int = 128,
        near: float = 0.0,
        far: float = 1.0,
        perturb: bool = True,
        raw_noise_std: float = 0.0,
        white_background: bool = False,
    ):
        super().__init__()

        self.num_samples = num_samples
        self.near = near
        self.far = far
        self.perturb = perturb
        self.raw_noise_std = raw_noise_std
        self.white_background = white_background

    def sample_along_rays(
        self,
        ray_origins: torch.Tensor,
        ray_directions: torch.Tensor,
        near: Optional[float] = None,
        far: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample points along rays.

        Args:
            ray_origins: Ray origins [B, 3]
            ray_directions: Ray directions (unit vectors) [B, 3]
            near: Near plane (if None, use self.near)
            far: Far plane (if None, use self.far)

        Returns:
            sample_points: Sampled 3D points [B, N, 3]
            t_vals: Sample depths [B, N]
        """
        B = ray_origins.shape[0]
        device = ray_origins.device

        near = near if near is not None else self.near
        far = far if far is not None else self.far

        # Create evenly-spaced samples
        t_vals = torch.linspace(near, far, self.num_samples, device=device)
        t_vals = t_vals.expand(B, self.num_samples)

        # Add random perturbation (stratified sampling)
        if self.perturb and self.training:
            # Get intervals between samples
            mids = 0.5 * (t_vals[..., 1:] + t_vals[..., :-1])
            upper = torch.cat([mids, t_vals[..., -1:]], dim=-1)
            lower = torch.cat([t_vals[..., :1], mids], dim=-1)

            # Uniform random samples in [0, 1)
            perturb_rand = torch.rand(t_vals.shape, device=device)

            # Perturbed sample locations
            t_vals = lower + (upper - lower) * perturb_rand

        # Compute 3D points: x = o + t * d
        sample_points = ray_origins.unsqueeze(1) + t_vals.unsqueeze(2) * ray_directions.unsqueeze(1)

        return sample_points, t_vals

    def render_rays(
        self,
        neural_field,
        ray_origins: torch.Tensor,
        ray_directions: torch.Tensor,
        near: Optional[float] = None,
        far: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Render rays through the neural field.

        Args:
            neural_field: NeuralField instance
            ray_origins: Ray origins [B, 3]
            ray_directions: Ray directions [B, 3]
            near: Near clipping plane
            far: Far clipping plane

        Returns:
            outputs: Dictionary containing:
                - "intensity": Rendered X-ray intensity [B]
                - "transmittance": Transmittance map [B]
                - "depth": Expected depth [B]
                - "weights": Rendering weights [B, N]
                - "density": Sampled densities [B, N]
        """
        # Sample points along rays
        sample_points, t_vals = self.sample_along_rays(
            ray_origins, ray_directions, near, far
        )

        # Query neural field
        B, N, _ = sample_points.shape
        densities = neural_field(sample_points.reshape(B * N, 3))
        densities = densities.reshape(B, N)

        # Add noise to densities for regularization (if training)
        if self.training and self.raw_noise_std > 0:
            noise = torch.randn_like(densities) * self.raw_noise_std
            densities = densities + noise

        # Compute distances between samples
        dists = t_vals[..., 1:] - t_vals[..., :-1]
        # Add large distance for last sample (to infinity)
        dists = torch.cat([dists, torch.full_like(dists[..., :1], 1e10)], dim=-1)

        # Compute alpha (opacity) values
        # α = 1 - exp(-σ·δ)
        alpha = 1.0 - torch.exp(-densities * dists)

        # Compute transmittance (accumulated transparency)
        # T_i = exp(-Σⱼ₌₁^(i-1) σⱼ·δⱼ) = Π (1 - α_j)
        # Use cumprod for numerical stability
        transmittance = torch.cumprod(
            torch.cat([torch.ones((B, 1), device=densities.device), 1.0 - alpha + 1e-10], dim=-1),
            dim=-1,
        )[:, :-1]

        # Compute rendering weights
        # w_i = T_i · α_i
        weights = transmittance * alpha

        # Render X-ray intensity (expected attenuation)
        # For X-ray CT, we're interested in line integrals
        # I = exp(-∫ μ ds) ≈ exp(-Σ σ_i · δ_i)
        line_integral = torch.sum(densities * dists, dim=-1)
        intensity = torch.exp(-line_integral)

        # Alternatively, use weighted sum (used in NeRF)
        # This is equivalent for small densities
        # intensity_alt = torch.sum(weights * densities, dim=-1)

        # Compute expected depth
        depth = torch.sum(weights * t_vals, dim=-1)

        # Final transmittance (transparency at the end of ray)
        final_transmittance = transmittance[:, -1]

        # Handle white/black background
        if self.white_background:
            intensity = intensity + (1.0 - final_transmittance)

        return {
            "intensity": intensity,
            "transmittance": final_transmittance,
            "depth": depth,
            "weights": weights,
            "density": densities,
            "t_vals": t_vals,
        }

    def render_batch(
        self,
        neural_field,
        rays: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Render a batch of rays.

        Args:
            neural_field: NeuralField instance
            rays: Dictionary containing:
                - "origins": Ray origins [B, 3]
                - "directions": Ray directions [B, 3]
                - "near": (optional) Near planes [B] or scalar
                - "far": (optional) Far planes [B] or scalar

        Returns:
            outputs: Rendering outputs
        """
        origins = rays["origins"]
        directions = rays["directions"]
        near = rays.get("near", None)
        far = rays.get("far", None)

        # Handle per-ray near/far if provided
        if near is not None and isinstance(near, torch.Tensor):
            near = near.item() if near.numel() == 1 else None
        if far is not None and isinstance(far, torch.Tensor):
            far = far.item() if far.numel() == 1 else None

        return self.render_rays(neural_field, origins, directions, near, far)

    def extra_repr(self) -> str:
        """String representation."""
        return (
            f"num_samples={self.num_samples}, "
            f"near={self.near}, far={self.far}, "
            f"perturb={self.perturb}"
        )


class ImportanceSampler(nn.Module):
    """
    Two-stage hierarchical sampling (coarse + fine).

    First samples uniformly, then importance samples based on density.

    Args:
        num_coarse_samples: Number of coarse samples (default: 64)
        num_fine_samples: Number of fine samples (default: 128)
        near: Near clipping plane
        far: Far clipping plane
    """

    def __init__(
        self,
        num_coarse_samples: int = 64,
        num_fine_samples: int = 128,
        near: float = 0.0,
        far: float = 1.0,
    ):
        super().__init__()

        self.num_coarse_samples = num_coarse_samples
        self.num_fine_samples = num_fine_samples
        self.near = near
        self.far = far

        # Create renderers
        self.coarse_renderer = VolumeRenderer(
            num_samples=num_coarse_samples,
            near=near,
            far=far,
        )

        self.fine_renderer = VolumeRenderer(
            num_samples=num_fine_samples,
            near=near,
            far=far,
            perturb=False,  # Don't perturb fine samples
        )

    def sample_pdf(
        self,
        bins: torch.Tensor,
        weights: torch.Tensor,
        num_samples: int,
    ) -> torch.Tensor:
        """
        Sample from probability distribution defined by weights.

        Args:
            bins: Bin edges [B, N+1]
            weights: Probability weights [B, N]
            num_samples: Number of samples to draw

        Returns:
            samples: Sampled t values [B, num_samples]
        """
        # Normalize weights to get PDF
        weights = weights + 1e-5  # Prevent division by zero
        pdf = weights / torch.sum(weights, dim=-1, keepdim=True)

        # Compute CDF
        cdf = torch.cumsum(pdf, dim=-1)
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)

        # Take uniform samples
        u = torch.rand(
            list(cdf.shape[:-1]) + [num_samples],
            device=weights.device,
        )

        # Invert CDF
        indices = torch.searchsorted(cdf, u, right=True)
        below = torch.clamp(indices - 1, min=0)
        above = torch.clamp(indices, max=cdf.shape[-1] - 1)

        # Interpolate
        indices_g = torch.stack([below, above], dim=-1)
        cdf_g = torch.gather(cdf.unsqueeze(-2).expand(*indices_g.shape[:-1], -1), -1, indices_g)
        bins_g = torch.gather(bins.unsqueeze(-2).expand(*indices_g.shape[:-1], -1), -1, indices_g)

        denom = cdf_g[..., 1] - cdf_g[..., 0]
        denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
        t = (u - cdf_g[..., 0]) / denom

        samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

        return samples

    def render_rays(
        self,
        neural_field,
        ray_origins: torch.Tensor,
        ray_directions: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Render rays with hierarchical sampling.

        Args:
            neural_field: NeuralField instance
            ray_origins: Ray origins [B, 3]
            ray_directions: Ray directions [B, 3]

        Returns:
            coarse_outputs: Coarse rendering outputs
            fine_outputs: Fine rendering outputs
        """
        # Coarse pass
        coarse_outputs = self.coarse_renderer.render_rays(
            neural_field, ray_origins, ray_directions
        )

        # Importance sampling for fine pass
        weights = coarse_outputs["weights"]
        t_vals = coarse_outputs["t_vals"]

        # Get bin edges (midpoints between t_vals)
        t_mids = 0.5 * (t_vals[..., 1:] + t_vals[..., :-1])

        # Sample new t values based on weights
        t_fine = self.sample_pdf(t_vals, weights[..., 1:-1], self.num_fine_samples)

        # Combine coarse and fine samples
        t_combined, _ = torch.sort(torch.cat([t_vals, t_fine], dim=-1), dim=-1)

        # Compute sample points
        sample_points = ray_origins.unsqueeze(1) + t_combined.unsqueeze(2) * ray_directions.unsqueeze(1)

        # Query neural field
        B, N, _ = sample_points.shape
        densities = neural_field(sample_points.reshape(B * N, 3))
        densities = densities.reshape(B, N)

        # Render with combined samples
        dists = t_combined[..., 1:] - t_combined[..., :-1]
        dists = torch.cat([dists, torch.full_like(dists[..., :1], 1e10)], dim=-1)

        alpha = 1.0 - torch.exp(-densities * dists)
        transmittance = torch.cumprod(
            torch.cat([torch.ones((B, 1), device=densities.device), 1.0 - alpha + 1e-10], dim=-1),
            dim=-1,
        )[:, :-1]
        weights_fine = transmittance * alpha

        line_integral = torch.sum(densities * dists, dim=-1)
        intensity = torch.exp(-line_integral)
        depth = torch.sum(weights_fine * t_combined, dim=-1)

        fine_outputs = {
            "intensity": intensity,
            "transmittance": transmittance[:, -1],
            "depth": depth,
            "weights": weights_fine,
            "density": densities,
            "t_vals": t_combined,
        }

        return coarse_outputs, fine_outputs


def render_image(
    neural_field,
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    renderer: VolumeRenderer,
    batch_size: int = 8192,
) -> torch.Tensor:
    """
    Render a full image by batching rays.

    Args:
        neural_field: NeuralField instance
        ray_origins: All ray origins [H*W, 3]
        ray_directions: All ray directions [H*W, 3]
        renderer: VolumeRenderer instance
        batch_size: Batch size for ray rendering

    Returns:
        image: Rendered intensity values [H*W]
    """
    num_rays = ray_origins.shape[0]
    intensities = []

    neural_field.eval()
    with torch.no_grad():
        for i in range(0, num_rays, batch_size):
            batch_origins = ray_origins[i:i + batch_size]
            batch_directions = ray_directions[i:i + batch_size]

            outputs = renderer.render_rays(
                neural_field, batch_origins, batch_directions
            )

            intensities.append(outputs["intensity"].cpu())

    return torch.cat(intensities, dim=0)


def compute_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """
    Compute Peak Signal-to-Noise Ratio.

    PSNR = 20 · log10(MAX) - 10 · log10(MSE)

    Args:
        pred: Predicted values
        target: Ground truth values
        max_val: Maximum possible value

    Returns:
        psnr: PSNR in dB
    """
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float("inf")
    psnr = 20 * torch.log10(torch.tensor(max_val)) - 10 * torch.log10(mse)
    return psnr.item()

"""
Loss Functions for Multi-Modal X-Ray CT Reconstruction

Implements physics-informed loss functions combining:
1. Data fidelity (reconstruction error)
2. Regularization (TV, smoothness)
3. Physics constraints
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import numpy as np


class MultiModalLoss(nn.Module):
    """
    Multi-modal loss function for joint laminography-tomography optimization.

    L_total = α·L_lamino + β·L_tomo + γ·L_reg + δ·L_physics

    where:
        L_lamino: Laminography reconstruction error
        L_tomo: Tomography reconstruction error
        L_reg: Regularization (TV + sparsity)
        L_physics: Physics-informed constraints

    Args:
        alpha_lamino: Weight for laminography loss (default: 1.0)
        beta_tomo: Weight for tomography loss (default: 0.5)
        gamma_tv: Weight for total variation (default: 0.01)
        lambda_smooth: Weight for smoothness (default: 0.001)
        lambda_sparse: Weight for L1 sparsity (default: 0.0)
        loss_type: "mse" or "l1" (default: "mse")
    """

    def __init__(
        self,
        alpha_lamino: float = 1.0,
        beta_tomo: float = 0.5,
        gamma_tv: float = 0.01,
        lambda_smooth: float = 0.001,
        lambda_sparse: float = 0.0,
        loss_type: str = "mse",
    ):
        super().__init__()

        self.alpha_lamino = alpha_lamino
        self.beta_tomo = beta_tomo
        self.gamma_tv = gamma_tv
        self.lambda_smooth = lambda_smooth
        self.lambda_sparse = lambda_sparse
        self.loss_type = loss_type

    def data_fidelity_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute data fidelity loss.

        Args:
            pred: Predicted intensities [B]
            target: Target intensities [B]

        Returns:
            loss: Data fidelity loss
        """
        if self.loss_type == "mse":
            return F.mse_loss(pred, target)
        elif self.loss_type == "l1":
            return F.l1_loss(pred, target)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

    def forward(
        self,
        pred_lamino: Optional[torch.Tensor],
        target_lamino: Optional[torch.Tensor],
        pred_tomo: Optional[torch.Tensor],
        target_tomo: Optional[torch.Tensor],
        neural_field: Optional[nn.Module] = None,
        compute_regularization: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss.

        Args:
            pred_lamino: Predicted laminography intensities
            target_lamino: Target laminography intensities
            pred_tomo: Predicted tomography intensities
            target_tomo: Target tomography intensities
            neural_field: NeuralField module (for regularization)
            compute_regularization: Whether to compute regularization terms

        Returns:
            losses: Dictionary of loss components
        """
        losses = {}

        # Data fidelity losses
        loss_total = 0.0

        if pred_lamino is not None and target_lamino is not None:
            loss_lamino = self.data_fidelity_loss(pred_lamino, target_lamino)
            losses["loss_lamino"] = loss_lamino
            loss_total += self.alpha_lamino * loss_lamino

        if pred_tomo is not None and target_tomo is not None:
            loss_tomo = self.data_fidelity_loss(pred_tomo, target_tomo)
            losses["loss_tomo"] = loss_tomo
            loss_total += self.beta_tomo * loss_tomo

        # Regularization losses
        if compute_regularization and neural_field is not None:
            if self.gamma_tv > 0:
                loss_tv = neural_field.total_variation()
                losses["loss_tv"] = loss_tv
                loss_total += self.gamma_tv * loss_tv

            if self.lambda_smooth > 0:
                loss_smooth = neural_field.smooth_regularization()
                losses["loss_smooth"] = loss_smooth
                loss_total += self.lambda_smooth * loss_smooth

            if self.lambda_sparse > 0:
                loss_sparse = neural_field.sparsity_regularization()
                losses["loss_sparse"] = loss_sparse
                loss_total += self.lambda_sparse * loss_sparse

        losses["loss_total"] = loss_total

        return losses

    def extra_repr(self) -> str:
        """String representation."""
        return (
            f"alpha_lamino={self.alpha_lamino}, "
            f"beta_tomo={self.beta_tomo}, "
            f"gamma_tv={self.gamma_tv}, "
            f"lambda_smooth={self.lambda_smooth}, "
            f"lambda_sparse={self.lambda_sparse}"
        )


def compute_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """
    Compute Peak Signal-to-Noise Ratio.

    PSNR = 20 · log₁₀(MAX / √MSE)

    Args:
        pred: Predicted values
        target: Ground truth values
        max_val: Maximum possible value (default: 1.0)

    Returns:
        psnr: PSNR in dB
    """
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float("inf")

    psnr = 20 * torch.log10(torch.tensor(max_val) / torch.sqrt(mse))
    return psnr.item()


def compute_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    max_val: float = 1.0,
) -> float:
    """
    Compute Structural Similarity Index Measure.

    SSIM measures perceived image quality.

    Args:
        pred: Predicted values [H, W] or [B, H, W]
        target: Ground truth values [H, W] or [B, H, W]
        window_size: Size of Gaussian window (default: 11)
        max_val: Maximum possible value

    Returns:
        ssim: SSIM value (higher is better, max 1.0)
    """
    # This is a simplified implementation
    # For production, use pytorch-msssim or torchmetrics

    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2

    # Ensure 3D tensors
    if pred.ndim == 2:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)

    # Compute means
    mu_pred = pred.mean()
    mu_target = target.mean()

    # Compute variances and covariance
    sigma_pred = pred.var()
    sigma_target = target.var()
    sigma_pred_target = ((pred - mu_pred) * (target - mu_target)).mean()

    # SSIM formula
    ssim = (
        (2 * mu_pred * mu_target + C1)
        * (2 * sigma_pred_target + C2)
        / ((mu_pred ** 2 + mu_target ** 2 + C1) * (sigma_pred + sigma_target + C2))
    )

    return ssim.item()


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using pre-trained VGG features.

    Useful for Stage 2 (diffusion refinement).

    Args:
        feature_layers: Layers to extract features from
    """

    def __init__(self, feature_layers: Optional[list] = None):
        super().__init__()

        try:
            import torchvision.models as models
            vgg = models.vgg16(pretrained=True).features
            self.feature_extractor = vgg.eval()

            # Freeze parameters
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

            if feature_layers is None:
                feature_layers = [3, 8, 15, 22]  # ReLU layers

            self.feature_layers = feature_layers

        except ImportError:
            print("Warning: torchvision not available. Perceptual loss disabled.")
            self.feature_extractor = None

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute perceptual loss.

        Args:
            pred: Predicted images [B, C, H, W]
            target: Target images [B, C, H, W]

        Returns:
            loss: Perceptual loss
        """
        if self.feature_extractor is None:
            return torch.tensor(0.0, device=pred.device)

        # Extract features
        loss = 0.0
        x_pred = pred
        x_target = target

        for i, layer in enumerate(self.feature_extractor):
            x_pred = layer(x_pred)
            x_target = layer(x_target)

            if i in self.feature_layers:
                loss += F.mse_loss(x_pred, x_target)

        return loss


class GradientLoss(nn.Module):
    """
    Gradient loss for edge preservation.

    L_grad = ||∇pred - ∇target||
    """

    def __init__(self, loss_type: str = "l1"):
        super().__init__()
        self.loss_type = loss_type

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute gradient loss.

        Args:
            pred: Predicted image [B, C, H, W]
            target: Target image [B, C, H, W]

        Returns:
            loss: Gradient loss
        """
        # Compute gradients using Sobel operators
        pred_grad_x = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        pred_grad_y = pred[:, :, 1:, :] - pred[:, :, :-1, :]

        target_grad_x = target[:, :, :, 1:] - target[:, :, :, :-1]
        target_grad_y = target[:, :, 1:, :] - target[:, :, :-1, :]

        if self.loss_type == "l1":
            loss_x = F.l1_loss(pred_grad_x, target_grad_x)
            loss_y = F.l1_loss(pred_grad_y, target_grad_y)
        else:  # mse
            loss_x = F.mse_loss(pred_grad_x, target_grad_x)
            loss_y = F.mse_loss(pred_grad_y, target_grad_y)

        return loss_x + loss_y


class ArtifactDiscriminatorLoss(nn.Module):
    """
    Adversarial loss to distinguish artifacts from real structure.

    Useful for learning what constitutes an artifact vs real feature.
    """

    def __init__(self, discriminator: nn.Module):
        super().__init__()
        self.discriminator = discriminator

    def forward(
        self,
        pred: torch.Tensor,
        is_real: bool = False,
    ) -> torch.Tensor:
        """
        Compute discriminator loss.

        Args:
            pred: Predicted images
            is_real: Whether this is real (artifact-free) or fake (with artifacts)

        Returns:
            loss: Discriminator loss
        """
        validity = self.discriminator(pred)

        if is_real:
            target = torch.ones_like(validity)
        else:
            target = torch.zeros_like(validity)

        loss = F.binary_cross_entropy_with_logits(validity, target)

        return loss


def create_loss_function(config: Dict) -> nn.Module:
    """
    Factory function to create loss function from config.

    Args:
        config: Loss configuration dictionary

    Returns:
        loss_fn: Loss function module
    """
    loss_type = config.get("type", "multimodal")

    if loss_type == "multimodal":
        return MultiModalLoss(
            alpha_lamino=config.get("alpha_lamino", 1.0),
            beta_tomo=config.get("beta_tomo", 0.5),
            gamma_tv=config.get("gamma_tv", 0.01),
            lambda_smooth=config.get("lambda_smooth", 0.001),
            lambda_sparse=config.get("lambda_sparse", 0.0),
            loss_type=config.get("data_loss_type", "mse"),
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

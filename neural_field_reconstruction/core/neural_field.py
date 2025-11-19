"""
Neural Field Implementation

Combines hash encoding with MLP to create a continuous 3D implicit representation
of the attenuation coefficient field μ(x).
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import numpy as np

from .hash_encoding import HashEncoding, PositionalEncoding, create_encoding
from .mlp import FullyConnectedNetwork, create_mlp, count_parameters

try:
    import tinycudann as tcnn
    TCNN_AVAILABLE = True
except ImportError:
    TCNN_AVAILABLE = False


class NeuralField(nn.Module):
    """
    Neural implicit representation of 3D attenuation field.

    Architecture:
        x (3D coords) → Hash Encoding → MLP → μ (density)

    The network learns a continuous function:
        μ: ℝ³ → ℝ₊

    that represents the X-ray attenuation coefficient at any 3D point.

    Args:
        bounds: Bounding box [[x_min, y_min, z_min], [x_max, y_max, z_max]]
        encoding_config: Configuration dict for hash encoding
        mlp_config: Configuration dict for MLP
        use_tcnn: Use tiny-cuda-nn acceleration (default: True)
    """

    def __init__(
        self,
        bounds: np.ndarray = np.array([[-1, -1, -1], [1, 1, 1]]),
        encoding_config: Optional[Dict[str, Any]] = None,
        mlp_config: Optional[Dict[str, Any]] = None,
        use_tcnn: bool = True,
    ):
        super().__init__()

        # Store bounding box
        self.register_buffer("bounds_min", torch.tensor(bounds[0], dtype=torch.float32))
        self.register_buffer("bounds_max", torch.tensor(bounds[1], dtype=torch.float32))

        # Default configurations
        if encoding_config is None:
            encoding_config = {
                "n_levels": 16,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                "base_resolution": 16,
                "finest_resolution": 2048,
                "use_tcnn": use_tcnn,
            }

        if mlp_config is None:
            mlp_config = {
                "hidden_dims": [64, 64, 64],
                "activation": "relu",
                "output_activation": "relu",  # Ensure positive density
                "use_tcnn": use_tcnn,
            }

        # Create encoding
        self.encoding = HashEncoding(**encoding_config)
        encoding_dim = self.encoding.get_output_dim()

        # Create MLP
        self.mlp = FullyConnectedNetwork(
            input_dim=encoding_dim,
            output_dim=1,  # Single density output
            **mlp_config,
        )

        # Use combined tcnn module for maximum speed
        self.use_tcnn = use_tcnn and TCNN_AVAILABLE
        if self.use_tcnn:
            self._init_combined_tcnn(encoding_config, mlp_config)

        # Cache for statistics
        self._stats = {}

    def _init_combined_tcnn(self, encoding_config: Dict, mlp_config: Dict):
        """
        Initialize combined tiny-cuda-nn module (encoding + MLP).

        This is faster than separate modules due to kernel fusion.
        """
        # Create combined config
        encoding_cfg = {
            "otype": "HashGrid",
            "n_levels": encoding_config["n_levels"],
            "n_features_per_level": encoding_config["n_features_per_level"],
            "log2_hashmap_size": encoding_config["log2_hashmap_size"],
            "base_resolution": encoding_config["base_resolution"],
            "per_level_scale": np.exp(
                np.log(encoding_config["finest_resolution"] / encoding_config["base_resolution"])
                / (encoding_config["n_levels"] - 1)
            ),
        }

        network_cfg = {
            "otype": "FullyFusedMLP",
            "activation": mlp_config.get("activation", "ReLU").upper(),
            "output_activation": mlp_config.get("output_activation", "ReLU").upper(),
            "n_neurons": mlp_config.get("hidden_dims", [64])[0],
            "n_hidden_layers": len(mlp_config.get("hidden_dims", [64])),
        }

        try:
            self.combined_network = tcnn.NetworkWithInputEncoding(
                n_input_dims=3,
                n_output_dims=1,
                encoding_config=encoding_cfg,
                network_config=network_cfg,
            )
            self.use_combined = True
        except Exception as e:
            print(f"Warning: Could not create combined tcnn module: {e}")
            print("Falling back to separate encoding + MLP")
            self.use_combined = False

    def normalize_coords(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize coordinates from world space to [0, 1]³.

        Args:
            x: Coordinates in world space [B, 3]

        Returns:
            x_norm: Normalized coordinates [B, 3] in [0, 1]³
        """
        return (x - self.bounds_min) / (self.bounds_max - self.bounds_min)

    def denormalize_coords(self, x_norm: torch.Tensor) -> torch.Tensor:
        """
        Denormalize coordinates from [0, 1]³ to world space.

        Args:
            x_norm: Normalized coordinates [B, 3] in [0, 1]³

        Returns:
            x: Coordinates in world space [B, 3]
        """
        return x_norm * (self.bounds_max - self.bounds_min) + self.bounds_min

    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        """
        Query the neural field at 3D coordinates.

        Args:
            x: Coordinates in world space [B, 3] or [B, N, 3]
            return_features: If True, also return intermediate features

        Returns:
            density: Attenuation coefficient μ(x) [B] or [B, N]
            features: (optional) Intermediate features [B, D] or [B, N, D]
        """
        original_shape = x.shape

        # Flatten if needed
        if x.ndim == 3:
            B, N, _ = x.shape
            x = x.reshape(B * N, 3)

        # Normalize to [0, 1]³
        x_norm = self.normalize_coords(x)

        # Clamp to valid range
        x_norm = torch.clamp(x_norm, 0.0, 1.0)

        # Forward pass
        if self.use_tcnn and hasattr(self, "use_combined") and self.use_combined:
            # Use fused tcnn module (fastest)
            density = self.combined_network(x_norm)
        else:
            # Separate encoding + MLP
            features = self.encoding(x_norm)
            density = self.mlp(features)

        # Ensure positive density
        density = torch.relu(density)

        # Reshape if needed
        if len(original_shape) == 3:
            density = density.reshape(B, N, 1)
            if return_features:
                features = features.reshape(B, N, -1)

        # Squeeze last dimension
        density = density.squeeze(-1)

        if return_features:
            return density, features
        else:
            return density

    def density(self, x: torch.Tensor) -> torch.Tensor:
        """
        Alias for forward() for clarity.

        Args:
            x: Coordinates [B, 3]

        Returns:
            density: μ(x) [B]
        """
        return self.forward(x)

    def gradient(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient ∇μ(x) of the density field.

        Args:
            x: Coordinates [B, 3]

        Returns:
            grad: Gradient ∇μ [B, 3]
        """
        x = x.requires_grad_(True)
        density = self.forward(x)

        # Compute gradient using autograd
        grad = torch.autograd.grad(
            outputs=density,
            inputs=x,
            grad_outputs=torch.ones_like(density),
            create_graph=True,
            retain_graph=True,
        )[0]

        return grad

    def total_variation(self, num_samples: int = 8192) -> torch.Tensor:
        """
        Compute total variation regularization term.

        TV(μ) ≈ E[||∇μ(x)||]

        Args:
            num_samples: Number of random samples for Monte Carlo estimate

        Returns:
            tv: Total variation estimate
        """
        # Sample random points in domain
        x = torch.rand(num_samples, 3, device=self.bounds_min.device)
        x = self.denormalize_coords(x)

        # Compute gradients
        grad = self.gradient(x)

        # TV = mean of gradient magnitudes
        tv = torch.mean(torch.norm(grad, dim=-1))

        return tv

    def smooth_regularization(self, num_samples: int = 8192) -> torch.Tensor:
        """
        Compute smoothness regularization term.

        R_smooth(μ) ≈ E[||∇μ(x)||²]

        Args:
            num_samples: Number of random samples

        Returns:
            smooth: Smoothness estimate
        """
        # Sample random points
        x = torch.rand(num_samples, 3, device=self.bounds_min.device)
        x = self.denormalize_coords(x)

        # Compute gradients
        grad = self.gradient(x)

        # Smoothness = mean of squared gradient magnitudes
        smooth = torch.mean(torch.sum(grad ** 2, dim=-1))

        return smooth

    def sparsity_regularization(self, num_samples: int = 8192) -> torch.Tensor:
        """
        Compute L1 sparsity regularization.

        R_sparse(μ) ≈ E[|μ(x)|]

        Args:
            num_samples: Number of random samples

        Returns:
            sparse: L1 norm estimate
        """
        # Sample random points
        x = torch.rand(num_samples, 3, device=self.bounds_min.device)
        x = self.denormalize_coords(x)

        # Compute densities
        density = self.forward(x)

        # L1 norm
        sparse = torch.mean(torch.abs(density))

        return sparse

    def export_volume(
        self,
        resolution: int = 256,
        batch_size: int = 8192,
        device: Optional[torch.device] = None,
    ) -> np.ndarray:
        """
        Export the neural field to a dense 3D volume.

        Args:
            resolution: Volume resolution (creates resolution³ voxels)
            batch_size: Batch size for evaluation
            device: Device for computation

        Returns:
            volume: Dense volume array [resolution, resolution, resolution]
        """
        if device is None:
            device = self.bounds_min.device

        self.eval()

        # Create grid
        x = torch.linspace(0, 1, resolution, device=device)
        y = torch.linspace(0, 1, resolution, device=device)
        z = torch.linspace(0, 1, resolution, device=device)

        grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing="ij")
        coords = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3)

        # Convert to world space
        coords = self.denormalize_coords(coords)

        # Evaluate in batches
        densities = []
        with torch.no_grad():
            for i in range(0, coords.shape[0], batch_size):
                batch = coords[i:i + batch_size]
                density = self.forward(batch)
                densities.append(density.cpu())

        # Concatenate and reshape
        volume = torch.cat(densities, dim=0).reshape(resolution, resolution, resolution)

        return volume.numpy()

    def get_parameters_count(self) -> Dict[str, int]:
        """
        Get parameter count breakdown.

        Returns:
            counts: Dictionary of parameter counts
        """
        encoding_params = count_parameters(self.encoding)
        mlp_params = count_parameters(self.mlp)
        total_params = encoding_params + mlp_params

        return {
            "encoding": encoding_params,
            "mlp": mlp_params,
            "total": total_params,
        }

    def extra_repr(self) -> str:
        """String representation."""
        params = self.get_parameters_count()
        return (
            f"bounds={self.bounds_min.tolist()} to {self.bounds_max.tolist()}, "
            f"parameters={params['total']:,} "
            f"(encoding: {params['encoding']:,}, mlp: {params['mlp']:,})"
        )


def create_neural_field(config: Dict[str, Any]) -> NeuralField:
    """
    Factory function to create a neural field from config.

    Args:
        config: Configuration dictionary

    Returns:
        neural_field: NeuralField instance
    """
    return NeuralField(**config)

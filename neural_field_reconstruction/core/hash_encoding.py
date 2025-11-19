"""
Multi-Resolution Hash Encoding

Implements the InstantNGP-style hash encoding for efficient
neural field representation.

References:
    Müller et al., "Instant Neural Graphics Primitives", SIGGRAPH 2022
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple

try:
    import tinycudann as tcnn
    TCNN_AVAILABLE = True
except ImportError:
    TCNN_AVAILABLE = False
    print("WARNING: tinycudann not available. Using PyTorch fallback (slower).")


class HashEncoding(nn.Module):
    """
    Multi-resolution hash encoding for 3D coordinates.

    This module implements the hash encoding from InstantNGP, which uses
    multiple resolution levels with hash tables for feature storage.

    Mathematical formulation:
        γ(x) = [h_1(x), h_2(x), ..., h_L(x)]
        h_l(x) = lookup(⌊x · 2^l⌋ mod T_l)

    Args:
        n_levels: Number of resolution levels (default: 16)
        n_features_per_level: Features per hash entry (default: 2)
        log2_hashmap_size: Log2 of hash table size (default: 19, i.e., 2^19 entries)
        base_resolution: Coarsest resolution (default: 16)
        finest_resolution: Finest resolution (default: 2048)
        use_tcnn: Use tiny-cuda-nn if available (default: True)

    Input:
        x: Coordinates of shape [B, 3] in range [0, 1]³

    Output:
        features: Encoded features of shape [B, n_levels * n_features_per_level]
    """

    def __init__(
        self,
        n_levels: int = 16,
        n_features_per_level: int = 2,
        log2_hashmap_size: int = 19,
        base_resolution: int = 16,
        finest_resolution: int = 2048,
        use_tcnn: bool = True,
    ):
        super().__init__()

        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = base_resolution
        self.finest_resolution = finest_resolution

        # Compute per-level scale factor
        # Resolution at level l: base_resolution * per_level_scale^l
        self.per_level_scale = np.exp(
            np.log(finest_resolution / base_resolution) / (n_levels - 1)
        )

        self.use_tcnn = use_tcnn and TCNN_AVAILABLE

        if self.use_tcnn:
            self._init_tcnn()
        else:
            self._init_pytorch()

        # Output dimension
        self.out_dim = n_levels * n_features_per_level

    def _init_tcnn(self):
        """Initialize tiny-cuda-nn hash encoding (fast)."""
        config = {
            "otype": "HashGrid",
            "n_levels": self.n_levels,
            "n_features_per_level": self.n_features_per_level,
            "log2_hashmap_size": self.log2_hashmap_size,
            "base_resolution": self.base_resolution,
            "per_level_scale": float(self.per_level_scale),
        }

        self.encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config=config,
        )

    def _init_pytorch(self):
        """Initialize PyTorch hash encoding (fallback, slower)."""
        # Create hash tables for each level
        self.hash_tables = nn.ModuleList()

        for level in range(self.n_levels):
            resolution = int(self.base_resolution * (self.per_level_scale ** level))
            # Use smaller table for PyTorch version to save memory
            table_size = min(2 ** self.log2_hashmap_size, resolution ** 3)

            # Initialize hash table with small random values
            table = nn.Parameter(
                torch.randn(table_size, self.n_features_per_level) * 0.0001
            )
            self.hash_tables.append(table)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode 3D coordinates using multi-resolution hash encoding.

        Args:
            x: Coordinates of shape [B, 3] in range [0, 1]³

        Returns:
            features: Encoded features of shape [B, n_levels * n_features_per_level]
        """
        if self.use_tcnn:
            return self._forward_tcnn(x)
        else:
            return self._forward_pytorch(x)

    def _forward_tcnn(self, x: torch.Tensor) -> torch.Tensor:
        """Fast forward pass using tiny-cuda-nn."""
        return self.encoding(x)

    def _forward_pytorch(self, x: torch.Tensor) -> torch.Tensor:
        """
        PyTorch fallback implementation.

        This is significantly slower but doesn't require CUDA compilation.
        """
        B = x.shape[0]
        device = x.device

        features_list = []

        for level, table in enumerate(self.hash_tables):
            # Compute resolution at this level
            resolution = self.base_resolution * (self.per_level_scale ** level)

            # Scale coordinates to grid resolution
            x_scaled = x * resolution

            # Get integer grid coordinates
            x_floor = torch.floor(x_scaled).long()
            x_frac = x_scaled - x_floor.float()

            # Wrap to valid range
            x_floor = x_floor % int(resolution)

            # Trilinear interpolation
            # Get 8 corner points of the cube
            features_interp = torch.zeros(
                B, self.n_features_per_level, device=device, dtype=x.dtype
            )

            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        # Corner coordinates
                        corner = x_floor + torch.tensor(
                            [i, j, k], device=device, dtype=torch.long
                        )
                        corner = corner % int(resolution)

                        # Hash to table index
                        hash_idx = self._spatial_hash(
                            corner[:, 0], corner[:, 1], corner[:, 2], table.shape[0]
                        )

                        # Lookup features
                        corner_features = table[hash_idx]

                        # Interpolation weights
                        weight = (
                            (i * x_frac[:, 0] + (1 - i) * (1 - x_frac[:, 0]))
                            * (j * x_frac[:, 1] + (1 - j) * (1 - x_frac[:, 1]))
                            * (k * x_frac[:, 2] + (1 - k) * (1 - x_frac[:, 2]))
                        )

                        features_interp += corner_features * weight.unsqueeze(1)

            features_list.append(features_interp)

        # Concatenate all levels
        return torch.cat(features_list, dim=1)

    @staticmethod
    def _spatial_hash(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, table_size: int) -> torch.Tensor:
        """
        Spatial hash function using prime numbers.

        H(x,y,z) = (x * π₁ ⊕ y * π₂ ⊕ z * π₃) mod table_size
        """
        # Large primes
        π1 = 1
        π2 = 2654435761
        π3 = 805459861

        return ((x * π1) ^ (y * π2) ^ (z * π3)) % table_size

    def get_output_dim(self) -> int:
        """Return output dimension."""
        return self.out_dim

    def extra_repr(self) -> str:
        """String representation."""
        return (
            f"n_levels={self.n_levels}, "
            f"n_features_per_level={self.n_features_per_level}, "
            f"log2_hashmap_size={self.log2_hashmap_size}, "
            f"base_resolution={self.base_resolution}, "
            f"finest_resolution={self.finest_resolution}, "
            f"backend={'tcnn' if self.use_tcnn else 'pytorch'}"
        )


class PositionalEncoding(nn.Module):
    """
    Vanilla positional encoding (for comparison).

    γ(x) = [sin(2^0 π x), cos(2^0 π x), ..., sin(2^(L-1) π x), cos(2^(L-1) π x)]

    Args:
        num_frequencies: Number of frequency levels (default: 10)
        include_input: Include original input coordinates (default: True)
    """

    def __init__(self, num_frequencies: int = 10, include_input: bool = True):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.include_input = include_input

        # Frequency bands
        self.freq_bands = 2.0 ** torch.arange(num_frequencies, dtype=torch.float32)

        # Output dimension
        self.out_dim = 3 * (1 if include_input else 0) + 3 * 2 * num_frequencies

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode coordinates with sinusoidal positional encoding.

        Args:
            x: Coordinates of shape [B, 3]

        Returns:
            features: Encoded features of shape [B, out_dim]
        """
        device = x.device
        freq_bands = self.freq_bands.to(device)

        # Apply each frequency
        features = []
        if self.include_input:
            features.append(x)

        for freq in freq_bands:
            features.append(torch.sin(freq * np.pi * x))
            features.append(torch.cos(freq * np.pi * x))

        return torch.cat(features, dim=-1)

    def get_output_dim(self) -> int:
        """Return output dimension."""
        return self.out_dim


# Factory function
def create_encoding(encoding_type: str = "hash", **kwargs) -> nn.Module:
    """
    Factory function to create an encoding module.

    Args:
        encoding_type: Type of encoding ("hash" or "positional")
        **kwargs: Additional arguments passed to the encoding constructor

    Returns:
        encoding: An encoding module
    """
    if encoding_type == "hash":
        return HashEncoding(**kwargs)
    elif encoding_type == "positional":
        return PositionalEncoding(**kwargs)
    else:
        raise ValueError(f"Unknown encoding type: {encoding_type}")

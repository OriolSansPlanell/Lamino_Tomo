"""
Fully Connected MLP Networks

Implements efficient MLPs for neural field representation.
"""

import torch
import torch.nn as nn
from typing import List, Optional

try:
    import tinycudann as tcnn
    TCNN_AVAILABLE = True
except ImportError:
    TCNN_AVAILABLE = False


class FullyConnectedNetwork(nn.Module):
    """
    Fully connected MLP with optional tiny-cuda-nn acceleration.

    Args:
        input_dim: Input dimension
        output_dim: Output dimension
        hidden_dims: List of hidden layer dimensions (default: [64, 64, 64])
        activation: Activation function (default: "relu")
        output_activation: Output activation (default: "relu" for positive density)
        use_bias: Use bias terms (default: True)
        use_tcnn: Use tiny-cuda-nn if available (default: True)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [64, 64, 64],
        activation: str = "relu",
        output_activation: str = "relu",
        use_bias: bool = True,
        use_tcnn: bool = True,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.output_activation = output_activation
        self.use_bias = use_bias

        self.use_tcnn = use_tcnn and TCNN_AVAILABLE

        if self.use_tcnn:
            self._init_tcnn()
        else:
            self._init_pytorch()

    def _init_tcnn(self):
        """Initialize tiny-cuda-nn FullyFusedMLP (fastest)."""
        config = {
            "otype": "FullyFusedMLP",
            "activation": self.activation.upper(),
            "output_activation": self.output_activation.upper(),
            "n_neurons": self.hidden_dims[0] if self.hidden_dims else 64,
            "n_hidden_layers": len(self.hidden_dims),
        }

        self.network = tcnn.Network(
            n_input_dims=self.input_dim,
            n_output_dims=self.output_dim,
            network_config=config,
        )

    def _init_pytorch(self):
        """Initialize PyTorch MLP (fallback)."""
        layers = []

        # Input layer
        dims = [self.input_dim] + self.hidden_dims + [self.output_dim]

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1], bias=self.use_bias))

            # Add activation (except for last layer)
            if i < len(dims) - 2:
                layers.append(self._get_activation(self.activation))
            else:
                # Output activation
                if self.output_activation != "none":
                    layers.append(self._get_activation(self.output_activation))

        self.network = nn.Sequential(*layers)

    @staticmethod
    def _get_activation(name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            "relu": nn.ReLU(inplace=True),
            "leaky_relu": nn.LeakyReLU(0.2, inplace=True),
            "elu": nn.ELU(inplace=True),
            "sigmoid": nn.Sigmoid(),
            "tanh": nn.Tanh(),
            "softplus": nn.Softplus(),
            "none": nn.Identity(),
        }

        if name.lower() not in activations:
            raise ValueError(f"Unknown activation: {name}")

        return activations[name.lower()]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input features of shape [B, input_dim]

        Returns:
            output: Output of shape [B, output_dim]
        """
        return self.network(x)

    def extra_repr(self) -> str:
        """String representation."""
        return (
            f"input_dim={self.input_dim}, "
            f"output_dim={self.output_dim}, "
            f"hidden_dims={self.hidden_dims}, "
            f"backend={'tcnn' if self.use_tcnn else 'pytorch'}"
        )


class ResidualMLP(nn.Module):
    """
    MLP with residual connections for deeper networks.

    Args:
        input_dim: Input dimension
        output_dim: Output dimension
        hidden_dim: Hidden layer dimension
        num_blocks: Number of residual blocks
        activation: Activation function
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        num_blocks: int = 4,
        activation: str = "relu",
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks

        # Input projection
        self.input_layer = nn.Linear(input_dim, hidden_dim)

        # Residual blocks
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(
                ResidualBlock(hidden_dim, activation)
            )

        # Output projection
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.output_activation = nn.ReLU()

        self.activation = FullyConnectedNetwork._get_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        h = self.activation(self.input_layer(x))

        for block in self.blocks:
            h = block(h)

        out = self.output_activation(self.output_layer(h))
        return out


class ResidualBlock(nn.Module):
    """Single residual block: h' = h + MLP(h)."""

    def __init__(self, dim: int, activation: str = "relu"):
        super().__init__()

        self.layer1 = nn.Linear(dim, dim)
        self.layer2 = nn.Linear(dim, dim)
        self.activation = FullyConnectedNetwork._get_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        identity = x
        out = self.activation(self.layer1(x))
        out = self.layer2(out)
        return self.activation(out + identity)


class ConditionedMLP(nn.Module):
    """
    MLP conditioned on additional features (e.g., for latent diffusion).

    Args:
        input_dim: Input coordinate dimension
        condition_dim: Conditioning feature dimension
        output_dim: Output dimension
        hidden_dims: Hidden layer dimensions
    """

    def __init__(
        self,
        input_dim: int,
        condition_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [128, 128, 128],
    ):
        super().__init__()

        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.output_dim = output_dim

        # Concatenate input and conditioning
        self.mlp = FullyConnectedNetwork(
            input_dim + condition_dim,
            output_dim,
            hidden_dims,
            use_tcnn=False,  # tcnn doesn't support variable conditioning yet
        )

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with conditioning.

        Args:
            x: Input coordinates [B, input_dim]
            condition: Conditioning features [B, condition_dim]

        Returns:
            output: Network output [B, output_dim]
        """
        # Concatenate input and conditioning
        combined = torch.cat([x, condition], dim=-1)
        return self.mlp(combined)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(model: nn.Module, method: str = "kaiming"):
    """
    Initialize network weights.

    Args:
        model: Network module
        method: Initialization method ("kaiming", "xavier", "normal")
    """
    for m in model.modules():
        if isinstance(m, nn.Linear):
            if method == "kaiming":
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            elif method == "xavier":
                nn.init.xavier_normal_(m.weight)
            elif method == "normal":
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            else:
                raise ValueError(f"Unknown initialization method: {method}")

            if m.bias is not None:
                nn.init.zeros_(m.bias)


# Factory function
def create_mlp(
    input_dim: int,
    output_dim: int,
    mlp_type: str = "standard",
    **kwargs
) -> nn.Module:
    """
    Factory function to create an MLP.

    Args:
        input_dim: Input dimension
        output_dim: Output dimension
        mlp_type: Type of MLP ("standard", "residual", "conditioned")
        **kwargs: Additional arguments

    Returns:
        mlp: An MLP module
    """
    if mlp_type == "standard":
        return FullyConnectedNetwork(input_dim, output_dim, **kwargs)
    elif mlp_type == "residual":
        return ResidualMLP(input_dim, output_dim, **kwargs)
    elif mlp_type == "conditioned":
        return ConditionedMLP(input_dim, output_dim=output_dim, **kwargs)
    else:
        raise ValueError(f"Unknown MLP type: {mlp_type}")

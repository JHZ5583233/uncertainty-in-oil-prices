from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset


def get_device() -> torch.device:
    """Auto-detect the best available device (GPU or CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")  # Apple Metal Performance Shaders
    else:
        return torch.device("cpu")


class BayesianLinear(torch.nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = False,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.in_dim: int = in_dim
        self.out_dim: int = out_dim
        self.bias: bool = bias
        self.device: torch.device = device if device is not None else get_device()

        self.w_log_var = nn.Parameter(
            -2 + 0.1 * torch.randn([self.in_dim, self.out_dim], device=self.device)
        )
        self.w_mu = nn.Parameter(
            0.1 * torch.randn([self.in_dim, self.out_dim], device=self.device)
        )

        # Same small initialization here
        if self.bias:
            self.bias_log_var = nn.Parameter(
                -2 + 0.1 * torch.randn(self.out_dim, device=self.device)
            )
            self.bias_mu = nn.Parameter(
                0.1 * torch.randn([self.out_dim], device=self.device)
            )

    def forward(self, x: Tensor) -> Tensor:
        # Sample weights from approximate posterior: mean + stddev * random noise - reparam trick
        weight: Tensor = self.w_mu + self.w_log_var.exp().sqrt() * torch.randn_like(
            self.w_log_var, device=self.device
        )
        if self.bias:
            # Sample bias similarly if enabled
            bias: Tensor = (
                self.bias_mu
                + self.bias_log_var.exp().sqrt()
                * torch.randn_like(self.bias_log_var, device=self.device)
            )
        else:
            # If no bias, use zero bias vector
            bias: Tensor = torch.zeros(self.out_dim, device=self.device)

        # Apply linear transformation using sampled weight and bias
        return F.linear(x, weight.t(), bias)

    def kl_div(self) -> Tensor:
        # Compute KL divergence between approximate posterior and standard normal prior for weights
        kl_div_W: Tensor = 0.5 * torch.sum(
            -self.w_log_var + self.w_log_var.exp() + self.w_mu**2 - 1
        )

        if self.bias:
            # Compute KL divergence for bias parameters if enabled
            kl_div_b: Tensor = 0.5 * torch.sum(
                -self.bias_log_var + self.bias_log_var.exp() + self.bias_mu**2 - 1
            )
        else:
            kl_div_b: Tensor = torch.tensor(0.0, device=self.device)

        return kl_div_W + kl_div_b


class BayesianNeuralNetwork(nn.Sequential):
    def __init__(
        self,
        in_dim: int = 2,
        use_bias: bool = False,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.device: torch.device = device if device is not None else get_device()
        self.BL1: BayesianLinear = BayesianLinear(
            in_dim, 5, bias=use_bias, device=self.device
        )
        self.BL2: BayesianLinear = BayesianLinear(
            5, 5, bias=use_bias, device=self.device
        )
        self.BL3: BayesianLinear = BayesianLinear(
            5, 1, bias=use_bias, device=self.device
        )

        self.activation: nn.Module = nn.Tanh()
        self.output_activation: nn.Module = nn.Sigmoid()

    def kl_div(self) -> Tensor:
        """Sums KL divergence across all layers."""
        kl_total: Tensor = torch.tensor(0.0, device=self.device)
        for lyr in self:
            if hasattr(lyr, "kl_div"):
                kl_total = kl_total + lyr.kl_div()
        return kl_total

    def forward(self, x: Tensor) -> Tensor:
        x: Tensor = self.activation(self.BL1(x))
        x: Tensor = self.activation(self.BL2(x))
        x: Tensor = self.BL3(x)
        x: Tensor = self.output_activation(x)
        return x

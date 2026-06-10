import torch
import torch.nn.functional as F
from torch import nn


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.mps.is_available():
        return "mps"
    elif torch.xpu.is_available():
        return "xps"

    return "cpu"


class BayesianLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, bias: bool = False, device: str = ""):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.bias = bias

        if device != "":
            self.device = device
        else:
            self.device = detect_device()

        self.w_log_var = nn.Parameter(
            -2 + 0.1 * torch.randn([self.in_dim, self.out_dim], device=self.device)
        )
        self.w_mu = nn.Parameter(
            0.1 * torch.randn([self.in_dim, self.out_dim], device=self.device)
        )

        if self.bias:
            self.bias_log_var = nn.Parameter(
                -2 + 0.1 * torch.randn(self.out_dim, device=self.device)
            )
            self.bias_mu = nn.Parameter(
                0.1 * torch.randn([self.out_dim], device=self.device)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight: torch.Tensor = (
            self.w_mu
            + self.w_log_var.exp().sqrt()
            * torch.randn_like(self.w_log_var, device=self.device)
        )
        if self.bias:
            bias: torch.Tensor = (
                self.bias_mu
                + self.bias_log_var.exp().sqrt()
                * torch.randn_like(self.bias_log_var, device=self.device)
            )
        else:
            bias = torch.zeros(self.out_dim, device=self.device)

        return F.linear(x, weight.t(), bias)

    def kl_div(self) -> torch.Tensor:
        kl_div_W = 0.5 * torch.sum(
            -self.w_log_var + self.w_log_var.exp() + self.w_mu**2 - 1
        )

        if self.bias:
            kl_div_b = 0.5 * torch.sum(
                -self.bias_log_var + self.bias_log_var.exp() + self.bias_mu**2 - 1
            )
        else:
            kl_div_b = 0

        return kl_div_W + kl_div_b


class BayesianNeuralNetwork(nn.Module):
    def __init__(self, in_dim=2, use_bias=False, device="cuda"):
        super().__init__()
        self.device = device
        self.BL1 = BayesianLinear(in_dim, 5, bias=use_bias, device=self.device)
        self.BL2 = BayesianLinear(5, 5, bias=use_bias, device=self.device)
        self.BL3 = BayesianLinear(5, 1, bias=use_bias, device=self.device)

        self.activation = nn.Tanh()

    def kl_div(self) -> torch.Tensor:
        kl_total: torch.Tensor = torch.tensor(0.0, device=self.device)
        for lyr in self.modules():
            if isinstance(lyr, BayesianLinear):
                kl_total = kl_total + lyr.kl_div()
        return kl_total

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.BL1(x))
        x = self.activation(self.BL2(x))
        x = self.BL3(x)
        return x

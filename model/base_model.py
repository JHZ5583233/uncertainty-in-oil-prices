import torch
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
    def __init__(self, in_dim: int, out_dim: int, bias: bool=False, device: str=''):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.bias = bias

        if device != '':
            self.device = device
        else:
            self.device = detect_device()

        self.w_log_var = nn.Parameter(-2 + 0.1* torch.randn([self.in_dim,self.out_dim],device=self.device))
        self.w_mu = nn.Parameter(0.1 * torch.randn([self.in_dim,self.out_dim],device=self.device))

        if self.bias:
            self.bias_log_var = nn.Parameter(-2 + 0.1 * torch.randn(self.out_dim,device=self.device))
            self.bias_mu = nn.Parameter(0.1 * torch.randn([self.out_dim],device=self.device))


    def forward(self, x):
         # Sample weights from approximate posterior: mean + stddev * random noise - reparam trick
        weight = self.w_mu + self.w_log_var.exp().sqrt() * torch.randn_like(self.w_log_var,device=self.device)
        if self.bias:
            # Sample bias similarly if enabled
            bias = self.bias_mu + self.bias_log_var.exp().sqrt() * torch.randn_like(self.bias_log_var ,device=self.device)
        else:
            # If no bias, use zero bias vector
            bias = torch.zeros(self.out_dim,device=self.device)
            
        # Apply linear transformation using sampled weight and bias
        return F.linear(x, weight.t(), bias)


    def kl_div(self):
            # Compute KL divergence between approximate posterior and standard normal prior for weights
            kl_div_W = 0.5 * torch.sum(-self.w_log_var + self.w_log_var.exp() + self.w_mu**2 - 1)

            if self.bias:
                # Compute KL divergence for bias parameters if enabled
                kl_div_b = 0.5 * torch.sum(-self.bias_log_var + self.bias_log_var.exp() + self.bias_mu**2 - 1)
            else:
                kl_div_b = 0
        
            return kl_div_W + kl_div_b 
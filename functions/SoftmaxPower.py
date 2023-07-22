import torch


class SoftmaxPower(torch.nn.Module):
    dim: int | None
    power: float

    def __init__(self, power: float = 2.0, dim: int | None = None) -> None:
        super().__init__()
        self.dim = dim
        self.power = power

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, "dim"):
            self.dim = None
        if not hasattr(self, "power"):
            self.power = 2.0

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output: torch.Tensor = torch.abs(input).pow(self.power)
        if self.dim is None:
            output = output / output.sum()
        else:
            output = output / output.sum(dim=self.dim, keepdim=True)
        return output

    def extra_repr(self) -> str:
        return f"dim={self.dim} ; power={self.power}"

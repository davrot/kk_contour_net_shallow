import torch


class SoftmaxPower(torch.nn.Module):
    dim: int | None
    power: float
    mean_mode: bool
    no_input_mode: bool

    def __init__(
        self,
        power: float = 2.0,
        dim: int | None = None,
        mean_mode: bool = False,
        no_input_mode: bool = False,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.power = power
        self.mean_mode = mean_mode
        self.no_input_mode = no_input_mode

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, "dim"):
            self.dim = None
        if not hasattr(self, "power"):
            self.power = 2.0
        if not hasattr(self, "mean_mode"):
            self.mean_mode = False
        if not hasattr(self, "no_input_mode"):
            self.no_input_mode = False

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.power != 0.0:
            output: torch.Tensor = torch.abs(input).pow(self.power)
        else:
            output: torch.Tensor = torch.exp(input)

        if self.dim is None:
            output = output / output.sum()
        else:
            output = output / output.sum(dim=self.dim, keepdim=True)

        if self.no_input_mode:
            return output
        elif self.mean_mode:
            return torch.abs(input).mean(dim=1, keepdim=True) * output
        else:
            return input * output

    def extra_repr(self) -> str:
        if self.power != 0.0:
            return (
                f"dim={self.dim}; "
                f"power={self.power}; "
                f"mean_mode={self.mean_mode}; "
                f"no_input_mode={self.no_input_mode}"
            )
        else:
            return (
                f"dim={self.dim}; "
                "exp-mode; "
                f"mean_mode={self.mean_mode}; "
                f"no_input_mode={self.no_input_mode}"
            )

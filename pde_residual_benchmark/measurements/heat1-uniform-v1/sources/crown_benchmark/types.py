"""Shared scalar-target contract; no numerical backend is selected here."""
from dataclasses import dataclass, field
from typing import Any
import torch


class BoundFailure(RuntimeError):
    """Invalid/unsupported computation: never replace it with a sampled bound."""


@dataclass
class BoundBatch:
    lower: torch.Tensor  # shape [batch], signed lower enclosure
    upper: torch.Tensor  # shape [batch], signed upper enclosure
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self, batch_size: int | None = None) -> "BoundBatch":
        if self.lower.ndim != 1 or self.upper.shape != self.lower.shape:
            raise BoundFailure("Expected one signed interval per scalar-target box")
        if batch_size is not None and len(self.lower) != batch_size:
            raise BoundFailure("Backend returned a different number of boxes")
        if not torch.isfinite(self.lower).all() or not torch.isfinite(self.upper).all():
            raise BoundFailure("Nonfinite bound")
        if torch.any(self.lower > self.upper):
            raise BoundFailure("Reversed bound")
        return self


def validate_boxes(lower, upper, input_dim: int, dtype=torch.float64, device="cpu"):
    lower = torch.as_tensor(lower, dtype=dtype, device=device)
    upper = torch.as_tensor(upper, dtype=dtype, device=device)
    if lower.ndim == 1:
        lower = lower.unsqueeze(0)
    if upper.ndim == 1:
        upper = upper.unsqueeze(0)
    if (lower.ndim != 2 or upper.shape != lower.shape or
            lower.shape[1] != input_dim or lower.shape[0] == 0):
        raise ValueError("Boxes must have matching nonempty [batch,input_dim] shapes")
    if not torch.isfinite(lower).all() or not torch.isfinite(upper).all():
        raise ValueError("Nonfinite box endpoint")
    if torch.any(lower > upper):
        raise ValueError("Reversed box")
    return lower, upper

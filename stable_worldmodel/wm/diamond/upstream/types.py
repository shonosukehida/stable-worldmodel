from typing import NamedTuple

from torch import Tensor


class Batch(NamedTuple):
    obs: Tensor
    act: Tensor
    mask_padding: Tensor


LossAndLogs = tuple[Tensor, dict[str, Tensor]]

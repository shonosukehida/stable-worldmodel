"""RS loss copied from s3fmp/src/s3fmp/policy.py (see provenance.md)."""

import math
from typing import Tuple
import torch


def rs_imle_loss(
    real: torch.Tensor,
    fake: torch.Tensor,
    epsilon: float,
    scale_by_dim: bool = True,
) -> Tuple[torch.Tensor, dict]:
    """RS-IMLE 損失. real: (B, T, D), fake: (B, N, T, D).

    各正解について, 距離が epsilon より大きい候補の中で最も近いものとの L2 距離 (T, D 方向で取る) を縮める.
    epsilon 以内の候補しかない正解は損失から外し, 残りの正解で batch 平均を取る.
    出典: vapo (https://github.com/oakwood-fujiken/vapo-aif, commit 0256ce9) vapo/policy/imle.py::rs_imle_loss
    (元は imle_policy (RSS 2025) の utils/losses.py).
    """
    (B, T, D) = real.shape
    N = fake.shape[1]
    dist = (
        (real.reshape(B, 1, T * D) - fake.reshape(B, N, T * D))
        .pow(2)
        .sum(-1)
        .clamp_min(1e-12)
        .sqrt()
    )
    eps = epsilon * math.sqrt(T * D) if scale_by_dim else epsilon
    dmax = dist.max().detach()
    valid = dist > eps
    min_dist = (dist + (~valid).float() * dmax).min(dim=1).values
    valid_real = (min_dist < dmax).float()
    loss = (min_dist * valid_real).sum() / valid_real.sum().clamp_min(1.0)
    logs = {
        'metrics/min_distance': dist.min().detach(),
        'metrics/max_distance': dmax,
        'metrics/mean_distance': dist.mean().detach(),
        'metrics/valid_ratio': valid_real.mean(),
    }
    return (loss, logs)

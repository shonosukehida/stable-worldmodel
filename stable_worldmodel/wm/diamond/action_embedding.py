"""Physical actions are absolute Cartesian targets in the dataset's recorded frame."""
import torch
from torch import nn


ACTION_FIELDS = ["x_m", "y_m", "z_m", "qx", "qy", "qz", "qw", "gripper_0_open_1_closed"]


def cartesian_embedding():
    return nn.Sequential(nn.Linear(8, 64, bias=False), nn.Flatten())


def validate_normalizer(spec):
    if (spec.get("method") != "WM.SafeStandardScaler.v1" or spec.get("fields") != ACTION_FIELDS
            or spec.get("key") != "action_cartesian" or spec.get("eps") != 1e-4):
        raise ValueError("Unsupported Cartesian normalization")
    for key in ("mean", "scale"):
        v = torch.tensor(spec[key], dtype=torch.float64)
        if v.shape != (8,) or not v.isfinite().all() or (key == "scale" and (v <= 0).any()):
            raise ValueError(f"Invalid normalization {key}")
    digest = spec.get("stats_sha256", "")
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ValueError("Missing WM statistics fingerprint")


def normalize_actions(raw, spec):
    """Input (...,8) in physical units. Output FP32 in the model's action space."""
    validate_normalizer(spec)
    raw = torch.as_tensor(raw).float()
    if raw.shape[-1:] != (8,) or not raw.isfinite().all():
        raise ValueError("Invalid absolute Cartesian action")
    return (raw - raw.new_tensor(spec["mean"])) / raw.new_tensor(spec["scale"])

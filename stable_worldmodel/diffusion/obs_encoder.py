import torch
import torch.nn as nn
from torchvision.models import resnet18


class ResNet18ObsEncoder(nn.Module):
    """
    Encode one RGB observation into a feature vector.

    Input:
        (B, 3, H, W)

    Output:
        (B, feature_dim)
    """

    def __init__(
        self,
        pretrained: bool = False,
    ):
        super().__init__()

        if pretrained:
            from torchvision.models import (
                ResNet18_Weights,
            )

            backbone = resnet18(
                weights=ResNet18_Weights.DEFAULT
            )

        else:
            backbone = resnet18(
                weights=None
            )

        # ResNet18 classifier input = 512
        self.feature_dim = (
            backbone.fc.in_features
        )

        # Remove classification layer
        backbone.fc = nn.Identity()

        self.backbone = backbone

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        if x.ndim != 4:
            raise ValueError(
                "Expected image tensor with shape "
                f"(B, C, H, W), got {x.shape}"
            )

        return self.backbone(x)

    def output_shape(self):
        return (self.feature_dim,)


class ConvObsEncoder(nn.Module):
    """Small RGB CNN with adaptive pooling and a configurable feature projection."""

    def __init__(self, feature_dim: int = 512):
        super().__init__()
        if not isinstance(feature_dim, int) or isinstance(feature_dim, bool) or feature_dim <= 0:
            raise ValueError("feature_dim must be a positive integer")
        self.feature_dim = feature_dim
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, feature_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f"Expected RGB tensor (B, 3, H, W), got {x.shape}")
        return self.backbone(x)

    def output_shape(self):
        return (self.feature_dim,)


def resolve_obs_encoder_config(config=None):
    """Normalize training settings; missing type retains historical ResNet18."""
    config = dict(config or {})
    encoder_type = config.get("type", "resnet18")
    if encoder_type == "resnet18":
        return encoder_type, {"pretrained": config.get("pretrained", False)}
    if encoder_type == "conv":
        return encoder_type, {"feature_dim": config.get("feature_dim", 512)}
    if encoder_type == "jepa":
        return encoder_type, {key: config.get(key) for key in
                              ("checkpoint", "world_config", "process_stats")}
    raise ValueError(f"Unknown encoder_type: {encoder_type!r}")


def create_obs_encoder(encoder_type="resnet18", encoder_config=None, *, load_weights=False):
    """Construct an encoder without wrapping or renaming its state_dict keys.

    load_weights skips pretrained downloads when checkpoint weights follow.
    """
    config = dict(encoder_config or {})
    if encoder_type == "resnet18":
        return ResNet18ObsEncoder(pretrained=False if load_weights else config.get("pretrained", False))
    if encoder_type == "conv":
        return ConvObsEncoder(**config)
    raise ValueError(f"Unknown encoder_type: {encoder_type!r}")

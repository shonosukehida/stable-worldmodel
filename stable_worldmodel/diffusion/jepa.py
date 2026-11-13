"""Frozen multimodal conditioning, sharing DiffusionPolicy's diffusion code."""
import torch
from torch import nn

from stable_worldmodel.policy import DiffusionPolicy


class FrozenJEPAEncoder(nn.Module):
    """Run the source JEPA.encode implementation without action inputs.

    Inputs are image-preprocessed, but proprio is raw. Its world-model training
    statistics are buffers, so training and deployment use the same transform.
    """

    def __init__(self, world_model, mean, std, image_size):
        super().__init__()
        self.world_model = world_model
        self.image_size = int(image_size)
        mean, std = torch.as_tensor(mean).float(), torch.as_tensor(std).float()
        if (mean.ndim != 2 or mean.shape[0] != 1 or mean.shape != std.shape
                or not torch.isfinite(mean).all() or not torch.isfinite(std).all()
                or not (std > 0).all() or self.image_size <= 0):
            raise ValueError("Invalid JEPA image size or proprio normalization statistics")
        self.register_buffer("proprio_mean", mean)
        self.register_buffer("proprio_std", std)
        self.requires_grad_(False)
        self.eval()
        with torch.no_grad():
            info = self.world_model.encode({
                "pixels": torch.zeros(1, 1, 3, self.image_size, self.image_size),
                "wrist_pixels": torch.zeros(1, 1, 3, self.image_size, self.image_size),
                "proprio": torch.zeros(1, 1, mean.shape[-1]),
            })
        self.feature_dims = tuple(info[k].shape[-1] for k in
                                  ("overhead_img_emb", "wrist_img_emb", "prop_emb"))
        if info["emb"].shape != (1, 1, sum(self.feature_dims)):
            raise ValueError("JEPA.encode returned incompatible multimodal embeddings")

    def train(self, mode=True):
        # Includes BN/dropout in both projectors and all encoders.
        return super().train(False)

    def output_shape(self):
        return (sum(self.feature_dims),)

    @torch.no_grad()
    def forward(self, info):
        prepared = {}
        for key in ("pixels", "wrist_pixels", "proprio"):
            if key not in info:
                raise KeyError(f"JEPA-DP requires {key}")
            prepared[key] = info[key].to(self.proprio_mean)
            if not torch.isfinite(prepared[key]).all():
                raise ValueError(f"Non-finite JEPA observation: {key}")
        prop = prepared["proprio"]
        if prop.ndim != 3 or prop.shape[-1] != self.proprio_mean.shape[-1]:
            raise ValueError("Invalid JEPA proprio shape")
        prepared["proprio"] = (prop - self.proprio_mean) / self.proprio_std
        return self.world_model.encode(prepared)["emb"]


class EmptyWristEncoder(nn.Module):
    """Compatibility handle: the combined encoder already includes the wrist."""
    def output_shape(self):
        return (0,)


class JEPADiffusionPolicy(DiffusionPolicy):
    use_proprio = True

    def _encode_observation_info(self, info):
        observations = {key: info[key] for key in ("pixels", "wrist_pixels", "proprio")}
        batch, steps = observations["pixels"].shape[:2]
        if steps != self.obs_horizon:
            raise ValueError(f"JEPA-DP needs exactly obs_horizon={self.obs_horizon} observations")
        embeddings = self.obs_encoder(observations)
        return embeddings.flatten(start_dim=1), batch, steps

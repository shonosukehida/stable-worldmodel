"""Multimodal JEPA conditioning, sharing DiffusionPolicy's diffusion code."""
import torch
from torch import nn

from stable_worldmodel.policy import DiffusionPolicy


FEATURE_MODULES = ("overhead_encoder", "wrist_encoder", "overhead_projector",
                   "wrist_projector", "prop_encoder")


class FrozenJEPAEncoder(nn.Module):
    """Run the source JEPA.encode implementation without action inputs.

    Inputs are image-preprocessed, but proprio is raw. Its world-model training
    statistics are buffers, so training and deployment use the same transform.
    """

    # Older pickled encoders predate the instance attribute.
    finetune_mode = "frozen"

    def __init__(self, world_model, mean, std, image_size, *, mode="frozen", unfreeze_last_n_blocks=3):
        super().__init__()
        self.world_model = world_model
        if mode not in ("frozen", "partial", "full"):
            raise ValueError(f"Unknown encoder.finetune.mode: {mode!r}")
        self.finetune_mode = mode
        self.unfreeze_last_n_blocks = unfreeze_last_n_blocks
        self.image_size = int(image_size)
        mean, std = torch.as_tensor(mean).float(), torch.as_tensor(std).float()
        if (mean.ndim != 2 or mean.shape[0] != 1 or mean.shape != std.shape
                or not torch.isfinite(mean).all() or not torch.isfinite(std).all()
                or not (std > 0).all() or self.image_size <= 0):
            raise ValueError("Invalid JEPA image size or proprio normalization statistics")
        self.register_buffer("proprio_mean", mean)
        self.register_buffer("proprio_std", std)
        self.requires_grad_(False)
        for module in self._finetune_modules():
            module.requires_grad_(True)
        if mode == "full":
            # JEPA.encode uses unmasked images and last_hidden_state, even
            # when the source ViT also has a mask token or a pooled output.
            for vit in (world_model.overhead_encoder, world_model.wrist_encoder):
                if getattr(vit, "pooler", None) is not None:
                    vit.pooler.requires_grad_(False)
                mask_token = getattr(getattr(vit, "embeddings", None), "mask_token", None)
                if mask_token is not None:
                    mask_token.requires_grad_(False)
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

    def _finetune_modules(self):
        if self.finetune_mode == "full":
            return [getattr(self.world_model, name) for name in FEATURE_MODULES]
        modules = []
        if self.finetune_mode == "partial":
            n = self.unfreeze_last_n_blocks
            for name in ("overhead_encoder", "wrist_encoder"):
                vit = getattr(self.world_model, name)
                blocks = getattr(getattr(vit, "encoder", None), "layer", None)
                if not isinstance(blocks, nn.ModuleList):
                    raise ValueError(f"Expected {name}.encoder.layer to contain ViT blocks")
                if type(n) is not int or not 1 <= n <= len(blocks):
                    raise ValueError(f"encoder.finetune.unfreeze_last_n_blocks must be an integer "
                                     f"in [1, {len(blocks)}] for {name}, got {n!r}")
                modules.extend(list(blocks)[-n:])
        return modules

    def train(self, mode=True):
        super().train(False)
        if mode and self.finetune_mode != "frozen":
            self.training = True
            for module in self._finetune_modules():
                module.train()
            if self.finetune_mode == "full":
                for vit in (self.world_model.overhead_encoder, self.world_model.wrist_encoder):
                    if getattr(vit, "pooler", None) is not None:
                        vit.pooler.eval()
        return self

    def output_shape(self):
        return (sum(self.feature_dims),)

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
        with torch.set_grad_enabled(self.training and torch.is_grad_enabled()):
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

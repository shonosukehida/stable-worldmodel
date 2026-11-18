from dataclasses import asdict

import torch
from torch import nn

from .action_embedding import cartesian_embedding
from .history import History
from .preprocessing import image_spec
from .upstream.diffusion.denoiser import Denoiser, DenoiserConfig, SigmaDistributionConfig
from .upstream.diffusion.diffusion_sampler import DiffusionSampler, DiffusionSamplerConfig
from .upstream.diffusion.inner_model import InnerModelConfig


DEFAULT_VIEWS = ("pixels", "wrist_pixels")
SIGMA = SigmaDistributionConfig(-0.4, 1.2, 0.002, 20)
ALIGNMENT = dict(version=1, action="stored_row_t", label="next-row leader absolute target",
                 stride=1, measured_hz=None, physical_causality="unverified")


def model_spec(views=DEFAULT_VIEWS):
    image = image_spec(views)
    return dict(image=image, history=4, action_embedding="shared Linear(8,64,bias=False), flatten(4)",
                unet=dict(img_channels=3*len(views), num_steps_conditioning=4, cond_channels=256,
                          depths=[2]*4, channels=[64]*4, attn_depths=[0]*4),
                edm=dict(sigma_data=0.5, sigma_offset_noise=0.3, distribution=asdict(SIGMA),
                         frames=6, targets=2), sampler=asdict(DiffusionSamplerConfig(3)),
                initial_noise="unit Gaussian", quantization="upstream wrap_model_output uint8 truncation")


class DiamondWorldModel(nn.Module):
    def __init__(self, views=DEFAULT_VIEWS):
        super().__init__()
        self.spec = model_spec(views)
        cfg = InnerModelConfig(**self.spec["unet"], num_actions=8)
        self.denoiser = Denoiser(DenoiserConfig(cfg, 0.5, 0.3))
        self.denoiser.inner_model.act_emb = cartesian_embedding()
        self.denoiser.setup_training(SIGMA)

    def forward(self, batch):
        return self.denoiser(batch)

    def _sampler(self, denoising_steps):
        cfg = DiffusionSamplerConfig(**self.spec["sampler"])
        if denoising_steps is not None:
            if not isinstance(denoising_steps, int) or denoising_steps < 1:
                raise ValueError("denoising_steps must be a positive integer")
            cfg.num_steps_denoising = denoising_steps
        return DiffusionSampler(self.denoiser, cfg)

    def _check(self, obs, actions, history_length):
        channels = self.spec["unet"]["img_channels"]
        if self.training:
            raise ValueError("Call model.eval() before sampling")
        if obs.ndim != 5 or not len(obs) or obs.shape[1:] != (4, channels, 64, 64):
            raise ValueError("Expected (B,4,C,64,64) observation history")
        if actions.shape != (len(obs), history_length, 8):
            raise ValueError("Wrong action history shape")
        for x in (obs, actions):
            if x.device != self.denoiser.device or x.dtype != torch.float32 or not x.isfinite().all():
                raise ValueError("Expected finite FP32 inputs on the model device")
        if obs.min() < -1.00001 or obs.max() > 1.00001:
            raise ValueError("Expected normalized DIAMOND observations")

    @staticmethod
    def _sample(sampler, obs, actions, generator):
        x, _ = sampler.sample(obs, actions, generator=generator)
        if not x.isfinite().all() or x.min() < -1.00001 or x.max() > 1.00001:
            raise FloatingPointError("Sampler returned an invalid image")
        return x

    @torch.inference_mode()
    def predict_next(self, obs_history, action_window, generator=None):
        self._check(obs_history, action_window, 4)
        return self._sample(self._sampler(None), obs_history, action_window, generator)

    @torch.inference_mode()
    def rollout_images(self, obs_history, past_actions, future_actions, generator=None, *,
                       denoising_steps=None):
        """Normalized actions (B,H,8) or (B,K,H,8). The returned shape preserves K."""
        self._check(obs_history, past_actions, 3)
        batch = len(obs_history)
        if (future_actions.ndim not in (3, 4) or future_actions.shape[0] != batch
                or future_actions.shape[-1] != 8 or not future_actions.numel()
                or future_actions.device != obs_history.device or future_actions.dtype != torch.float32
                or not future_actions.isfinite().all()):
            raise ValueError("Invalid normalized future action sequence")
        leading = future_actions.shape[:-2]
        candidates = leading[-1] if len(leading) == 2 else 1
        actions = future_actions.reshape(-1, *future_actions.shape[-2:])
        history = History(obs_history.repeat_interleave(candidates, 0), past_actions.repeat_interleave(candidates, 0))
        sampler = self._sampler(denoising_steps)
        frames = []
        for next_action in actions.unbind(dim=1):
            window = history.action_window(next_action)
            frame = self._sample(sampler, history.obs, window, generator)
            frames.append(frame)
            history = history.advance(frame, window)
        return torch.stack(frames, dim=1).reshape(*leading, len(frames), *frame.shape[1:])

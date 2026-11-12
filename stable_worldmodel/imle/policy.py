"""IMLE interface adapter. Importing legacy policies never initializes IMLE."""

import torch
import torch.nn.functional as F
from stable_worldmodel.policy import BasePolicy, DiffusionPolicy
from .losses import rs_imle_loss


class IMLEPolicy(DiffusionPolicy):
    """Reuse only the DP observation/action interface, never its denoising path.

    Inheritance retains existing robot buffers and gripper reinforcement dispatch.
    No DP constructor, scheduler, model initialization or sampling is invoked.
    """

    def __init__(
        self,
        model,
        obs_encoder,
        wrist_obs_encoder,
        pred_horizon,
        obs_horizon,
        action_horizon,
        action_dim,
        noise_scale=1.0,
        n_samples_per_condition=128,
        epsilon=0.01,
        epsilon_scale_by_dim=True,
        gripper_loss_weight=0.003125,
        process=None,
        transform=None,
        use_proprio=False,
        proprio_stats=None,
        **kwargs,
    ):
        BasePolicy.__init__(self, **kwargs)
        if not (
            1 <= obs_horizon
            and 1 <= action_horizon
            and obs_horizon - 1 + action_horizon <= pred_horizon
        ):
            raise ValueError(
                'Invalid IMLE horizons: execution chunk must fit prediction'
            )
        if n_samples_per_condition < 1 or noise_scale <= 0 or epsilon < 0:
            raise ValueError('Invalid IMLE sampling parameters')
        from .proprio import validate_proprio_stats
        if not isinstance(use_proprio, bool):
            raise ValueError('IMLE use_proprio must be boolean')
        self.use_proprio = use_proprio
        self.proprio_stats = validate_proprio_stats(proprio_stats) if use_proprio else None
        self.type = 'imle'
        self.model, self.obs_encoder = model, obs_encoder
        self.wrist_obs_encoder = wrist_obs_encoder
        self.pred_horizon, self.obs_horizon = pred_horizon, obs_horizon
        self.action_horizon, self.action_dim = action_horizon, action_dim
        self.noise_dim = model.gen_dim
        self.noise_scale = noise_scale
        self.n_samples_per_condition = n_samples_per_condition
        self.epsilon, self.epsilon_scale_by_dim = epsilon, epsilon_scale_by_dim
        self.gripper_loss_weight = gripper_loss_weight
        self.num_inference_steps = 1
        self.process, self.transform = process or {}, transform or {}

    def _condition(self, pixels, wrist_pixels, proprio=None):
        if pixels.shape[1] < self.obs_horizon or wrist_pixels is None:
            raise ValueError(
                'IMLE requires complete overhead and wrist observation histories'
            )
        flat, batch, steps = self._encode_observation(pixels, wrist_pixels)
        if self.use_proprio:
            from .proprio import normalize_proprio
            if proprio is None or tuple(proprio.shape) != (batch, steps, 8):
                raise ValueError('IMLE proprio requires shape (B, obs_horizon, 8)')
            state = normalize_proprio(proprio, self.proprio_stats, device=self.device, dtype=self.dtype)
            return torch.cat([flat.reshape(batch, steps, -1), state], dim=-1)
        return flat.reshape(batch, steps * 2, -1)

    def _noise(self, batch):
        return (
            torch.randn(
                batch,
                self.pred_horizon,
                self.noise_dim,
                device=self.device,
                dtype=self.dtype,
            )
            * self.noise_scale
        )

    def predict_action(self, info_dict):
        info = self._prepare_info(dict(info_dict))
        cond = self._condition(info['pixels'], info.get('wrist_pixels'), info.get('proprio'))
        trajectory = self.model(cond, self._noise(cond.shape[0]))
        start = self.obs_horizon - 1
        return {
            'action': trajectory[:, start : start + self.action_horizon],
            'action_pred': trajectory,
        }

    @torch.no_grad()
    def sample_action_sequences(
        self, info_dict, num_samples=50, denormalize=False
    ):
        if (
            isinstance(num_samples, bool)
            or not isinstance(num_samples, int)
            or num_samples < 1
        ):
            raise ValueError('num_samples must be a positive integer')
        info = self._prepare_info(dict(info_dict))
        cond = self._condition(info['pixels'], info.get('wrist_pixels'), info.get('proprio'))
        batch = cond.shape[0]
        repeated = cond.repeat_interleave(num_samples, dim=0)
        result = self.model(
            repeated, self._noise(batch * num_samples)
        ).reshape(batch, num_samples, self.pred_horizon, self.action_dim)
        if denormalize:
            for key in ('action_cartesian', 'action', 'action_joint'):
                if key in self.process:
                    shape = result.shape
                    physical = self.process[key].inverse_transform(
                        result.cpu().numpy().reshape(-1, self.action_dim)
                    )
                    result = torch.as_tensor(
                        physical.reshape(shape),
                        device=self.device,
                        dtype=self.dtype,
                    )
                    break
            else:
                raise ValueError(
                    'Physical actions require checkpoint normalization statistics'
                )
        if not torch.isfinite(result).all():
            raise ValueError('IMLE produced nonfinite actions')
        return result

    def compute_loss(self, batch):
        cond = self._condition(batch['pixels'], batch.get('wrist_pixels'), batch.get('proprio'))
        action = batch['action'].to(device=self.device, dtype=self.dtype)
        if action.shape != (cond.shape[0], self.pred_horizon, self.action_dim):
            raise ValueError(
                'Teacher action trajectory shape does not match IMLE horizons'
            )
        b, n = cond.shape[0], self.n_samples_per_condition
        joints = action[..., : self.noise_dim]
        fake = self.model.generate(
            self._noise(b * n), cond.repeat_interleave(n, 0)
        ).reshape(b, n, self.pred_horizon, self.noise_dim)
        loss, logs = rs_imle_loss(
            joints, fake, self.epsilon, self.epsilon_scale_by_dim
        )
        logs['imle'] = loss.detach()
        if self.model.separate_gripper:
            grip = self.model.gripper(joints, cond)
            grip_l1 = (
                F.l1_loss(
                    grip, action[..., self.noise_dim :], reduction='none'
                )
                .sum((-2, -1))
                .mean()
            )
            logs['gripper_l1'] = grip_l1.detach()
            loss = loss + self.gripper_loss_weight * grip_l1
        self.loss_metrics = logs
        return loss

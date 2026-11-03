"""s3fmp's one-shot conditional U-Net and teacher-forced gripper head."""

import torch
from torch import nn
from .networks import ConditionalUnet1d, UNetConfig, ConvConfig


def unet_config(channels, kernel_size, attention, heads, scale):
    return UNetConfig(
        channels=tuple(channels),
        has_attn=attention,
        nhead=heads,
        cond_pred_scale=scale,
        conv_cfg=ConvConfig(
            activation='Mish',
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        ),
    )


class ActionGenerator(nn.Module):
    def __init__(
        self,
        action_dim,
        pred_horizon,
        obs_horizon,
        feature_dim,
        channels=(64, 128, 256),
        gripper_channels=(32, 64, 128),
        kernel_size=5,
        heads=8,
        separate_gripper=True,
    ):
        super().__init__()
        if pred_horizon % (2 ** (len(channels) - 1)):
            raise ValueError(
                'pred_horizon must be divisible by the generator downsample factor'
            )
        if separate_gripper and pred_horizon % (
            2 ** (len(gripper_channels) - 1)
        ):
            raise ValueError(
                'pred_horizon must be divisible by the gripper downsample factor'
            )
        self.gen_dim = action_dim - int(separate_gripper)
        self.pred_horizon = pred_horizon
        self.separate_gripper = separate_gripper
        # Two cameras are separate conditioning tokens; the reference uses image/state tokens.
        self.policy = ConditionalUnet1d(
            feature_dim * obs_horizon * 2,
            (self.gen_dim, pred_horizon),
            unet_config(channels, kernel_size, True, heads, True),
        )
        if separate_gripper:
            self.gripper_head = ConditionalUnet1d(
                feature_dim,
                (self.gen_dim, pred_horizon),
                unet_config(gripper_channels, 3, False, None, False),
            )
            self.gripper_out = nn.Conv1d(self.gen_dim, 1, kernel_size=1)

    def generate(self, noise, cond):
        return self.policy(
            noise.transpose(-1, -2), cond.flatten(-2, -1)
        ).transpose(-1, -2)

    def gripper(self, joints, cond):
        h = self.gripper_head(joints.transpose(-1, -2), cond.mean(dim=1))
        return self.gripper_out(h).transpose(-1, -2)

    def forward(self, cond, noise):
        traj = self.generate(noise, cond)
        if self.separate_gripper:
            traj = torch.cat([traj, self.gripper(traj, cond)], dim=-1)
        return traj

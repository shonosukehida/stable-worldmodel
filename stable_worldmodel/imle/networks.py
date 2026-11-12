"""1D modules from ml-networks c672fb376b28b96c7c58cf6bbceb351176b0642d.

Only the non-hypernetwork 1D path used by s3fmp is exposed. Numerical
operations, parameter names and initialization order are retained.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from omegaconf import ListConfig
from typing import Any, Dict, Literal, Tuple, Optional
import numpy as np
import torch
from torch import nn
from einops import rearrange
from einops.layers.torch import Rearrange


@dataclass
class ConvConfig:
    activation: str
    kernel_size: int
    stride: int
    padding: int
    output_padding: int = 0
    dilation: int = 1
    groups: int = 1
    bias: bool = True
    padding_mode: Literal['zeros', 'reflect', 'replicate', 'circular'] = (
        'zeros'
    )
    dropout: float = 0.0
    norm: Literal['batch', 'group', 'none'] = 'none'
    norm_cfg: Dict[str, Any] = field(default_factory=dict)
    norm_first: bool = False
    scale_factor: int = 0

    def __post_init__(self) -> None:
        if self.norm == 'none':
            self.norm_cfg = {}
        else:
            self.norm_cfg = dict(**self.norm_cfg)


@dataclass
class UNetConfig:
    channels: Tuple[int, ...]
    conv_cfg: ConvConfig
    cond_pred_scale: bool = False
    nhead: Optional[int] = None
    has_attn: bool = False
    use_shuffle: bool = False
    use_hypernet: bool = False
    hyper_mlp_cfg: Optional[Any] = None

    def __post_init__(self):
        if self.has_attn:
            assert self.nhead is not None, (
                'nhead must be specified when has_attn is True.'
            )
        if isinstance(self.channels, list) or isinstance(
            self.channels, ListConfig
        ):
            self.channels = tuple(self.channels)


class Activation(nn.Module):
    def __init__(self, activation, **kwargs):
        super().__init__()
        if 'glu' not in activation.lower():
            kwargs.pop('dim', None)
        self.activation = getattr(nn, activation)(**kwargs)

    def forward(self, x):
        return self.activation(x)


def get_norm(
    norm: Literal['layer', 'rms', 'group', 'batch2d', 'batch1d', 'none'],
    **kwargs: Any,
) -> nn.Module:
    if norm == 'layer':
        return nn.LayerNorm(**kwargs)
    if norm == 'rms':
        return nn.RMSNorm(**kwargs)
    if norm == 'group':
        return nn.GroupNorm(**kwargs)
    if norm == 'batch2d':
        return nn.BatchNorm2d(**kwargs)
    if norm == 'batch1d':
        return nn.BatchNorm1d(**kwargs)
    return nn.Identity()


class ConvNormActivation1d(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, cfg: ConvConfig
    ) -> None:
        super().__init__()
        out_channels_ = out_channels
        if 'glu' in cfg.activation.lower():
            out_channels_ *= 2
        if cfg.scale_factor > 0:
            out_channels_ *= abs(cfg.scale_factor)
        elif cfg.scale_factor < 0:
            out_channels_ //= abs(cfg.scale_factor)
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels_,
            kernel_size=cfg.kernel_size,
            stride=cfg.stride,
            padding=cfg.padding,
            dilation=cfg.dilation,
            groups=cfg.groups,
            bias=cfg.bias,
            padding_mode=cfg.padding_mode,
        )
        if cfg.norm != 'none' and cfg.norm != 'group':
            cfg.norm_cfg['num_features'] = out_channels_
        elif cfg.norm == 'group':
            cfg.norm_cfg['num_channels'] = (
                in_channels if cfg.norm_first else out_channels_
            )
        if cfg.scale_factor > 0:
            self.horizontal_shuffle = HorizonShuffle(cfg.scale_factor)
        elif cfg.scale_factor < 0:
            self.horizontal_shuffle = HorizonUnShuffle(abs(cfg.scale_factor))
        else:
            self.horizontal_shuffle = nn.Identity()
        self.norm = get_norm(cfg.norm, **cfg.norm_cfg)
        self.activation = Activation(cfg.activation, dim=-2)
        self.dropout = (
            nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()
        )
        self.norm_first = cfg.norm_first

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm_first:
            x = self.norm(x)
            x = self.conv(x)
            x = self.horizontal_shuffle(x)
            x = self.activation(x)
            x = self.dropout(x)
        else:
            x = self.conv(x)
            x = self.horizontal_shuffle(x)
            x = self.norm(x)
            x = self.activation(x)
            x = self.dropout(x)
        return x


class Attention1d(nn.Module):
    def __init__(
        self,
        channels: int,
        nhead: int = None,
        patch_size: int = 1,
        attn_cfg: Any = None,
    ):
        super().__init__()
        self.channels = channels
        if nhead is None:
            assert attn_cfg is not None, (
                'Either nhead or attn_cfg must be provided'
            )
            self.n_heads = attn_cfg.nhead
            self.patch_size = attn_cfg.patch_size
        else:
            self.n_heads = nhead
            self.patch_size = patch_size
        assert channels % self.n_heads == 0, (
            f'q,k,v channels {channels} is not divisible by num_head_channels {nhead}'
        )
        cfg = ConvConfig(
            kernel_size=1,
            padding=0,
            stride=1,
            activation='Identity',
            dropout=0.0,
        )
        first_cfg = cfg
        first_cfg.norm_first = True
        self.qkv = ConvNormActivation1d(channels, channels * 3, first_cfg)
        self.proj_out = ConvNormActivation1d(channels, channels, cfg)

    def qkv_attention(self, qkv):
        (bs, width, length) = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        (q, k, v) = qkv.reshape(bs * self.n_heads, ch * 3, length).split(
            ch, dim=1
        )
        scale = 1 / np.sqrt(np.sqrt(ch))
        weight = torch.einsum('bct,bcs->bts', q * scale, k * scale)
        weight = torch.softmax(
            weight - torch.max(weight, dim=-1, keepdim=True)[0], dim=-1
        )
        a = torch.einsum('bts,bcs->bct', weight, v)
        return a.reshape(bs, -1, length)

    def forward(self, x, *args):
        (b, c, *spatial) = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(x)
        qkv = rearrange(qkv, 'b c (t p) -> b (c p) t', p=self.patch_size)
        h = self.qkv_attention(qkv)
        h = rearrange(h, 'b (c p) t -> b c (t p)', p=self.patch_size)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


class HorizonShuffle(nn.Module):
    def __init__(self, upscale_factor: int = 2):
        super().__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x: torch.Tensor):
        (b, c, t) = x.shape
        assert c % self.upscale_factor == 0, (
            'Input length must be divisible by upscale_factor.'
        )
        x = rearrange(x, 'b (c u) t -> b c (t u)', u=self.upscale_factor)
        return x


class HorizonUnShuffle(nn.Module):
    def __init__(self, downscale_factor: int = 2):
        super().__init__()
        self.downscale_factor = downscale_factor

    def forward(self, x: torch.Tensor):
        (b, c, t) = x.shape
        assert t % self.downscale_factor == 0, (
            'Input length must be divisible by upscale_factor.'
        )
        x = rearrange(x, 'b c (t u) -> b (c u) t', u=self.downscale_factor)
        return x


class ConditionalUnet1d(nn.Module):
    def __init__(
        self, feature_dim: int, obs_shape: tuple[int, int], cfg: UNetConfig
    ):
        super().__init__()
        if cfg.use_hypernet:
            raise ValueError(
                'This port supports only s3fmp IMLE non-hypernetwork U-Nets'
            )
        all_dims = [obs_shape[0]] + list(cfg.channels)
        start_dim = cfg.channels[0]
        self.obs_shape = obs_shape
        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList(
            [
                ConditionalResidualBlock1d(
                    mid_dim,
                    mid_dim,
                    cond_dim=feature_dim,
                    conv_cfg=cfg.conv_cfg,
                    cond_predict_scale=cfg.cond_pred_scale,
                ),
                Attention1d(mid_dim, cfg.nhead)
                if cfg.has_attn
                else nn.Identity(),
                ConditionalResidualBlock1d(
                    mid_dim,
                    mid_dim,
                    cond_dim=feature_dim,
                    conv_cfg=cfg.conv_cfg,
                    cond_predict_scale=cfg.cond_pred_scale,
                ),
            ]
        )
        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= len(in_out) - 1
            down_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1d(
                            dim_in,
                            dim_out,
                            cond_dim=feature_dim,
                            conv_cfg=cfg.conv_cfg,
                            cond_predict_scale=cfg.cond_pred_scale,
                        ),
                        Attention1d(dim_out, cfg.nhead)
                        if cfg.has_attn
                        else nn.Identity(),
                        ConditionalResidualBlock1d(
                            dim_out,
                            dim_out,
                            cond_dim=feature_dim,
                            conv_cfg=cfg.conv_cfg,
                            cond_predict_scale=cfg.cond_pred_scale,
                        ),
                        Downsample1d(dim_out, cfg.use_shuffle)
                        if not is_last
                        else nn.Identity(),
                    ]
                )
            )
        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= len(in_out) - 1
            up_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1d(
                            dim_out * 2,
                            dim_in,
                            cond_dim=feature_dim,
                            conv_cfg=cfg.conv_cfg,
                            cond_predict_scale=cfg.cond_pred_scale,
                        ),
                        Attention1d(dim_in, cfg.nhead)
                        if cfg.has_attn
                        else nn.Identity(),
                        ConditionalResidualBlock1d(
                            dim_in,
                            dim_in,
                            cond_dim=feature_dim,
                            conv_cfg=cfg.conv_cfg,
                            cond_predict_scale=cfg.cond_pred_scale,
                        ),
                        Upsample1d(dim_in, cfg.use_shuffle)
                        if not is_last
                        else nn.Identity(),
                    ]
                )
            )
        final_conv = nn.Sequential(
            ConvNormActivation1d(start_dim, start_dim, cfg.conv_cfg),
            nn.Conv1d(start_dim, obs_shape[0], 1),
        )
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

    def forward(self, base: torch.Tensor, cond: torch.Tensor):
        batch_shape = base.shape[:-2]
        assert base.shape[-2:] == self.obs_shape, (
            f'Input shape {base.shape[-2:]} does not match expected shape {self.obs_shape}'
        )
        base = base.reshape(-1, *self.obs_shape)
        global_feature = cond.reshape(-1, cond.shape[-1])
        x = base
        h = []
        for resnet, attn, resnet2, downsample in self.down_modules:
            x = resnet(x, global_feature)
            x = attn(x)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)
        for mid_module in self.mid_modules:
            if isinstance(mid_module, nn.Identity):
                x = mid_module(x)
            else:
                x = mid_module(x, global_feature)
        for resnet, attn, resnet2, upsample in self.up_modules:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = attn(x)
            x = resnet2(x, global_feature)
            x = upsample(x)
        x = self.final_conv(x)
        x = x.reshape(*batch_shape, *self.obs_shape)
        return x


class ConditionalResidualBlock1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        conv_cfg: ConvConfig,
        cond_predict_scale=False,
    ):
        super().__init__()
        self.first_conv = ConvNormActivation1d(
            in_channels, out_channels, conv_cfg
        )
        self.last_conv = ConvNormActivation1d(
            out_channels, out_channels, conv_cfg
        )
        cond_channels = out_channels
        if cond_predict_scale:
            cond_channels = out_channels * 2
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Linear(cond_dim, cond_channels),
            Rearrange('batch c -> batch c 1'),
        )
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, cond):
        out = self.first_conv(x)
        embed = self.cond_encoder(cond)
        if self.cond_predict_scale:
            embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
            scale = embed[:, 0, ...]
            bias = embed[:, 1, ...]
            out = scale * out + bias
        else:
            out = out + embed
        out = self.last_conv(out)
        out = out + self.residual_conv(x)
        return out


class Downsample1d(nn.Module):
    def __init__(self, dim: int, use_shuffle: bool = False):
        super().__init__()
        if use_shuffle:
            self.conv = nn.Sequential(
                HorizonUnShuffle(2),
                nn.Conv1d(dim * 2, dim, kernel_size=3, stride=1, padding=1),
            )
        else:
            self.conv = nn.Conv1d(dim, dim, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim: int, use_shuffle: bool = False):
        super().__init__()
        if use_shuffle:
            self.conv = nn.Sequential(
                nn.Conv1d(dim, dim * 2, kernel_size=3, stride=1, padding=1),
                HorizonShuffle(2),
            )
        else:
            self.conv = nn.ConvTranspose1d(
                dim, dim, kernel_size=4, stride=2, padding=1
            )

    def forward(self, x):
        return self.conv(x)

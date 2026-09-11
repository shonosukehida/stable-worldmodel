import math

import einops
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2

        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(
            torch.arange(half_dim, device=x.device,) * -emb
        )

        emb = x[:, None] * emb[None, :]

        return torch.cat(
            (emb.sin(), emb.cos(),), dim=-1,
        )


class Downsample1d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()

        self.conv = nn.Conv1d(
            dim,
            dim,
            kernel_size=3,
            stride=2,
            padding=1,
        )

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()

        self.conv = nn.ConvTranspose1d(
            dim,
            dim,
            kernel_size=4,
            stride=2,
            padding=1,
        )

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    def __init__(
        self,
        inp_channels: int,
        out_channels: int,
        kernel_size: int,
        n_groups: int = 8,
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(
                inp_channels,
                out_channels,
                kernel_size,
                padding=kernel_size // 2,
            ),
            nn.GroupNorm(
                n_groups,
                out_channels,
            ),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 3,
        n_groups: int = 8,
        cond_predict_scale: bool = False,
    ):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    n_groups=n_groups,
                ),
                Conv1dBlock(
                    out_channels,
                    out_channels,
                    kernel_size,
                    n_groups=n_groups,
                ),
            ]
        )

        cond_channels = out_channels

        if cond_predict_scale:
            cond_channels = out_channels * 2

        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels

        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(
                cond_dim,
                cond_channels,
            ),
            Rearrange("b t -> b t 1"),
        )

        if in_channels != out_channels:
            self.residual_conv = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=1,
            )
        else:
            self.residual_conv = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:

        out = self.blocks[0](x)

        embed = self.cond_encoder(cond)

        if self.cond_predict_scale:
            embed = embed.reshape(
                embed.shape[0],
                2,
                self.out_channels,
                1,
            )

            scale = embed[:, 0]
            bias = embed[:, 1]

            out = scale * out + bias

        else:
            out = out + embed

        out = self.blocks[1](out)

        return out + self.residual_conv(x)


class ConditionalUnet1D(nn.Module):
    def __init__(
        self,
        input_dim: int,
        local_cond_dim: int | None = None,
        global_cond_dim: int | None = None,
        diffusion_step_embed_dim: int = 256,
        down_dims=(256, 512, 1024),
        kernel_size: int = 5,
        n_groups: int = 8,
        cond_predict_scale: bool = True,
    ):
        super().__init__()

        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(
                diffusion_step_embed_dim
            ),
            nn.Linear(
                diffusion_step_embed_dim,
                diffusion_step_embed_dim * 4,
            ),
            nn.Mish(),
            nn.Linear(
                diffusion_step_embed_dim * 4,
                diffusion_step_embed_dim,
            ),
        )

        cond_dim = diffusion_step_embed_dim

        if global_cond_dim is not None:
            cond_dim += global_cond_dim

        in_out = list(
            zip(
                all_dims[:-1],
                all_dims[1:],
            )
        )

        self.local_cond_encoder = None

        if local_cond_dim is not None:
            _, dim_out = in_out[0]

            self.local_cond_encoder = nn.ModuleList(
                [
                    ConditionalResidualBlock1D(
                        local_cond_dim,
                        dim_out,
                        cond_dim=cond_dim,
                        kernel_size=kernel_size,
                        n_groups=n_groups,
                        cond_predict_scale=cond_predict_scale,
                    ),
                    ConditionalResidualBlock1D(
                        local_cond_dim,
                        dim_out,
                        cond_dim=cond_dim,
                        kernel_size=kernel_size,
                        n_groups=n_groups,
                        cond_predict_scale=cond_predict_scale,
                    ),
                ]
            )

        mid_dim = all_dims[-1]

        self.mid_modules = nn.ModuleList(
            [
                ConditionalResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    cond_dim,
                    kernel_size,
                    n_groups,
                    cond_predict_scale,
                ),
                ConditionalResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    cond_dim,
                    kernel_size,
                    n_groups,
                    cond_predict_scale,
                ),
            ]
        )

        self.down_modules = nn.ModuleList()

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= len(in_out) - 1

            self.down_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_in,
                            dim_out,
                            cond_dim,
                            kernel_size,
                            n_groups,
                            cond_predict_scale,
                        ),
                        ConditionalResidualBlock1D(
                            dim_out,
                            dim_out,
                            cond_dim,
                            kernel_size,
                            n_groups,
                            cond_predict_scale,
                        ),
                        (
                            nn.Identity()
                            if is_last
                            else Downsample1d(dim_out)
                        ),
                    ]
                )
            )

        self.up_modules = nn.ModuleList()

        for ind, (dim_in, dim_out) in enumerate(
            reversed(in_out[1:])
        ):
            is_last = ind >= len(in_out) - 1

            self.up_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_out * 2,
                            dim_in,
                            cond_dim,
                            kernel_size,
                            n_groups,
                            cond_predict_scale,
                        ),
                        ConditionalResidualBlock1D(
                            dim_in,
                            dim_in,
                            cond_dim,
                            kernel_size,
                            n_groups,
                            cond_predict_scale,
                        ),
                        (
                            nn.Identity()
                            if is_last
                            else Upsample1d(dim_in)
                        ),
                    ]
                )
            )

        self.final_conv = nn.Sequential(
            Conv1dBlock(
                start_dim,
                start_dim,
                kernel_size,
            ),
            nn.Conv1d(
                start_dim,
                input_dim,
                kernel_size=1,
            ),
        )

        self.diffusion_step_encoder = (
            diffusion_step_encoder
        )

    def forward(
        self,
        sample: torch.Tensor,
        timestep,
        local_cond=None,
        global_cond=None,
        **kwargs,
    ) -> torch.Tensor:

        # (B, T, D) -> (B, D, T)
        sample = einops.rearrange(
            sample,
            "b h d -> b d h",
        )

        if not torch.is_tensor(timestep):
            timestep = torch.tensor(
                [timestep],
                dtype=torch.long,
                device=sample.device,
            )

        elif timestep.ndim == 0:
            timestep = timestep[None].to(
                sample.device
            )

        timestep = timestep.expand(
            sample.shape[0]
        )

        global_feature = (
            self.diffusion_step_encoder(timestep)
        )

        if global_cond is not None:
            global_feature = torch.cat(
                [global_feature, global_cond,], dim=-1,
            )

        x = sample
        h = []

        for (resnet, resnet2, downsample,) in self.down_modules:

            x = resnet(x, global_feature,)

            x = resnet2(x, global_feature,)

            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature,)

        for (resnet, resnet2, upsample,) in self.up_modules:

            x = torch.cat(
                (x, h.pop(),), dim=1,
            )

            x = resnet(x, global_feature,)

            x = resnet2(x, global_feature,)

            x = upsample(x)

        x = self.final_conv(x)

        # (B, D, T) -> (B, T, D)
        x = einops.rearrange(
            x,
            "b d h -> b h d",
        )

        return x
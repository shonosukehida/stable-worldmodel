"""Run executable reference AST only; no s3fmp training/IO or optional imports."""

from __future__ import annotations

# Imports below populate the reference AST execution namespace.
# ruff: noqa: F401, F403
import ast
from pathlib import Path
from dataclasses import dataclass, field
from typing import *
import math
import numpy as np
import torch
from torch import nn
from einops import rearrange
from einops.layers.torch import Rearrange
from omegaconf import DictConfig, ListConfig

torch.set_num_threads(1)
import argparse

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--s3fmp', type=Path, required=True)
parser.add_argument('--ml-networks', type=Path, required=True)
parser.add_argument(
    '--output',
    type=Path,
    default=Path(__file__).with_name('s3fmp_reference.npz'),
)
args = parser.parse_args()
root = args.ml_networks / 'src/ml_networks'
ns = globals().copy()
selections = {
    'config.py': ['ConvConfig', 'UNetConfig'],
    'activations.py': ['Activation'],
    'layers.py': [
        'get_norm',
        'ConvNormActivation1d',
        'HorizonShuffle',
        'HorizonUnShuffle',
        'Attention1d',
    ],
    'unet.py': [
        'ConditionalUnet1d',
        'ConditionalResidualBlock1d',
        'Downsample1d',
        'Upsample1d',
    ],
}
for file, names in selections.items():
    nodes = [
        n
        for n in ast.parse((root / file).read_text()).body
        if getattr(n, 'name', None) in names
    ]
    exec(
        compile(
            ast.Module(body=nodes, type_ignores=[]), str(root / file), 'exec'
        ),
        ns,
    )
s3 = args.s3fmp / 'src/s3fmp/policy.py'
tree = ast.parse(s3.read_text())
loss = next(n for n in tree.body if getattr(n, 'name', None) == 'rs_imle_loss')
exec(compile(ast.Module(body=[loss], type_ignores=[]), str(s3), 'exec'), ns)
node = next(n for n in tree.body if getattr(n, 'name', None) == 'IMLEPolicy')
methods = [
    n
    for n in node.body
    if getattr(n, 'name', None) in ('_generate', '_gripper')
]
for method in methods:
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(s3), 'exec'),
        ns,
    )


def unet_config(channels, kernel_size, attention, heads, scale):
    return ns['UNetConfig'](
        channels=tuple(channels),
        has_attn=attention,
        nhead=heads,
        cond_pred_scale=scale,
        conv_cfg=ns['ConvConfig'](
            activation='Mish',
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        ),
    )


# Instantiate the actual reference modules in s3fmp's original initialization order.
class Reference(nn.Module):
    def __init__(self):
        super().__init__()
        self.policy = ns['ConditionalUnet1d'](
            128, (7, 16), unet_config((8, 16, 32), 5, True, 8, True)
        )
        self.gripper_head = ns['ConditionalUnet1d'](
            8, (7, 16), unet_config((8, 16, 32), 3, False, None, False)
        )
        self.gripper_out = nn.Conv1d(7, 1, 1)

    generate = ns['_generate']
    gripper = ns['_gripper']


torch.manual_seed(137)
m = Reference()
cond = torch.randn(4, 16, 8)
noise = torch.randn(4, 16, 7)
target = torch.randn(2, 16, 8)
joints = m.generate(noise, cond)
pred = torch.cat([joints, m.gripper(joints, cond)], -1)
loss, _ = ns['rs_imle_loss'](
    target[..., :7], joints.reshape(2, 2, 16, 7), 0.01, True
)
grip = m.gripper(target[..., :7], cond[::2])
loss = (
    loss
    + 0.00625
    * torch.nn.functional.l1_loss(grip, target[..., 7:], reduction='none')
    .sum((-2, -1))
    .mean()
)
loss.backward()
payload = dict(
    cond=cond.numpy(),
    noise=noise.numpy(),
    target=target.numpy(),
    prediction=pred.detach().numpy(),
    loss=loss.detach().numpy(),
)
payload.update(
    {'grad/' + name: p.grad.numpy() for name, p in m.named_parameters()}
)
import hashlib

payload['reference_policy_sha256'] = np.array(
    hashlib.sha256(s3.read_bytes()).hexdigest()
)
payload['reference_unet_sha256'] = np.array(
    hashlib.sha256((root / 'unet.py').read_bytes()).hexdigest()
)
payload['torch_version'] = np.array(str(torch.__version__))
np.savez_compressed(args.output, **payload)
print('Frozen reference loss:', loss.item())

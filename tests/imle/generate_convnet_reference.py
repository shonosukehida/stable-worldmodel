"""Freeze the real ml-networks ConvNet encoder path used by s3fmp.

Read-only AST extraction avoids importing s3fmp training/robot dependencies.
"""

from __future__ import annotations

# Imports populate the executable reference namespace.
# ruff: noqa: F401, F403
import argparse
import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import *
import hashlib
import numpy as np
import torch
from torch import nn
from omegaconf import DictConfig, ListConfig

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--ml-networks', type=Path, required=True)
parser.add_argument(
    '--output',
    type=Path,
    default=Path(__file__).with_name('convnet_reference.npz'),
)
args = parser.parse_args()
root = args.ml_networks / 'src/ml_networks'
ns = globals().copy()
ns['BaseModule'] = nn.Module
# These paths are tested by isinstance in Encoder but are not selected.
for name in ('ViTConfig', 'ResNetConfig', 'MLPConfig', 'SpatialSoftmaxConfig'):
    ns[name] = type(name, (), {})
selected = {
    'config.py': [
        'ConvConfig',
        'ConvNetConfig',
        'LinearConfig',
        'AdaptiveAveragePoolingConfig',
    ],
    'activations.py': ['Activation'],
    'layers.py': ['get_norm', 'ConvNormActivation', 'LinearNormActivation'],
    'utils.py': ['conv_out', 'conv_out_shape'],
    'vision.py': ['ConvNet', 'Encoder'],
}
for file, names in selected.items():
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
backbone = ns['ConvNetConfig'](
    channels=(16, 32, 64),
    conv_cfgs=tuple(
        ns['ConvConfig'](activation='Mish', kernel_size=3, stride=2, padding=1)
        for _ in range(3)
    ),
)
fc = ns['AdaptiveAveragePoolingConfig'](
    additional_layer=ns['LinearConfig'](activation='Identity')
)
torch.set_num_threads(1)
torch.manual_seed(137)
encoder = ns['Encoder'](128, (3, 96, 128), backbone, fc)
pixels = (
    torch.linspace(-1, 1, 2 * 3 * 96 * 128)
    .reshape(2, 3, 96, 128)
    .requires_grad_()
)
features = encoder(pixels)
features.square().mean().backward()
payload = dict(
    pixels=pixels.detach().numpy(),
    features=features.detach().numpy(),
    input_grad=pixels.grad.numpy(),
    torch_version=np.array(str(torch.__version__)),
)
for index, parameter in enumerate(encoder.parameters()):
    payload[f'parameter_{index}'] = parameter.detach().numpy()
    payload[f'gradient_{index}'] = parameter.grad.numpy()
for file in ('vision.py', 'layers.py', 'config.py'):
    payload[f'sha256_{file}'] = np.array(
        hashlib.sha256((root / file).read_bytes()).hexdigest()
    )
np.savez_compressed(args.output, **payload)
print(
    'Reference ConvNet parameter count:',
    sum(p.numel() for p in encoder.parameters()),
)

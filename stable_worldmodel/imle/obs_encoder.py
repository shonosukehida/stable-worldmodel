"""s3fmp IMLE image ConvNet, without a runtime ml-networks dependency.

Reference: models/cfg/xarm/imle.yaml; ml-networks c672fb3
vision.Encoder/ConvNet and layers.ConvNormActivation/LinearNormActivation.
The configured reference path has no normalization, dropout, or attention.
"""

from torch import nn


def convnet_parameters(config=None):
    """Resolve all supported architecture parameters for checkpointing."""
    values = dict(
        channels=[16, 32, 64],
        kernel_sizes=[3, 3, 3],
        strides=[2, 2, 2],
        paddings=[1, 1, 1],
        activation='Mish',
        pool_size=1,
    )
    config = dict(config or {})
    unknown = set(config) - set(values)
    if unknown:
        raise ValueError(f'Unknown IMLE ConvNet parameters: {sorted(unknown)}')
    values.update(config)
    for name in ('channels', 'kernel_sizes', 'strides', 'paddings'):
        values[name] = list(values[name])
    depth = len(values['channels'])
    if depth < 1 or any(
        len(values[key]) != depth
        for key in ('kernel_sizes', 'strides', 'paddings')
    ):
        raise ValueError(
            'ConvNet channels/kernel_sizes/strides/paddings must have equal nonzero length'
        )
    for key in ('channels', 'kernel_sizes', 'strides', 'paddings'):
        minimum = 0 if key == 'paddings' else 1
        if any(
            isinstance(x, bool) or not isinstance(x, int) or x < minimum
            for x in values[key]
        ):
            raise ValueError(f'Invalid ConvNet {key}')
    if values['activation'] not in (
        'Mish',
        'ReLU',
        'GELU',
        'SiLU',
        'Identity',
    ):
        raise ValueError('Unsupported ConvNet activation')
    if (
        isinstance(values['pool_size'], bool)
        or not isinstance(values['pool_size'], int)
        or values['pool_size'] < 1
    ):
        raise ValueError('ConvNet pool_size must be a positive integer')
    return values


class ConvNetObsEncoder(nn.Module):
    def __init__(self, feature_dim=128, **architecture):
        super().__init__()
        architecture = convnet_parameters(architecture)
        self.architecture = architecture
        self.feature_dim = feature_dim
        layers = []
        in_channels = 3
        for channels, kernel, stride, padding in zip(
            architecture['channels'],
            architecture['kernel_sizes'],
            architecture['strides'],
            architecture['paddings'],
        ):
            layers.extend(
                [
                    nn.Conv2d(
                        in_channels,
                        channels,
                        kernel,
                        stride,
                        padding,
                        bias=True,
                    ),
                    getattr(nn, architecture['activation'])(),
                ]
            )
            in_channels = channels
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(architecture['pool_size'])
        self.flatten = nn.Flatten()
        self.projection = nn.Linear(
            in_channels * architecture['pool_size'] ** 2, feature_dim
        )

    def output_shape(self):
        return (self.feature_dim,)

    def forward(self, pixels):
        if pixels.ndim != 4 or pixels.shape[1] != 3:
            raise ValueError('ConvNet encoder expects RGB images (B,3,H,W)')
        return self.projection(self.flatten(self.pool(self.conv(pixels))))

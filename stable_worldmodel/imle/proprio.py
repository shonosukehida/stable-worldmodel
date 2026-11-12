"""IMLE-only Cartesian state statistics; raw quaternions are never altered."""
import torch

LAYOUT = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw', 'gripper']
CONVENTIONS = dict(proprio_dim=8, proprio_normalization='zscore',
                   proprio_layout=LAYOUT, proprio_position_units='m',
                   proprio_quaternion_order='xyzw',
                   proprio_gripper_convention='0=open,1=closed')


def validate_proprio_stats(stats):
    if not isinstance(stats, dict):
        raise ValueError('IMLE use_proprio=true requires checkpoint proprio statistics')
    for key, expected in CONVENTIONS.items():
        if stats.get(key) != expected:
            raise ValueError(f'Invalid IMLE proprio metadata: {key}')
    result = dict(stats)
    for key in ('proprio_mean', 'proprio_std'):
        if key not in stats:
            raise ValueError(f'Missing IMLE proprio statistic: {key}')
        value = torch.as_tensor(stats[key]).detach().cpu().float().clone()
        if value.shape != (8,) or not torch.isfinite(value).all():
            raise ValueError(f'{key} must contain eight finite values')
        if key == 'proprio_std' and (value <= 0).any():
            raise ValueError('proprio_std must be positive')
        result[key] = value
    return result


def fit_proprio_stats(train_values):
    x = torch.as_tensor(train_values, dtype=torch.float64)
    if x.ndim < 2 or x.shape[-1] != 8:
        raise ValueError("Train proprio must have eight components per observation")
    x = x.reshape(-1, 8)
    if not len(x) or not torch.isfinite(x).all():
        raise ValueError('IMLE Train proprio must be nonempty and finite')
    std = x.std(0, correction=0)
    stats = dict(CONVENTIONS, proprio_mean=x.mean(0).float(),
                 proprio_std=torch.where(std < 1e-4, 1., std).float())
    return validate_proprio_stats(stats)


def normalize_proprio(values, stats, device=None, dtype=torch.float32):
    x = torch.as_tensor(values, device=device, dtype=dtype)
    if x.shape[-1] != 8 or not torch.isfinite(x).all():
        raise ValueError('IMLE proprio must have eight finite components')
    result = (x - stats['proprio_mean'].to(x)) / stats['proprio_std'].to(x)
    if not torch.isfinite(result).all():
        raise ValueError('Nonfinite normalized IMLE proprio')
    return result


def inverse_proprio(values, stats):
    return values * stats['proprio_std'].to(values) + stats['proprio_mean'].to(values)

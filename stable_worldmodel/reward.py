"""Terminal latent rewards; the default retains the original reduction order."""
import math
from numbers import Real

import torch


def _validate_weight(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
        raise ValueError(f'{name} must be a finite nonnegative number')
    return float(value)


def validate_lambda_prop(value):
    return _validate_weight(value, 'lambda_prop')


def _terminal_squared(rollout_output, image_dim):
    pred = rollout_output['predicted_emb'][:, :, -1]
    goal = rollout_output['goal_emb'][:, -1]
    if (pred.ndim != 3 or goal.ndim != 2 or pred.shape[0] != goal.shape[0]
            or pred.shape[-1] != goal.shape[-1]):
        raise ValueError('Expected matching terminal predicted (B,K,D) and goal (B,D) latents')
    if isinstance(image_dim, bool) or not isinstance(image_dim, int) or not 0 < image_dim < pred.shape[-1]:
        raise ValueError('image_dim must explicitly separate nonempty image and proprio latent blocks')
    return (pred - goal[:, None]) ** 2


def latent_goal_costs(rollout_output, image_dim):
    """Legacy image/proprio decomposition, retaining its original reduction."""
    squared = _terminal_squared(rollout_output, image_dim)
    return squared[..., :image_dim].sum(-1), squared[..., image_dim:].sum(-1)


def latent_goal_modality_costs(rollout_output, overhead_dim, image_dim):
    """Squared L2 costs in concatenation order: overhead, wrist, proprio."""
    squared = _terminal_squared(rollout_output, image_dim)
    if (isinstance(overhead_dim, bool) or not isinstance(overhead_dim, int)
            or not 0 < overhead_dim < image_dim):
        raise ValueError('overhead_dim must explicitly separate nonempty overhead and wrist latent blocks')
    return (squared[..., :overhead_dim].sum(-1),
            squared[..., overhead_dim:image_dim].sum(-1),
            squared[..., image_dim:].sum(-1))


def latent_goal_reward(rollout_output: dict, info_dict: dict, lambda_prop=1.0,
                       image_dim=None, *, lambda_overhead=1.0, lambda_wrist=1.0,
                       overhead_dim=None) -> torch.Tensor:
    prop = validate_lambda_prop(lambda_prop)
    overhead = _validate_weight(lambda_overhead, 'lambda_overhead')
    wrist = _validate_weight(lambda_wrist, 'lambda_wrist')
    if overhead != 1.0 or wrist != 1.0:
        overhead_cost, wrist_cost, prop_cost = latent_goal_modality_costs(
            rollout_output, overhead_dim, image_dim)
        return -(overhead * overhead_cost + wrist * wrist_cost + prop * prop_cost)
    if prop != 1.0:
        # Preserve the arithmetic of existing lambda_prop-only configurations.
        image_cost, prop_cost = latent_goal_costs(rollout_output, image_dim)
        return -(image_cost + prop * prop_cost)
    # Keep the original arithmetic, including reduction over all dimensions.
    pred_emb = rollout_output['predicted_emb']
    goal_emb = rollout_output['goal_emb']
    pred_final = pred_emb[:, :, -1]
    goal_final = goal_emb[:, -1]
    return -((pred_final - goal_final[:, None]) ** 2).sum(dim=-1)


def _output_dim(module):
    # The current JEPA projectors/prop encoder are MLPs ending in Linear.
    if isinstance(module, torch.nn.Linear):
        return module.out_features
    net = getattr(module, 'net', None)
    if isinstance(net, torch.nn.Sequential) and isinstance(net[-1], torch.nn.Linear):
        return net[-1].out_features
    raise ValueError('Cannot determine latent boundary: expected a projector/prop MLP ending in Linear')


def model_latent_modality_dimensions(model):
    try:
        overhead = _output_dim(model.overhead_projector)
        wrist = _output_dim(model.wrist_projector)
        prop = _output_dim(model.prop_encoder)
    except AttributeError as error:
        raise ValueError('Cannot determine image/proprio latent boundary from this WM structure') from error
    if min(overhead, wrist, prop) < 1:
        raise ValueError('Image and proprio latent blocks must be nonempty')
    return overhead, wrist, prop


def model_latent_dimensions(model):
    """Retain the public image-boundary/total-dimension API."""
    overhead, wrist, prop = model_latent_modality_dimensions(model)
    return overhead + wrist, overhead + wrist + prop


class LatentGoalReward:
    """Configured two-argument reward callable, without changing GPCPolicy APIs.

    The second positional argument remains lambda_prop for existing callers.
    """
    def __init__(self, model, lambda_prop=1.0, *, lambda_overhead=1.0, lambda_wrist=1.0):
        self.lambda_prop = validate_lambda_prop(lambda_prop)
        self.lambda_overhead = _validate_weight(lambda_overhead, 'lambda_overhead')
        self.lambda_wrist = _validate_weight(lambda_wrist, 'lambda_wrist')
        self.overhead_dim = self.wrist_dim = self.prop_dim = None
        self.image_dim = self.latent_dim = None
        try:
            self.overhead_dim, self.wrist_dim, self.prop_dim = model_latent_modality_dimensions(model)
            self.image_dim = self.overhead_dim + self.wrist_dim
            self.latent_dim = self.image_dim + self.prop_dim
        except ValueError:
            if self._weighted:
                raise

    @property
    def _weighted(self):
        return any(weight != 1.0 for weight in
                   (self.lambda_overhead, self.lambda_wrist, self.lambda_prop))

    def __call__(self, rollout_output, info_dict):
        if self._weighted:
            for name in ('predicted_emb', 'goal_emb'):
                if rollout_output[name].shape[-1] != self.latent_dim:
                    raise ValueError('WM latent dimension differs from the inferred model structure')
        return latent_goal_reward(
            rollout_output, info_dict, self.lambda_prop, self.image_dim,
            lambda_overhead=self.lambda_overhead, lambda_wrist=self.lambda_wrist,
            overhead_dim=self.overhead_dim)


def latent_reward_metadata(policy):
    reward = policy.reward_fn
    return dict(lambda_prop=getattr(reward, 'lambda_prop', 1.0),
                lambda_overhead=getattr(reward, 'lambda_overhead', 1.0),
                lambda_wrist=getattr(reward, 'lambda_wrist', 1.0),
                image_latent_dimension=getattr(reward, 'image_dim', None),
                overhead_latent_dimension=getattr(reward, 'overhead_dim', None),
                wrist_latent_dimension=getattr(reward, 'wrist_dim', None),
                proprio_latent_dimension=getattr(reward, 'prop_dim', None),
                reward='negative terminal (lambda_overhead * overhead squared L2 + '
                       'lambda_wrist * wrist squared L2 + lambda_prop * proprio squared L2)')

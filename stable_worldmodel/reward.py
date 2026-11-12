"""Terminal latent rewards; the default retains the original reduction order."""
import math
from numbers import Real

import torch


def validate_lambda_prop(value):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
        raise ValueError('lambda_prop must be a finite nonnegative number (recommended range: 0..1)')
    return float(value)


def latent_goal_costs(rollout_output, image_dim):
    pred = rollout_output['predicted_emb'][:, :, -1]
    goal = rollout_output['goal_emb'][:, -1]
    if (pred.ndim != 3 or goal.ndim != 2 or pred.shape[0] != goal.shape[0]
            or pred.shape[-1] != goal.shape[-1]):
        raise ValueError('Expected matching terminal predicted (B,K,D) and goal (B,D) latents')
    if isinstance(image_dim, bool) or not isinstance(image_dim, int) or not 0 < image_dim < pred.shape[-1]:
        raise ValueError('image_dim must explicitly separate nonempty image and proprio latent blocks')
    squared = (pred - goal[:, None]) ** 2
    return squared[..., :image_dim].sum(-1), squared[..., image_dim:].sum(-1)


def latent_goal_reward(rollout_output: dict, info_dict: dict, lambda_prop=1.0,
                       image_dim=None) -> torch.Tensor:
    weight = validate_lambda_prop(lambda_prop)
    if weight != 1.0:
        image_cost, prop_cost = latent_goal_costs(rollout_output, image_dim)
        return -(image_cost + weight * prop_cost)
    # Keep the original arithmetic, including reduction over all dimensions.
    pred_emb = rollout_output["predicted_emb"]
    goal_emb = rollout_output["goal_emb"]
    pred_final = pred_emb[:, :, -1]
    goal_final = goal_emb[:, -1]
    rewards = -((pred_final - goal_final[:, None]) ** 2).sum(dim=-1)
    return rewards


def _output_dim(module):
    # The current JEPA projectors/prop encoder are MLPs ending in Linear.
    if isinstance(module, torch.nn.Linear):
        return module.out_features
    net = getattr(module, 'net', None)
    if isinstance(net, torch.nn.Sequential) and isinstance(net[-1], torch.nn.Linear):
        return net[-1].out_features
    raise ValueError('Cannot determine latent boundary: expected a projector/prop MLP ending in Linear')


def model_latent_dimensions(model):
    try:
        overhead = _output_dim(model.overhead_projector)
        wrist = _output_dim(model.wrist_projector)
        prop = _output_dim(model.prop_encoder)
    except AttributeError as error:
        raise ValueError('Cannot determine image/proprio latent boundary from this WM structure') from error
    if min(overhead, wrist, prop) < 1:
        raise ValueError('Image and proprio latent blocks must be nonempty')
    return overhead + wrist, overhead + wrist + prop


class LatentGoalReward:
    """Configured two-argument reward callable, without changing GPCPolicy APIs."""
    def __init__(self, model, lambda_prop=1.0):
        self.lambda_prop = validate_lambda_prop(lambda_prop)
        self.image_dim = self.latent_dim = None
        try:
            self.image_dim, self.latent_dim = model_latent_dimensions(model)
        except ValueError:
            if self.lambda_prop != 1.0:
                raise

    def __call__(self, rollout_output, info_dict):
        if self.lambda_prop != 1.0:
            for name in ('predicted_emb', 'goal_emb'):
                if rollout_output[name].shape[-1] != self.latent_dim:
                    raise ValueError('WM latent dimension differs from the inferred model structure')
        return latent_goal_reward(rollout_output, info_dict, self.lambda_prop, self.image_dim)


def latent_reward_metadata(policy):
    reward = policy.reward_fn
    return dict(lambda_prop=getattr(reward, 'lambda_prop', 1.0),
                image_latent_dimension=getattr(reward, 'image_dim', None),
                reward='negative terminal (image squared L2 + lambda_prop * proprio squared L2)')

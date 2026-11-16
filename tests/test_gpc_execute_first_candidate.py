"""Hardware-free selection regression, including the exact previous GPC code."""
import ast
from pathlib import Path
import subprocess
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from stable_worldmodel.policy import GPCPolicy

BASELINE_COMMIT = "01cfc9aa649776cac2b0bba3c0184727c0004022"


def baseline_class():
    root = Path(__file__).resolve().parents[1]
    source = subprocess.check_output(
        ['git', 'show', f'{BASELINE_COMMIT}:stable_worldmodel/policy.py'], cwd=root, text=True)
    node = next(n for n in ast.parse(source).body
                if isinstance(n, ast.ClassDef) and n.name == 'GPCPolicy')
    ns = dict(torch=torch, np=np, BasePolicy=object, DiffusionPolicy=object)
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<baseline>', 'exec'), ns)
    return ns['GPCPolicy']


def make_policy(cls=GPCPolicy, **kwargs):
    candidates = torch.arange(2*4*6*8, dtype=torch.float32).reshape(2, 4, 6, 8)
    dp = Mock(obs_horizon=3, action_horizon=3)
    dp.sample_action_sequences = Mock(return_value=candidates.clone())
    reward = Mock(return_value=torch.tensor([[1., 4., 4., -2.], [-1., 0., 8., 7.]]))
    policy = cls(dp, None, reward, num_candidates=4, **kwargs)
    policy.normalize_action_for_world_model = Mock(side_effect=lambda a: a / 10)
    policy.rollout = Mock(return_value={'all_candidates': True})
    return policy, candidates


@pytest.mark.parametrize('explicit', [False, True])
def test_disabled_matches_previous_selection_and_output_exactly(explicit):
    baseline, _ = make_policy(baseline_class())
    policy, _ = make_policy(**({'execute_first_candidate': False} if explicit else {}))
    np.testing.assert_array_equal(policy.get_action_chunk({}), baseline.get_action_chunk({}))
    np.testing.assert_array_equal(policy.get_action({}), baseline.get_action({}))
    assert policy.best_idx.tolist() == [1, 2]  # First maximum wins ties.
    assert torch.equal(policy.best_idx, policy.selected_candidate_index)
    torch.testing.assert_close(policy.rollout.call_args.kwargs['action_sequences'],
                               baseline.rollout.call_args.kwargs['action_sequences'], rtol=0, atol=0)


def test_first_candidate_still_generates_rolls_out_and_scores_every_candidate():
    policy, candidates = make_policy(execute_first_candidate=True)
    result = policy.get_action_chunk({'observation': 'same'})
    np.testing.assert_array_equal(result, candidates[:, 0, 2:5].numpy())
    assert policy.best_idx.tolist() == [1, 2]
    assert policy.selected_candidate_index.tolist() == [0, 0]
    policy.diffusion_policy.sample_action_sequences.assert_called_once_with(
        {'observation': 'same'}, num_samples=4, denormalize=True)
    policy.rollout.assert_called_once()
    torch.testing.assert_close(policy.rollout.call_args.kwargs['action_sequences'],
                               candidates[:, :, 2:5] / 10, rtol=0, atol=0)
    policy.reward_fn.assert_called_once_with({'all_candidates': True}, {'observation': 'same'})


def test_first_candidate_does_not_bypass_reward_validation():
    policy, _ = make_policy(execute_first_candidate=True)
    policy.reward_fn.return_value = torch.ones(2, 1)
    with pytest.raises(ValueError, match='reward_fn must return shape'):
        policy.get_action_chunk({})

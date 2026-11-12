"""Portable CPU tests for the IMLE state path, without robot dependencies."""
import pytest
import torch
from torch import nn
from stable_worldmodel.imle import ActionGenerator, IMLEPolicy
from stable_worldmodel.imle.proprio import fit_proprio_stats, normalize_proprio


def test_state_condition_backward_and_seed():
    torch.set_num_threads(1)
    stats=fit_proprio_stats(torch.arange(32).reshape(4,8).float())
    encoder=lambda:nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(3,8))
    policy=IMLEPolicy(
        ActionGenerator(8,16,8,8,channels=(8,16,32),gripper_channels=(8,16,32),use_proprio=True),
        encoder(),encoder(),16,8,8,8,use_proprio=True,proprio_stats=stats,n_samples_per_condition=2,
    )
    batch=dict(pixels=torch.randn(1,8,3,8,8),wrist_pixels=torch.randn(1,8,3,8,8),
               proprio=torch.randn(1,8,8),action=torch.randn(1,16,8))
    cond=policy._condition(batch['pixels'],batch['wrist_pixels'],batch['proprio'])
    assert cond.shape==(1,8,24)
    torch.testing.assert_close(cond[...,-8:],normalize_proprio(batch['proprio'],stats),rtol=0,atol=0)
    loss=policy.compute_loss(batch); loss.backward()
    assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in policy.model.parameters())
    torch.manual_seed(5); expected=policy.predict_action(batch); rng=torch.get_rng_state()
    torch.manual_seed(5); actual=policy.predict_action(batch)
    torch.testing.assert_close(expected['action_pred'],actual['action_pred'],rtol=0,atol=0)
    assert torch.equal(rng,torch.get_rng_state())
    assert actual['action'].shape==(1,8,8)


@pytest.mark.parametrize('value',[None,{},dict(proprio_dim=7)])
def test_missing_or_invalid_state_statistics(value):
    from stable_worldmodel.imle.proprio import validate_proprio_stats
    with pytest.raises(ValueError):validate_proprio_stats(value)

import pytest
import torch
from torch.nn import functional as F

from stable_worldmodel.wm.diamond.action_embedding import cartesian_embedding
from stable_worldmodel.wm.diamond.model import DiamondWorldModel
from stable_worldmodel.wm.diamond.preprocessing import pack_views, unpack_views


def test_embedding_and_letterbox():
    emb = cartesian_embedding()
    idx = torch.tensor([[1, 4, 2, 7]])
    assert torch.equal(emb(F.one_hot(idx, 8).float()), emb[0].weight.T[idx].flatten(1))
    action = torch.ones(2, 4, 8, requires_grad=True)
    emb(action).square().sum().backward()
    assert action.grad.abs().sum() > 0 and emb[0].weight.grad.abs().sum() > 0
    views = ["pixels", "wrist_pixels"]
    raw = {k: torch.tensor([i*100+10, i*100+20, i*100+30]).view(3, 1, 1).expand(3, 48, 64)
           for i, k in enumerate(views)}
    packed = pack_views(raw, views)
    assert packed.shape == (6, 64, 64) and (packed[:, :8] == -1).all()
    assert torch.equal(packed[:, 10, 10], torch.cat([raw[k][:, 0, 0] for k in views]).float()/255*2-1)
    for k, x in unpack_views(packed, views, (48, 64)).items():
        assert torch.equal(x, raw[k].byte())


@pytest.mark.parametrize("views", [("pixels",), ("pixels", "wrist_pixels")])
def test_cartesian_model_and_rollout(views):
    torch.manual_seed(2)
    model = DiamondWorldModel(views).eval()
    inner = model.denoiser.inner_model
    c = len(views)*3
    assert inner.conv_in.in_channels == 5*c and inner.conv_out.out_channels == c
    with torch.no_grad():
        for name, p in inner.named_parameters():
            if name.endswith(("conv_out.weight", "conv2.weight")):
                p.normal_(std=0.02)
    obs = torch.rand(1, 4, c, 64, 64)*2-1
    past, future = torch.randn(1, 3, 8), torch.randn(1, 2, 8)
    def gen():
        return torch.Generator().manual_seed(444)
    state = torch.get_rng_state()
    first = model.predict_next(obs, torch.cat((past, future[:, :1]), 1), gen())
    g = gen()
    out = model.rollout_images(obs, past, future, g)
    assert torch.equal(out[:, 0], first)
    g = gen()
    window = torch.cat((past, future[:, :1]), 1)
    x = model.predict_next(obs, window, g)
    y = model.predict_next(torch.cat((obs[:, 1:], x[:, None]), 1),
                           torch.cat((window[:, 1:], future[:, 1:]), 1), g)
    assert torch.equal(out[:, 1], y) and not out.requires_grad
    assert not torch.equal(out, model.rollout_images(obs, past, -future, gen()))
    assert torch.equal(torch.get_rng_state(), state)
    assert torch.equal(out, model.rollout_images(obs, past, future[:, None], gen())[:, 0])
    several = model.rollout_images(obs, past, future[:, None].expand(-1, 3, -1, -1), gen())
    assert several.shape == (1, 3, 2, c, 64, 64)
    paired_obs = torch.cat((obs, -obs))
    paired_past = torch.cat((past, -past))
    f = torch.stack((future[:, :1], -future[:, :1]), dim=1).repeat(2, 1, 1, 1)
    multi = model.rollout_images(paired_obs, paired_past, f, gen())
    flat = model.rollout_images(paired_obs.repeat_interleave(2, 0), paired_past.repeat_interleave(2, 0),
                               f.flatten(0, 1), gen())
    assert torch.equal(multi.flatten(0, 1), flat)
    for invalid in (future[..., :7], future[:, :0], future.double(), future*float("nan")):
        with pytest.raises(ValueError):
            model.rollout_images(obs, past, invalid)

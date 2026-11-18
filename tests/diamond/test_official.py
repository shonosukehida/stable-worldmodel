"""Independent execution of the pinned source bytes. The comparison tolerance is zero."""
import builtins
import hashlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import torch
from torch import nn

from stable_worldmodel.wm import diamond
from stable_worldmodel.wm.diamond.upstream.diffusion import denoiser, diffusion_sampler, inner_model
from stable_worldmodel.wm.diamond.upstream.types import Batch


def reference(monkeypatch):
    root = Path(diamond.__file__).parent
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["commit"] == "5bcd1599755b4f2fae8e5e079e02f0728e174965"
    for name, digest in manifest["runtime_sha256"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
    for name, hashes in manifest["sources"].items():
        path = root / "source" / (name + ".source" if name.endswith(".py") else name)
        data = path.read_bytes()
        assert hashlib.sha256(data).hexdigest() == hashes["sha256"]
        assert hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest() == hashes["git_blob"]

    # Only annotation imports are stubbed. Relative imports resolve to independent modules.
    def upstream_import(name, *args):
        if name == "data":
            return SimpleNamespace(Batch=Batch)
        if name == "utils":
            return SimpleNamespace(LossAndLogs=tuple)
        return builtins.__import__(name, *args)

    mods = {}
    for suffix in ["", ".diffusion", ".blocks", ".diffusion.inner_model",
                   ".diffusion.denoiser", ".diffusion.diffusion_sampler"]:
        name = "_diamond_oracle" + suffix
        mod = ModuleType(name)
        monkeypatch.setitem(sys.modules, name, mod)
        if suffix in ("", ".diffusion"):
            mod.__path__ = []
            continue
        mod.__package__ = name.rsplit(".", 1)[0]
        mod.__dict__["__builtins__"] = dict(vars(builtins), __import__=upstream_import)
        path = root / "source/src/models" / (suffix[1:].replace(".", "/") + ".py.source")
        exec(compile(path.read_text(), str(path), "exec"), mod.__dict__)
        mods[name.split(".")[-1]] = mod
    return SimpleNamespace(**mods)


def build(api):
    cfg = api.inner_model.InnerModelConfig(3, 4, 256, [2]*4, [64]*4, [0]*4, 18)
    net = api.denoiser.Denoiser(api.denoiser.DenoiserConfig(cfg, 0.5, 0.3))
    net.setup_training(api.denoiser.SigmaDistributionConfig(-0.4, 1.2, 0.002, 20))
    return net


def exact(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype
    assert torch.isfinite(a).all()
    assert torch.equal(a.contiguous().view(torch.uint8).flatten() if a.ndim else a.reshape(1).view(torch.uint8),
                       b.contiguous().view(torch.uint8).flatten() if b.ndim else b.reshape(1).view(torch.uint8))


def compare_forward(net, batch):
    trace = []
    targets, predictions, coefficients = [], [], []
    noise = net.apply_noise
    wrap = net.wrap_model_output
    def record_noise(x, sigma, offset):
        targets.append(x.clone())
        return noise(x, sigma, offset)
    def record_reconstruction(x, out, cs):
        result = wrap(x, out, cs)
        assert not result.requires_grad
        predictions.append(result.clone())
        coefficients.append((x.clone(), out, cs))
        return result
    def record_inputs(module, inputs):
        trace.append(tuple(arg.detach().clone() for arg in inputs))
    net.apply_noise = record_noise
    net.wrap_model_output = record_reconstruction
    hook = net.inner_model.register_forward_pre_hook(record_inputs)
    torch.manual_seed(37)
    loss = net(batch)[0]
    hook.remove()
    del net.apply_noise, net.wrap_model_output
    losses = []
    for i, (x, out, cs) in enumerate(coefficients):
        exact(targets[i], batch.obs[:, i+4])
        exact(trace[i][3], batch.act[:, i:i+4])
        past = batch.obs[:, i:i+4].clone()
        if i:
            past[:, -1] = predictions[0]
        exact(trace[i][2], past.flatten(1, 2) / 0.5)
        valid = batch.mask_padding[:, 4+i]
        target = (targets[i] - cs.c_skip * x) / cs.c_out
        losses.append(torch.nn.functional.mse_loss(out[valid], target[valid]))
    exact(loss, (losses[0] + losses[1]) / 2)
    assert not torch.equal(predictions[0], batch.obs[:, 4])
    assert torch.autograd.grad(losses[1], coefficients[0][1], allow_unused=True, retain_graph=True)[0] is None
    loss.backward()
    return [loss.detach(), *targets, *predictions, *(v for step in trace for v in step)]


def test_official_rgb_discrete_fp32(monkeypatch):
    oracle = reference(monkeypatch)
    local = SimpleNamespace(denoiser=denoiser, inner_model=inner_model, diffusion_sampler=diffusion_sampler)
    torch.manual_seed(123)
    a, b = build(oracle), build(local)
    # Exercise the conditioning path that zero-initialized residual/output weights hide.
    with torch.no_grad():
        for name, weight in a.named_parameters():
            if name.endswith(("conv2.weight", "conv_out.weight", "out_proj.weight")):
                nn.init.normal_(weight, std=0.01)
    b.load_state_dict(a.state_dict(), strict=True)
    obs = torch.rand(2, 6, 3, 64, 64)*2-1
    act = torch.randint(18, (2, 6))
    mask = torch.ones(2, 6, dtype=torch.bool)
    mask[0, 4] = mask[1, 5] = False
    batch = Batch(obs, act, mask)
    results = []
    for api, net in ((oracle, a), (local, b)):
        torch.manual_seed(17)
        sigma = net.sample_sigma_training(2, obs.device)
        zero = torch.zeros_like(obs[:, 4])
        offset = net.apply_noise(zero, sigma*0, 0.3)
        exact(offset, offset[:, :, :1, :1].expand_as(offset))
        noisy = net.apply_noise(obs[:, 4], sigma, 0.3)
        cs = net.compute_conditioners(sigma)
        past, actions = obs[:, :4].flatten(1, 2), act[:, :4]
        features = []
        def capture_unet(module, inputs, result):
            image, downs, ups = result
            features.append(image.detach())
            features.extend(x.detach() for level in downs + ups for x in level)
        hook = net.inner_model.unet.register_forward_hook(capture_unet)
        raw = net.compute_model_output(noisy, past, actions, cs)
        hook.remove()
        prequant = noisy*cs.c_skip + raw*cs.c_out
        denoised = net.denoise(noisy, sigma, past, actions)
        exact(denoised, prequant.clamp(-1, 1).add(1).div(2).mul(255).byte().div(255).mul(2).sub(1))
        outputs = [sigma, offset, noisy, *vars(cs).values(), net.inner_model.act_emb(actions),
                   *features, raw.detach(), prequant.detach(), denoised, *compare_forward(net, batch)]
        sampler = api.diffusion_sampler.DiffusionSampler(net, api.diffusion_sampler.DiffusionSamplerConfig(3))
        torch.manual_seed(543)
        unit_noise = torch.randn(obs[:, 0].shape, generator=torch.Generator().manual_seed(543))
        history = obs[:, :4]
        for i in range(3):
            sample, trajectory = sampler.sample(history, act[:, i:i+4])
            if i == 0:
                exact(unit_noise, trajectory[0])
            outputs.extend(trajectory)
            history = torch.cat((history[:, 1:], sample[:, None]), dim=1)
        results.append(outputs)
    for x, y in zip(*results, strict=True):
        exact(x, y)
    for (n, p), (m, q) in zip(a.named_parameters(), b.named_parameters(), strict=True):
        assert n == m and p.grad is not None
        exact(p.grad, q.grad)
    assert a.inner_model.act_emb[0].weight.grad.abs().sum() > 0

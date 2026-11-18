"""DIAMOND v1 checkpoints. Load only trusted training files (optimizer + Python/NumPy RNG)."""
import json
import os
import platform
import random
import tempfile
from pathlib import Path

import numpy as np
import torch

from .action_embedding import validate_normalizer
from .model import ALIGNMENT, DiamondWorldModel, model_spec


def manifest():
    return json.loads(Path(__file__).with_name("manifest.json").read_text())


def capture_rng(generator, device):
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                windows=generator.get_state(), cuda=torch.cuda.get_rng_state(device) if device.type == "cuda" else None)


def restore_rng(state, generator, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    generator.set_state(state["windows"])
    if device.type == "cuda":
        if state["cuda"] is None:
            raise ValueError("No training GPU RNG in checkpoint")
        torch.cuda.set_rng_state(state["cuda"], device)


def validate(payload):
    required = {"version", "model_type", "upstream_sha", "manifest", "spec", "model", "normalizer",
                "config", "dataset", "split", "alignment", "optimizer", "scheduler", "rng", "epoch",
                "global_step", "versions"}
    if not required <= payload.keys() or payload["version"] != 1 or payload["model_type"] != "diamond_cartesian":
        raise ValueError("Unsupported or incomplete DIAMOND checkpoint")
    source = manifest()
    if payload["manifest"] != source or payload["upstream_sha"] != source["commit"]:
        raise ValueError("DIAMOND source/patch manifest mismatch")
    spec = payload["spec"]
    if spec != model_spec(spec["image"]["views"]) or payload["alignment"] != ALIGNMENT:
        raise ValueError("Unsupported architecture, image, EDM, sampler or timing contract")
    validate_normalizer(payload["normalizer"])
    split = payload["split"]
    cfg = payload["config"]
    settings = {"dataset", "stats", "views", "action_alignment", "batch_size", "seed", "device",
                "steps", "lr", "eps", "weight_decay", "warmup_steps", "grad_clip", "val_fraction",
                "validate_every", "val_batches", "save_every", "train_windows", "output", "resume"}
    if (not settings <= cfg.keys() or cfg["views"] != spec["image"]["views"]
            or cfg["action_alignment"] != ALIGNMENT["action"] or split["seed"] != cfg["seed"]
            or not {"python", "torch", "cuda", "numpy"} <= payload["versions"].keys()):
        raise ValueError("Invalid training configuration or library metadata")
    ds = payload["dataset"]
    if not {"id", "fingerprint", "schema", "fingerprint_method", "duplicate_action_episodes"} <= ds.keys():
        raise ValueError("Missing dataset provenance")
    if (not split["train"] or not split["val"] or set(split["train"]) & set(split["val"])
            or any(len(split[k]) != len(set(split[k])) for k in ("train", "val"))
            or len(ds["fingerprint"]) != 64 or payload["global_step"] < 0 or payload["epoch"] < 0):
        raise ValueError("Invalid dataset, split or training progress")
    if (not {"state", "param_groups"} <= payload["optimizer"].keys()
            or "last_epoch" not in payload["scheduler"] or
            not {"python", "numpy", "torch", "cuda", "windows"} <= payload["rng"].keys()):
        raise ValueError("Missing training state")


def save_checkpoint(path, model, optimizer, scheduler, generator, *, config, dataset, split,
                    normalizer, global_step, epoch):
    path = Path(path)
    source = manifest()
    payload = dict(version=1, model_type="diamond_cartesian", upstream_sha=source["commit"],
                   manifest=source, spec=model.spec, model=model.state_dict(), normalizer=normalizer,
                   config=config, dataset=dataset, split=split, alignment=ALIGNMENT,
                   optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                   rng=capture_rng(generator, model.denoiser.device), global_step=global_step, epoch=epoch,
                   versions=dict(python=platform.python_version(), torch=str(torch.__version__),
                                 cuda=torch.version.cuda, numpy=np.__version__))
    validate(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".diamond-", suffix=".pt") as tmp:
        torch.save(payload, tmp)
        tmp.flush()
        os.fsync(tmp.fileno())
        os.link(tmp.name, path)  # Atomic publish, fail if a checkpoint already exists.


def read_checkpoint(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    validate(payload)
    return payload


def load_for_inference(path, device="cpu", *, expected_normalizer=None):
    payload = read_checkpoint(path)
    if expected_normalizer is not None and expected_normalizer != payload["normalizer"]:
        raise ValueError("WM action normalization mismatch")
    with torch.random.fork_rng(devices=[]):
        model = DiamondWorldModel(payload["spec"]["image"]["views"])
    model.load_state_dict(payload["model"], strict=True)
    metadata = {k: v for k, v in payload.items() if k not in {"model", "optimizer", "scheduler", "rng"}}
    return model.to(device).eval(), metadata

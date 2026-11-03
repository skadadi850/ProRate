"""Loader of ProRate checkpoints, released or freshly trained."""
import dataclasses

import torch

from prorate.model import ProtRig, ProtRigConfig


def load_protrig(ckpt_path, device="cpu"):
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = blob["cfg"]
    cfg = ProtRigConfig(**cfg) if isinstance(cfg, dict) else dataclasses.replace(cfg)
    model = ProtRig(cfg).to(device)
    model.load_state_dict(blob["model"])
    model.codebook.initialized = True
    return model.eval()

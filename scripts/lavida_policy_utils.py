"""LaViDa/OpenPI policy helpers only. No libero, no robosuite, no patches.

Use from eval_lavida_libero or policy_server_openpi:
  from scripts.lavida_policy_utils import build_policy, load_norm_stats, IMAGE_SIZE, MAX_TOKEN_LEN, PadTokenizedPrompt
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

OPENPI_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(OPENPI_ROOT / "src"))

import openpi.models.model as _model
import openpi.policies.libero_policy as libero_policy
import openpi.policies.policy as _policy
import openpi.transforms as transforms
from openpi.shared import normalize as _normalize
from openpi.training import config as _config

IMAGE_SIZE = 384
MAX_TOKEN_LEN = 48


def ensure_uint8_hwc(img):
    """Ensure image is uint8 HWC (no libero/robosuite deps)."""
    img = np.asarray(img)
    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    if img.ndim == 3 and img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    return img


class PadTokenizedPrompt:
    def __init__(self, max_len: int = MAX_TOKEN_LEN):
        self.max_len = max_len

    def __call__(self, data):
        tokens = np.asarray(data["tokenized_prompt"])
        mask = np.asarray(data["tokenized_prompt_mask"])
        if tokens.ndim == 1:
            tokens, mask = tokens[None, ...], mask[None, ...]
        seq_len = tokens.shape[-1]
        if seq_len < self.max_len:
            tokens = np.concatenate(
                [tokens, np.zeros((*tokens.shape[:-1], self.max_len - seq_len), dtype=tokens.dtype)],
                axis=-1,
            )
            mask = np.concatenate(
                [mask, np.zeros((*mask.shape[:-1], self.max_len - seq_len), dtype=bool)], axis=-1
            )
        else:
            tokens, mask = tokens[..., : self.max_len], mask[..., : self.max_len]
        data["tokenized_prompt"], data["tokenized_prompt_mask"] = tokens, mask
        return data


def load_norm_stats(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    stats = data["norm_stats"]
    return {
        k: _normalize.NormStats(
            mean=np.array(v["mean"], dtype=np.float32),
            std=np.array(v["std"], dtype=np.float32),
        )
        for k, v in stats.items()
    }


def build_policy(checkpoint_dir: str, norm_stats_path: str, device: str) -> _policy.Policy:
    checkpoint_dir = pathlib.Path(checkpoint_dir)
    weight_path = str(checkpoint_dir / "model.safetensors")
    train_config = _config.get_config("train_lavida_libero")
    model_config = train_config.model
    norm_stats = load_norm_stats(norm_stats_path)
    model = model_config.load_pytorch(train_config, weight_path)
    model = model.to(device).eval()
    tokenizer = __import__(
        "openpi.models.tokenizer", fromlist=["LaViDaTokenizer"]
    ).LaViDaTokenizer(MAX_TOKEN_LEN)
    return _policy.Policy(
        model,
        transforms=[
            libero_policy.LiberoInputs(model_type=_model.ModelType.PI0),
            transforms.InjectDefaultPrompt(None),
            transforms.Normalize(norm_stats, use_quantiles=False),
            transforms.ResizeImages(IMAGE_SIZE, IMAGE_SIZE),
            transforms.TokenizePrompt(tokenizer, discrete_state_input=False),
            PadTokenizedPrompt(MAX_TOKEN_LEN),
            transforms.PadStatesAndActions(model_config.action_dim),
        ],
        output_transforms=[
            transforms.Unnormalize(norm_stats, use_quantiles=False),
            libero_policy.LiberoOutputs(),
        ],
        is_pytorch=True,
        pytorch_device=device,
    )

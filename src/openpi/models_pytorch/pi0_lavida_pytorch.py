"""PI0-style model with LaViDa as the VLM backbone.

Single backbone: VLM features -> projector -> action head -> MSE loss.
No flow matching; same observation/actions interface as PI0 for the trainer.
"""

import logging
import time
import torch
from torch import nn
import torch.nn.functional as F

from openpi.models_pytorch.lavida_pytorch import LavidaPytorch

# Default action-expert hidden size (projected VLM dim).
ACTION_EXPERT_DIM = 256


def _to_device(x, device):
    if hasattr(x, "to"):
        return x.to(device)
    return torch.as_tensor(x, device=device)


class PI0LavidaPytorch(nn.Module):
    """PI0-style model: LaViDa VLM -> projector -> action head -> loss."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        lavida_path = getattr(config, "lavida_model_path", "/data/models/biyuz/hf_home/models/lavida-llada-v1.0-instruct")
        dtype = torch.bfloat16 if getattr(config, "dtype", "bfloat16") == "bfloat16" else torch.float32

        self.vlm = LavidaPytorch(lavida_path, torch_dtype=dtype)
        hidden_size = self.vlm.hidden_size
        action_dim = config.action_dim
        action_horizon = config.action_horizon

        self.projector = nn.Linear(hidden_size, ACTION_EXPERT_DIM, dtype=dtype)
        self.action_expert = nn.Linear(ACTION_EXPERT_DIM, action_horizon * action_dim, dtype=dtype)

        self.action_dim = action_dim
        self.action_horizon = action_horizon

        # Freeze VLM: only train projector + action_expert (action head).
        for p in self.vlm.parameters():
            p.requires_grad = False

    def forward(self, observation, actions):
        """Compute action prediction loss from observation and target actions.

        observation: OpenPI Observation (has .images, .tokenized_prompt, etc.)
        actions: (B, action_horizon, action_dim)
        Returns: loss tensor, shape (B,) or (B, action_horizon) for reduction='none'.
        """
        device = next(self.parameters()).device
        # Get first image (e.g. base_0_rgb) and input_ids from observation
        images_dict = observation.images if hasattr(observation, "images") else observation["image"]
        if isinstance(images_dict, dict):
            pixel_values = list(images_dict.values())[0]
        else:
            pixel_values = images_dict
        pixel_values = _to_device(pixel_values, device)
        if pixel_values.dim() == 3:
            pixel_values = pixel_values.unsqueeze(0)

        tokenized_prompt = observation.tokenized_prompt if hasattr(observation, "tokenized_prompt") else observation["tokenized_prompt"]
        input_ids = _to_device(tokenized_prompt, device).long()
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        vlm_features = self.vlm(input_ids=input_ids, images=pixel_values)
        # Last token as sentence representation
        vlm_features = vlm_features[:, -1, :]
        # Keep same dtype as projector/action_expert (bfloat16) to avoid mat1/mat2 dtype mismatch.
        projected = self.projector(vlm_features)
        action_logits = self.action_expert(projected)
        B = action_logits.size(0)
        action_logits = action_logits.view(B, self.action_horizon, self.action_dim)

        actions = _to_device(actions, device).float()
        # Align batch: LaViDa internal forward can change batch size; use min to avoid shape mismatch.
        B_use = min(action_logits.size(0), actions.size(0))
        pred = action_logits[:B_use].float()
        tgt = actions[:B_use]
        loss = F.mse_loss(pred, tgt, reduction="none")
        return loss

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, **kwargs):
        """Inference: predict action chunk (B, action_horizon, action_dim). No denoising."""
        # Reuse same path as forward (images + tokenized_prompt -> vlm -> projector -> action_expert).
        images_dict = observation.images if hasattr(observation, "images") else observation["image"]
        if isinstance(images_dict, dict):
            pixel_values = list(images_dict.values())[0]
        else:
            pixel_values = images_dict
        pixel_values = _to_device(pixel_values, device)
        if pixel_values.dim() == 3:
            pixel_values = pixel_values.unsqueeze(0)

        tokenized_prompt = observation.tokenized_prompt if hasattr(observation, "tokenized_prompt") else observation["tokenized_prompt"]
        input_ids = _to_device(tokenized_prompt, device).long()
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        t0 = time.perf_counter()
        vlm_features = self.vlm(input_ids=input_ids, images=pixel_values)
        vlm_features = vlm_features[:, -1, :]
        projected = self.projector(vlm_features)
        action_logits = self.action_expert(projected)
        latency_ms = (time.perf_counter() - t0) * 1000
        # Performance monitoring: keep latency print. If latency stays <50ms and actions don't change, vision likely not active.
        logging.getLogger(__name__).info(
            "[pi0_lavida] sample_actions latency=%.1f ms (if consistently <50ms with constant actions, vision not used)",
            latency_ms,
        )
        B = action_logits.size(0)
        actions = action_logits.view(B, self.action_horizon, self.action_dim).float()
        return actions

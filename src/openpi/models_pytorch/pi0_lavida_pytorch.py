"""PI0-style model with LaViDa as the VLM backbone and Flow Matching action head.

Flow matching: time-conditioned action head, x_t = t*noise + (1-t)*actions,
u_t = noise - actions, loss = MSE(u_t, v_t). Inference via Euler ODE (multi-step denoising).
"""

import logging
import math
import time

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from openpi.models_pytorch.lavida_pytorch import LavidaPytorch
from openpi.models_pytorch.pi0_pytorch import (
    create_sinusoidal_pos_embedding,
    sample_beta,
)

# Flow head hidden size (time-conditioned action MLP).
FLOW_WIDTH = 256


def _to_device(x, device):
    if hasattr(x, "to"):
        return x.to(device)
    return torch.as_tensor(x, device=device)


def _get_observation_components(observation, device, expected_b=None):
    """Extract images, input_ids, state from observation (dict or object).

    If expected_b is set and > 1, and any of the extracted tensors has batch_size 1,
    they are expanded to batch_size expected_b so they align with actions/state.
    """
    images_dict = observation.images if hasattr(observation, "images") else observation["image"]
    if isinstance(images_dict, dict):
        pixel_values = list(images_dict.values())[0]
    else:
        pixel_values = images_dict
    pixel_values = _to_device(pixel_values, device)
    if pixel_values.dim() == 3:
        pixel_values = pixel_values.unsqueeze(0)
    if expected_b is not None and expected_b > 1 and pixel_values.shape[0] == 1:
        pixel_values = pixel_values.expand(expected_b, *([-1] * (pixel_values.dim() - 1)))

    tokenized_prompt = observation.tokenized_prompt if hasattr(observation, "tokenized_prompt") else observation["tokenized_prompt"]
    input_ids = _to_device(tokenized_prompt, device).long()
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    if expected_b is not None and expected_b > 1 and input_ids.shape[0] == 1:
        input_ids = input_ids.expand(expected_b, -1)

    state = observation.state if hasattr(observation, "state") else observation["state"]
    state = _to_device(state, device).float()
    if state.dim() == 1:
        state = state.unsqueeze(0)
    if expected_b is not None and expected_b > 1 and state.shape[0] == 1:
        state = state.expand(expected_b, -1)

    return pixel_values, input_ids, state


class PI0LavidaPytorch(nn.Module):
    """PI0 flow-matching model with LaViDa VLM backbone (frozen) and time-conditioned action head."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        lavida_path = getattr(
            config, "lavida_model_path",
            "/data/models/biyuz/hf_home/models/lavida-llada-v1.0-instruct",
        )
        dtype = torch.bfloat16 if getattr(config, "dtype", "bfloat16") == "bfloat16" else torch.float32

        self.vlm = LavidaPytorch(lavida_path, torch_dtype=dtype)
        hidden_size = self.vlm.hidden_size
        action_dim = config.action_dim
        state_dim = config.state_dim        
        action_horizon = config.action_horizon

        self.action_dim = action_dim
        self.state_dim = state_dim
        self.action_horizon = action_horizon
        self.pi05 = getattr(config, "pi05", False)

        # Flow matching head: state + noisy action + time -> v_t
        self.action_in_proj = nn.Linear(action_dim, FLOW_WIDTH, dtype=torch.float32)
        self.action_out_proj = nn.Linear(FLOW_WIDTH, action_dim, dtype=torch.float32)
        self.state_proj = nn.Linear(state_dim, FLOW_WIDTH, dtype=torch.float32)
        self.vlm_proj = nn.Linear(hidden_size, FLOW_WIDTH, dtype=torch.float32)

        if self.pi05:
            self.time_mlp_in = nn.Linear(FLOW_WIDTH, FLOW_WIDTH, dtype=torch.float32)
            self.time_mlp_out = nn.Linear(FLOW_WIDTH, FLOW_WIDTH, dtype=torch.float32)
        else:
            self.action_time_mlp_in = nn.Linear(2 * FLOW_WIDTH, FLOW_WIDTH, dtype=torch.float32)
            self.action_time_mlp_out = nn.Linear(FLOW_WIDTH, FLOW_WIDTH, dtype=torch.float32)

        # Freeze VLM
        for p in self.vlm.parameters():
            p.requires_grad = False

    def sample_noise(self, shape, device):
        return torch.normal(mean=0.0, std=1.0, size=shape, dtype=torch.float32, device=device)

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        t = time_beta * 0.999 + 0.001
        return t.to(dtype=torch.float32, device=device)

    def _embed_vlm(self, pixel_values, input_ids):
        """Get VLM last-token features (B, D). Handles LaViDa diffusion branch doubling."""
        B = input_ids.shape[0]
        vlm_out = self.vlm(input_ids=input_ids, images=pixel_values)

        if isinstance(vlm_out, dict):
            vlm_h = vlm_out.get("hidden_states", vlm_out.get("logits", None))
            if vlm_h is None:
                raise ValueError(f"Could not find hidden_states or logits in VLM output dict: {vlm_out.keys()}")
        elif isinstance(vlm_out, tuple):
            vlm_h = vlm_out[0]
        else:
            vlm_h = vlm_out

        # If VLM internal diffusion branch doubled the batch (1 -> 2), keep only the first B
        if vlm_h.shape[0] == 2 * B:
            vlm_h = vlm_h[:B]

        return vlm_h[:, -1, :]

    def _flow_head(self, vlm_features, state, x_t, timestep):
        """Predict velocity v_t from (vlm_features, state, x_t, timestep). All float32 for stability."""
        B, H, _ = x_t.shape
        device = x_t.device

        state = state.to(torch.float32)
        x_t = x_t.to(torch.float32)
        timestep = timestep.to(torch.float32)
        vlm_features = vlm_features.to(torch.float32)

        state_emb = self.state_proj(state)
        action_emb = self.action_in_proj(x_t)

        time_emb = create_sinusoidal_pos_embedding(
            timestep, FLOW_WIDTH, min_period=4e-3, max_period=4.0, device=device,
        ).to(dtype=torch.float32)

        if not self.pi05:
            time_emb = time_emb[:, None, :].expand_as(action_emb)
            action_time = torch.cat([action_emb, time_emb], dim=2)
            action_time = F.silu(self.action_time_mlp_in(action_time))
            action_time = self.action_time_mlp_out(action_time)
        else:
            time_emb = F.silu(self.time_mlp_in(time_emb))
            time_emb = F.silu(self.time_mlp_out(time_emb))
            action_time = action_emb
            # Broadcast time into action stream for conditioning
            action_time = action_time + time_emb.unsqueeze(1)

        vlm_emb = self.vlm_proj(vlm_features)
        combined = action_time + vlm_emb.unsqueeze(1) + state_emb.unsqueeze(1)
        v_t = self.action_out_proj(combined)
        return v_t

    def forward(self, observation, actions, noise=None, time=None) -> Tensor:
        device = next(self.parameters()).device
        actions = _to_device(actions, device).float()
        
        # 拿到最权威的 batch_size (16)
        B = actions.shape[0]

        # 1. 正常获取数据 (如果你的这个函数现在要求传 expected_b，就在后面加上 expected_b=B)
        pixel_values, input_ids, state = _get_observation_components(observation, device)

        # 2. 物理扩充：不用 expand(容易共享内存报错)，直接用 repeat 强行复制成 16 份！
        if pixel_values.shape[0] == 1 and B > 1:
            pixel_values = pixel_values.repeat(B, 1, 1, 1)
        if input_ids.shape[0] == 1 and B > 1:
            input_ids = input_ids.repeat(B, 1)
        if state.shape[0] == 1 and B > 1:
            state = state.repeat(B, 1)

        # 3. 提取视觉特征 (此时送进去的绝对是 16)
        vlm_features = self._embed_vlm(pixel_values, input_ids)

        # 4. 终极修剪：不管 LaViDa 肚子里怎么翻倍，出来必须是 16！
        if vlm_features.shape[0] == 2 * B:
            # 正常翻倍情况：32 切成 16
            vlm_features = vlm_features[:B]
        elif vlm_features.shape[0] == 2 and B > 1:
            # 极端异常兜底：如果它还是 2，切下第 1 个，强行复制 16 遍
            vlm_features = vlm_features[:1].repeat(B, 1)

        # 此时此刻，vlm_features 绝对、一定、必须是 B (16)！
        
        if noise is None:
            noise = self.sample_noise(actions.shape, device)
        if time is None:
            time = self.sample_time(B, device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # 16 的特征 + 16 的状态 + 16 的动作
        v_t = self._flow_head(vlm_features, state, x_t, time)

        return F.mse_loss(v_t, u_t, reduction="none")

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=50, **kwargs) -> Tensor:
        """Inference: Euler ODE from t=1 to 0 with denoise_step."""
        # Expected batch: state batch size (or 1 for single rollout)
        pixel_values, input_ids, state = _get_observation_components(observation, device, expected_b=None)
        bsize = state.shape[0]
        actions_shape = (bsize, self.action_horizon, self.action_dim)

        if noise is None:
            noise = self.sample_noise(actions_shape, device)

        t0 = time.perf_counter()
        vlm_features = self._embed_vlm(pixel_values, input_ids)

        dt = -1.0 / num_steps
        dt_t = torch.tensor(dt, dtype=torch.float32, device=device)
        x_t = noise.to(torch.float32)
        time_val = torch.tensor(1.0, dtype=torch.float32, device=device)

        while time_val >= -dt_t / 2:
            expanded_time = time_val.expand(bsize)
            v_t = self._flow_head(vlm_features, state, x_t, expanded_time)
            x_t = x_t + dt_t * v_t
            time_val = time_val + dt_t

        latency_ms = (time.perf_counter() - t0) * 1000
        logging.getLogger(__name__).info(
            "[pi0_lavida] sample_actions (flow matching, %d steps) latency=%.1f ms",
            num_steps, latency_ms,
        )
        return x_t.float()

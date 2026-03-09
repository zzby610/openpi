# """LaViDa (diffusion VLM) backbone for OpenPI.

# Uses a vision-capable class (LlavaLladaForMaskedDiffusion) so forward accepts images/pixel_values.
# If that class is not available, tries adding model path to sys.path and re-importing.
# Refuses to fall back to text-only AutoModelForCausalLM so the policy does not run blind.
# """

# import inspect
# import logging
# import os
# import sys
# from typing import Any

# import torch
# from torch import nn

# logger = logging.getLogger(__name__)

# LlavaLladaForMaskedDiffusion = None


# def _import_llava_llada(model_path: str):
#     """Import LlavaLladaForMaskedDiffusion. Add model directory to sys.path[0] before first import to fix relative imports."""
#     global LlavaLladaForMaskedDiffusion
#     # Add model directory (and parents) at sys.path[0] so "llava" package and .llada relative imports resolve
#     candidates = []
#     if model_path:
#         p = os.path.dirname(model_path) if os.path.isfile(model_path) else model_path
#         if os.path.isdir(p):
#             candidates.append(p)
#             candidates.append(os.path.dirname(p))
#             candidates.append(os.path.dirname(os.path.dirname(p)))
#     for p in candidates:
#         if p and os.path.isdir(p) and p not in sys.path:
#             sys.path.insert(0, p)
#             logger.info("[lavida_pytorch] Added to sys.path[0] for llava/llada import: %s", p)
#             break
#     try:
#         from llava_llada import LlavaLladaForMaskedDiffusion as _Cls
#         LlavaLladaForMaskedDiffusion = _Cls
#         return _Cls
#     except Exception as e:
#         logger.debug("[lavida_pytorch] Import after path add failed: %s", e)
#     for p in candidates:
#         if p in sys.path:
#             sys.path.remove(p)
#     return None


# def _load_lavida_model(model_path: str, torch_dtype: torch.dtype):
#     """Load the LaViDa model as LlavaLladaForMaskedDiffusion only. No AutoModelForCausalLM fallback (would be blind)."""
#     cls = LlavaLladaForMaskedDiffusion if LlavaLladaForMaskedDiffusion is not None else _import_llava_llada(model_path)
#     if cls is None:
#         raise RuntimeError(
#             "LlavaLladaForMaskedDiffusion could not be imported. Add the LLaVA/LaViDa repo to PYTHONPATH: "
#             "export PYTHONPATH=/path/to/llava:$PYTHONPATH (or set sys.path[0] to the repo root). "
#             "Do not use AutoModelForCausalLM — the policy would be blind."
#         )
#     try:
#         model = cls.from_pretrained(
#             model_path,
#             torch_dtype=torch_dtype,
#             trust_remote_code=True,
#             local_files_only=True,
#         )
#         logger.info(
#             "[lavida_pytorch] Loaded with %s (vision + diffusion)",
#             cls.__name__,
#         )
#         return model
#     except Exception as e:
#         logger.exception("[lavida_pytorch] %s.from_pretrained failed: %s", cls.__name__, e)
#         raise RuntimeError(
#             "LaViDa vision model failed to load. Ensure LLaVA/LaViDa repo is on PYTHONPATH and "
#             "checkpoint is for LlavaLladaForMaskedDiffusion."
#         ) from e


# def _model_accepts_vision(model) -> bool:
#     """Return True if model.forward() accepts images or pixel_values (vision-capable)."""
#     try:
#         sig = inspect.signature(model.forward)
#         params = set(sig.parameters.keys())
#         return "images" in params or "pixel_values" in params
#     except (TypeError, ValueError):
#         return False


# class LavidaPytorch(nn.Module):
#     """LaViDa VLM backbone: causal LM + vision, bfloat16, returns last hidden state."""

#     def __init__(self, model_path: str, torch_dtype: torch.dtype = torch.bfloat16):
#         super().__init__()
#         self.model = _load_lavida_model(model_path, torch_dtype)
#         self.hidden_size = getattr(self.model.config, "d_model", self.model.config.hidden_size)
#         try:
#             sig = inspect.signature(self.model.forward)
#             forward_param_names = list(sig.parameters.keys())
#             logger.info(
#                 "[lavida_pytorch] model.forward parameter names: %s",
#                 forward_param_names,
#             )
#             if not _model_accepts_vision(self.model):
#                 logger.warning(
#                     "[lavida_pytorch] No vision-related parameter in forward (expected 'images' or 'pixel_values'). "
#                     "Loaded class may be wrong.",
#                 )
#         except (TypeError, ValueError) as e:
#             logger.warning("[lavida_pytorch] Could not inspect model.forward signature: %s", e)

#     def _adapt_kwargs_to_forward_signature(self, kwargs: dict[str, Any]) -> dict[str, Any]:
#         """Filter and rename kwargs so they match the underlying model.forward() signature.
#         E.g. if the model expects pixel_values but we have images, rename; drop unsupported keys.
#         Clamp input_ids to vocab_size to avoid device-side index out of bounds.
#         """
#         had_visual = "images" in kwargs or "pixel_values" in kwargs

#         try:
#             sig = inspect.signature(self.model.forward)
#             forward_params = set(sig.parameters.keys())
#         except (TypeError, ValueError):
#             return kwargs

#         # Safety: clamp input_ids to valid vocab range to avoid CUDA index asserts
#         if "input_ids" in kwargs:
#             vocab_size = getattr(self.model.config, "vocab_size", getattr(self.model.config, "padded_vocab_size", None))
#             if vocab_size is not None:
#                 ids = kwargs["input_ids"]
#                 if ids.dtype in (torch.long, torch.int, torch.int32, torch.int64):
#                     invalid = (ids < 0) | (ids >= vocab_size)
#                     if invalid.any():
#                         kwargs["input_ids"] = ids.clamp(0, vocab_size - 1)
#                         logger.debug(
#                             "[lavida_pytorch] Clamped %s input_ids to [0, vocab_size-1=%s]",
#                             invalid.sum().item(),
#                             vocab_size - 1,
#                         )

#         model_dtype = next(self.model.parameters()).dtype

#         def _to_chw(t):
#             if t.ndim == 4 and t.shape[-1] == 3:
#                 return t.permute(0, 3, 1, 2)
#             return t

#         # Hard rename: map to the key the model expects. Never drop visual.
#         if "images" in kwargs and "images" not in forward_params and "pixel_values" in forward_params:
#             img = kwargs.pop("images", None)
#             if img is not None:
#                 kwargs["pixel_values"] = _to_chw(img.to(model_dtype))
#         if "pixel_values" in kwargs and "pixel_values" not in forward_params and "images" in forward_params:
#             pv = kwargs.pop("pixel_values", None)
#             if pv is not None:
#                 pv = pv.to(model_dtype)
#                 kwargs["images"] = pv.permute(0, 2, 3, 1) if pv.ndim == 4 and pv.shape[1] == 3 else pv
#         if "image_sizes" in kwargs and "image_sizes" not in forward_params:
#             kwargs.pop("image_sizes", None)

#         filtered = {k: v for k, v in kwargs.items() if k in forward_params}
#         dropped = set(kwargs.keys()) - set(filtered.keys())
#         has_visual = "images" in filtered or "pixel_values" in filtered

#         if had_visual and not has_visual:
#             logger.error(
#                 "[lavida_pytorch] VISION DROPPED: caller passed images/pixel_values but model.forward() "
#                 "does not accept 'images' or 'pixel_values'. Dropped keys: %s. Model is effectively BLIND.",
#                 dropped,
#             )
#         if dropped and not dropped.issubset({"image_sizes"}):
#             logger.debug(
#                 "[lavida_pytorch] Dropped kwargs not in model.forward signature: %s", dropped
#             )
#         return filtered

#     def forward(
#         self,
#         input_ids: torch.Tensor,
#         images: torch.Tensor | None = None,
#         image_sizes: list[list[int]] | None = None,
#     ) -> torch.Tensor:
#         """Run forward; return last hidden state. Uses current-frame-only (slicing) strategy.

#         If input_ids is 3D (B, T, L) or images 5D (B, T, C, H, W), only the last frame is passed
#         so the model receives standard 2D/4D and runs in single-frame mode. Output is (B, 1*L, D).
#         """
#         # --- Slice to current frame: 3D input_ids (B, T, L) -> (B, L) ---
#         if input_ids.dim() == 3:
#             input_ids = input_ids[:, -1, :]
#         batch_size = input_ids.size(0)
#         seq_len = input_ids.size(1)

#         if image_sizes is None:
#             image_sizes = [[384, 384]] * batch_size

#         # --- Slice to current frame: 5D images (B, T, C, H, W) -> (B, C, H, W); 4D with T>1 take last ---
#         if images is not None:
#             model_dtype = next(self.model.parameters()).dtype
#             if images.dim() == 5:
#                 images = images[:, -1, ...]
#             elif images.dim() == 4 and images.size(0) > batch_size:
#                 # e.g. (B*T, C, H, W) with B*T > B: take last B frames as "current"
#                 images = images[-batch_size:]
#             images = images.to(model_dtype)

#         dummy_labels = torch.full_like(input_ids, -100, device=input_ids.device, dtype=torch.long)
#         attention_mask = torch.ones(batch_size, seq_len, device=input_ids.device, dtype=torch.long)

#         kwargs: dict[str, Any] = {
#             "input_ids": input_ids,
#             "attention_mask": attention_mask,
#             "labels": dummy_labels,
#             "output_hidden_states": True,
#             "return_dict": True,
#         }
#         if images is not None:
#             kwargs["images"] = images
#         if image_sizes is not None:
#             kwargs["image_sizes"] = image_sizes

#         kwargs = self._adapt_kwargs_to_forward_signature(kwargs)

#         pv = kwargs.get("pixel_values")
#         im = kwargs.get("images")
#         visual_tensor = pv if pv is not None else im
#         if visual_tensor is None:
#             logger.warning(
#                 "[lavida_pytorch] forward: pixel_values and images are both None — model is BLIND (no vision input)."
#             )
#         else:
#             logger.debug(
#                 "[lavida_pytorch] forward: visual input shape %s (dtype=%s)",
#                 tuple(visual_tensor.shape),
#                 visual_tensor.dtype,
#             )

#         # Prefer model.generate() to enable multi-step diffusion denoising (latency >>50ms when vision+diffusion active)
#         hidden = self._forward_or_generate(kwargs)
#         return hidden.reshape(hidden.size(0), 1 * hidden.size(1), -1)

#     def _forward_or_generate(self, kwargs: dict[str, Any]) -> torch.Tensor:
#         """Call model.generate() when available (diffusion denoising); else model(**kwargs). Returns last-layer hidden (B, L, D)."""
#         gen = getattr(self.model, "generate", None)
#         if gen is not None and callable(gen):
#             try:
#                 # Pass through kwargs that generate() accepts (e.g. input_ids, images/pixel_values, image_sizes, output_hidden_states)
#                 gen_kwargs = {k: v for k, v in kwargs.items() if k != "labels"}
#                 if "output_hidden_states" not in gen_kwargs:
#                     gen_kwargs["output_hidden_states"] = True
#                 out = gen(**gen_kwargs)
#                 if hasattr(out, "hidden_states") and out.hidden_states is not None:
#                     return out.hidden_states[-1]
#                 if hasattr(out, "encoder_hidden_states") and out.encoder_hidden_states is not None:
#                     return out.encoder_hidden_states[-1]
#                 if isinstance(out, (list, tuple)) and len(out) > 0:
#                     last = out[-1]
#                     if hasattr(last, "hidden_states"):
#                         return last.hidden_states[-1]
#                     if isinstance(last, torch.Tensor) and last.dim() == 3:
#                         return last
#             except Exception as e:
#                 logger.warning(
#                     "[lavida_pytorch] model.generate() failed (%s); falling back to forward()",
#                     e,
#                 )
#         try:
#             outputs = self.model(**kwargs)
#         except Exception as e:
#             logger.warning("[lavida_pytorch] model.forward failed: %s", e)
#             raise
#         return outputs.hidden_states[-1]





"""LaViDa VLM backbone for OpenPI.

Code path: model class is imported from the LaViDa source repo (PYTHONPATH).
Weight path: only config.json and safetensors are read from the weight directory;
no .py files are executed from there (avoids broken scripts in the cache dir).
"""

import logging
from pathlib import Path

import torch
from torch import nn
from safetensors.torch import load_file

logger = logging.getLogger(__name__)

# Default weight path (read-only: config + safetensors).
DEFAULT_WEIGHT_PATH = "/data/models/biyuz/hf_home/models/lavida-llada-v1.0-instruct"


def _import_lavida_model_class():
    """Import LlavaLladaForMaskedDiffusion and LlavaLladaConfig from the LaViDa source repo.
    Requires PYTHONPATH to include the LaViDa repo root (e.g. export PYTHONPATH=.../LaViDa:$PYTHONPATH).
    """
    try:
        from llava.model.language_model.llava_llada import (
            LlavaLladaForMaskedDiffusion,
            LlavaLladaConfig,
        )
        return LlavaLladaForMaskedDiffusion, LlavaLladaConfig
    except ImportError as e:
        logger.error(
            "[lavida_pytorch] Failed to import from LaViDa repo. "
            "Set PYTHONPATH to include the LaViDa source root: "
            "export PYTHONPATH=/export/ra/zoubiyu/repo/LaViDa:$PYTHONPATH. Error: %s",
            e,
        )
        raise


def _load_lavida_model(weight_path: str, torch_dtype: torch.dtype):
    """Load LaViDa model using code from the LaViDa repo and weights from weight_path only.

    - Code path: LlavaLladaForMaskedDiffusion is imported from llava.model... (LaViDa repo).
    - Weight path: we only read config.json and *.safetensors from weight_path; no Python
      files are executed from that directory (so broken .py in the cache dir are never run).
    - We do NOT use trust_remote_code=True with the weight path to avoid loading any script from it.
    """
    weight_path = weight_path or DEFAULT_WEIGHT_PATH
    weight_dir = Path(weight_path)
    if not weight_dir.is_dir():
        raise FileNotFoundError(f"[lavida_pytorch] Weight directory not found: {weight_path}")

    ModelClass, ConfigClass = _import_lavida_model_class()

    # Load config from weight dir (reads config.json only; no .py executed).
    config = ConfigClass.from_pretrained(str(weight_dir))
    # Build model structure from the class we imported (LaViDa repo code).
    model = ModelClass(config)
    model = model.to(torch_dtype)

    # Load state dict from safetensors only (no code from weight dir).
    safetensors_files = sorted(weight_dir.glob("*.safetensors"))
    if not safetensors_files:
        raise FileNotFoundError(
            f"[lavida_pytorch] No .safetensors files in {weight_path}. "
            "We only load weights from safetensors, not from the directory's Python scripts."
        )
    state_dict = {}
    for f in safetensors_files:
        state_dict.update(load_file(str(f)))
    load_result = model.load_state_dict(state_dict, strict=False)
    if load_result.missing_keys:
        logger.warning("[lavida_pytorch] load_state_dict missing_keys: %s", load_result.missing_keys[:20])
    if load_result.unexpected_keys:
        logger.warning("[lavida_pytorch] load_state_dict unexpected_keys: %s", load_result.unexpected_keys[:20])

    logger.info(
        "[lavida_pytorch] Loaded LaViDa from repo code + weights from %s (%d safetensors)",
        weight_path,
        len(safetensors_files),
    )
    return model

class LavidaPytorch(nn.Module):
    def __init__(self, model_path: str, torch_dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.model = _load_lavida_model(model_path, torch_dtype)
        self.hidden_size = getattr(self.model.config, "hidden_size", 4096)

    def forward(self, input_ids: torch.Tensor, images: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        # 修改本文件后需重启 policy server（policy_server_openpi.py）才能生效
        # 处理输入维度
        if input_ids.dim() == 3: input_ids = input_ids[:, -1, :]
        if images is not None and images.dim() == 5: images = images[:, -1, ...]

        model_dtype = next(self.model.parameters()).dtype
        device = input_ids.device
        # llava_llada.py 无条件对 attention_mask/labels 做原地赋值，两者都不能为 None（否则报 NoneType item assignment）
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device).clone()
        labels = torch.full_like(input_ids, -100, dtype=torch.long, device=device).clone()
        # 按 llava_llada.forward 参数顺序传参，避免被父类或 **kwargs 过滤掉
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            images=images.to(model_dtype) if images is not None else None,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = outputs.hidden_states[-1]
        return hidden.reshape(hidden.size(0), 1 * hidden.size(1), -1)
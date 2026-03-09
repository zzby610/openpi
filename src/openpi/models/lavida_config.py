# """Configuration for the LaViDa-based PI0 model variant.

# Fully independent from LaMDA; uses LaViDa (diffusion VLM) as backbone with 384x384 images.
# """

# import dataclasses

# import flax.nnx as nnx
# import jax
# import jax.numpy as jnp
# from typing_extensions import override

# from openpi.models import model as _model
# from openpi.shared import array_typing as at

# LAVIDA_IMAGE_RESOLUTION = (384, 384)


# @dataclasses.dataclass(frozen=True)
# class LaViDaConfig(_model.BaseModelConfig):
#     """Config for PI0 with LaViDa VLM backbone (model_name pi0_lavida)."""

#     model_name: str = "pi0_lavida"
#     dtype: str = "bfloat16"
#     lavida_model_path: str = "/data/models/biyuz/hf_home/models/lavida-llada-v1.0-instruct"

#     # Match libero_lerobot dataset: 7D actions (norm_stats has 7 dims), horizon 50.
#     action_dim: int = 7
#     action_horizon: int = 50
#     max_token_len: int = 48

#     @property
#     @override
#     def model_type(self) -> _model.ModelType:
#         return _model.ModelType.PI0

#     @override
#     def create(self, rng: at.KeyArrayLike):
#         raise NotImplementedError(
#             "LaViDaConfig.create() is not supported for JAX. Use PyTorch training."
#         )

#     def load_pytorch(self, train_config, weight_path: str):
#         import torch
#         from safetensors.torch import load_file
#         from openpi.models_pytorch.pi0_lavida_pytorch import PI0LavidaPytorch

#         model = PI0LavidaPytorch(config=self)
#         if not weight_path:
#             return model

#         print(f"📥 Loading checkpoint from {weight_path}")
#         state_dict = load_file(weight_path)
#         checkpoint_prefixes = {k.split(".")[0] for k in state_dict.keys()}
#         print(f"[lavida_config] Checkpoint key prefixes: {sorted(checkpoint_prefixes)} (expect 'vlm' or 'model' for VLM)")

#         def _is_critical_missing(key: str) -> bool:
#             k = key.lower()
#             return "visual_tower" in k or "mm_projector" in k or "action_expert" in k

#         def _remap_for_pi0(sd):
#             """Build state_dict for PI0LavidaPytorch: map model.xxx -> vlm.model.xxx so 30k-step weights load."""
#             out = {}
#             for k, v in sd.items():
#                 if k.startswith("model.") and not k.startswith("vlm."):
#                     out["vlm." + k] = v
#                 else:
#                     out[k] = v
#             return out

#         # First try: load as-is (checkpoint may already use vlm.*, projector.*, action_expert.*)
#         load_result = model.load_state_dict(state_dict, strict=False)
#         missing_keys = list(load_result.missing_keys)
#         unexpected_keys = list(load_result.unexpected_keys)

#         # If model expects vlm.* but checkpoint has model.*, remap and retry
#         needs_remap = any(m.startswith("vlm.") for m in missing_keys) and any(
#             k.startswith("model.") and not k.startswith("vlm.") for k in state_dict
#         )
#         if needs_remap:
#             remapped = _remap_for_pi0(state_dict)
#             print("[lavida_config] Remapping checkpoint: model.xxx -> vlm.model.xxx (30k-step weights)")
#             load_result = model.load_state_dict(remapped, strict=False)
#             missing_keys = list(load_result.missing_keys)
#             unexpected_keys = list(load_result.unexpected_keys)

#         # Detailed table of missing_keys (do not use strict=False quietly)
#         print("[lavida_config] ---- missing_keys (detailed) ----")
#         critical_missing = []
#         for i, k in enumerate(missing_keys, 1):
#             critical = _is_critical_missing(k)
#             if critical:
#                 critical_missing.append(k)
#             print(f"  [{i}] {k}  |  critical={critical}")
#         print(f"[lavida_config] ---- unexpected_keys: {len(unexpected_keys)} ----")
#         for i, k in enumerate(unexpected_keys[:50], 1):
#             print(f"  [{i}] {k}")
#         if len(unexpected_keys) > 50:
#             print(f"  ... and {len(unexpected_keys) - 50} more")

#         if critical_missing:
#             raise RuntimeError(
#                 "[lavida_config] CRITICAL: The following keys are missing (visual_tower/mm_projector/action_expert). "
#                 "30k-step weights did not load. missing_keys: %s. Fix checkpoint key names or remapping."
#                 % critical_missing
#             )

#         if missing_keys:
#             print("[lavida_config] WARNING: load_state_dict was non-strict. Missing keys (non-critical):", missing_keys)
#         return model

#     @override
#     def inputs_spec(self, *, batch_size: int = 1):
#         h, w = LAVIDA_IMAGE_RESOLUTION
#         image_spec = jax.ShapeDtypeStruct([batch_size, h, w, 3], jnp.float32)
#         image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

#         with at.disable_typechecking():
#             observation_spec = _model.Observation(
#                 images={
#                     "base_0_rgb": image_spec,
#                     "left_wrist_0_rgb": image_spec,
#                     "right_wrist_0_rgb": image_spec,
#                 },
#                 image_masks={
#                     "base_0_rgb": image_mask_spec,
#                     "left_wrist_0_rgb": image_mask_spec,
#                     "right_wrist_0_rgb": image_mask_spec,
#                 },
#                 state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
#                 tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
#                 tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
#             )
#         action_spec = jax.ShapeDtypeStruct(
#             [batch_size, self.action_horizon, self.action_dim], jnp.float32,
#         )
#         return observation_spec, action_spec

#     def get_freeze_filter(self) -> nnx.filterlib.Filter:
#         return nnx.Nothing




import dataclasses
import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override
from openpi.models import model as _model
from openpi.shared import array_typing as at

@dataclasses.dataclass(frozen=True)
class LaViDaConfig(_model.BaseModelConfig):
    model_name: str = "pi0_lavida"
    dtype: str = "bfloat16"
    lavida_model_path: str = "/data/models/biyuz/hf_home/models/lavida-llada-v1.0-instruct"
    action_dim: int = 7
    state_dim: int = 8
    action_horizon: int = 50
    max_token_len: int = 48

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.PI0

    def load_pytorch(self, train_config, weight_path: str):
        import torch
        from safetensors.torch import load_file
        from openpi.models_pytorch.pi0_lavida_pytorch import PI0LavidaPytorch

        model = PI0LavidaPytorch(config=self)
        if not weight_path: return model

        print(f"📥 正在加载 30k 步权重: {weight_path}")
        state_dict = load_file(weight_path)
        
        # 强制检查关键权重前缀
        # 如果模型里叫 vlm.model.xxx，但权重文件里叫 model.xxx，就进行重命名
        has_vlm_prefix = any(k.startswith("vlm.") for k in model.state_dict().keys())
        has_model_prefix_in_file = any(k.startswith("model.") for k in state_dict.keys())

        if has_vlm_prefix and has_model_prefix_in_file:
            print("[lavida_config] 发现前缀不匹配，正在自动对齐权重名 (model -> vlm.model)...")
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("model."):
                    new_state_dict["vlm." + k] = v
                else:
                    new_state_dict[k] = v
            state_dict = new_state_dict

        # 加载权重，开启严格模式检查
        res = model.load_state_dict(state_dict, strict=False)
        
        # 检查是否丢了核心部件
        missing_vision = [k for k in res.missing_keys if "visual" in k or "projector" in k]
        if missing_vision:
            print(f"⚠️ 警告: 视觉权重缺失，模型可能无法看图: {missing_vision[:3]}...")
        else:
            print("✅ 权重加载成功，视觉和动作模块已对齐！")
            
        return model

    @override
    def inputs_spec(self, *, batch_size: int = 1):
        image_spec = jax.ShapeDtypeStruct([batch_size, 384, 384, 3], jnp.float32)
        observation_spec = _model.Observation(
            images={"base_0_rgb": image_spec},
            state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
            tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
        )
        return observation_spec, jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

    def create(self, rng): raise NotImplementedError("Use PyTorch.")
    def get_freeze_filter(self): return nnx.Nothing
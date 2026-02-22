"""Configuration for the LaMDA-based PI0 model variant.

Mirrors ``pi0_config.Pi0Config`` but targets the LLaDA-8B backbone with
SigLIP vision encoder at 518×518 resolution.
"""

import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at

LAMDA_IMAGE_RESOLUTION = (518, 518)


@dataclasses.dataclass(frozen=True)
class LaMDAConfig(_model.BaseModelConfig):
    """Config for PI0 with LaMDA (LLaDA-8B) VLM backbone."""

    dtype: str = "bfloat16"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Path to the LaMDA / LLaDA-8B pretrained checkpoint.
    lamda_model_path: str = "/data/models/biyuz/hf_home/models/LLaDA-8B-Base"

    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    pi05: bool = False
    discrete_state_input: bool = None  # type: ignore

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike):
        raise NotImplementedError(
            "LaMDAConfig.create() is not supported for JAX. Use load_pytorch() for PyTorch training."
        )

    def load_pytorch(self, train_config, weight_path: str):
        import safetensors.torch

        from openpi.models_pytorch.pi0_lamda_pytorch import PI0LamdaPytorch

        model = PI0LamdaPytorch(config=self)
        if weight_path:
            safetensors.torch.load_model(model, weight_path)
        return model

    @override
    def inputs_spec(self, *, batch_size: int = 1):
        h, w = LAMDA_IMAGE_RESOLUTION
        image_spec = jax.ShapeDtypeStruct([batch_size, h, w, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct(
            [batch_size, self.action_horizon, self.action_dim], jnp.float32,
        )
        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        return nnx.Nothing

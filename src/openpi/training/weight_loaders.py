import dataclasses
import logging
import re
from typing import Protocol, runtime_checkable

import flax.traverse_util
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download

logger = logging.getLogger(__name__)

# Prefixes belonging to the gen branch that must be discarded when loading
# LaMDA weights for action-prediction-only use.
_LAMDA_GEN_PREFIXES = (
    "time_embedder.",
    "vae2llm.",
    "llm2vae.",
    "latent_pos_embed.",
)
# Prefixes that belong to the und branch and should be kept.
_LAMDA_KEEP_PREFIXES = (
    "language_model.",
    "vit_model.",
    "connector.",
    "vit_pos_embed.",
)


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "gs://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        # Add all missing LoRA weights.
        return _merge_params(loaded_params, params, missing_regex=".*lora.*")


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")


@dataclasses.dataclass(frozen=True)
class LaMDAWeightLoader:
    """Loads LaMDA und-branch weights from safetensors, discarding the gen branch.

    The gen-branch parameters (``time_embedder``, ``vae2llm``, ``llm2vae``,
    ``latent_pos_embed``) are filtered out.  The Action Expert parameters are
    *not* present in the pretrained checkpoint and will remain randomly
    initialised (``strict=False``).

    Args:
        checkpoint_path: Path to the directory or single ``.safetensors`` file
            containing the full LaMDA checkpoint.
    """

    checkpoint_path: str

    def load_pytorch(self, model) -> None:
        """Load filtered weights into a PyTorch ``nn.Module`` (in-place).

        Uses ``strict=False`` so that keys only present in the model (e.g.
        action expert projections) are left at their initialised values.
        """
        import glob
        import os

        import safetensors.torch

        path = self.checkpoint_path

        # Collect all safetensors shards
        if os.path.isdir(path):
            shard_files = sorted(glob.glob(os.path.join(path, "*.safetensors")))
        else:
            shard_files = [path]

        if not shard_files:
            raise FileNotFoundError(f"No safetensors files found at {path}")

        filtered_state: dict = {}
        for shard in shard_files:
            shard_state = safetensors.torch.load_file(shard, device="cpu")
            for k, v in shard_state.items():
                if any(k.startswith(p) for p in _LAMDA_GEN_PREFIXES):
                    continue
                filtered_state[k] = v

        missing, unexpected = model.load_state_dict(filtered_state, strict=False)
        logger.info(
            "LaMDAWeightLoader: loaded %d params, %d missing (random init), %d unexpected (skipped).",
            len(filtered_state),
            len(missing),
            len(unexpected),
        )
        if missing:
            logger.info("Missing keys (expected – action expert etc.): %s", missing[:20])

    def load(self, params: at.Params) -> at.Params:
        """JAX/Flax weight loading path (filters gen branch keys)."""
        loaded_params = _model.restore_params(download.maybe_download(self.checkpoint_path), restore_type=np.ndarray)
        flat = flax.traverse_util.flatten_dict(loaded_params, sep="/")
        filtered = {
            k: v for k, v in flat.items()
            if not any(k.startswith(p.rstrip(".")) for p in _LAMDA_GEN_PREFIXES)
        }
        loaded_params = flax.traverse_util.unflatten_dict(filtered, sep="/")
        return _merge_params(loaded_params, params, missing_regex=".*")


def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.

    Returns:
        A new dictionary with the merged parameters.
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    # First, take all weights that are a subset of the reference weights.
    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            result[k] = v.astype(flat_ref[k].dtype) if v.dtype != flat_ref[k].dtype else v

    flat_loaded.clear()

    # Then, merge any missing weights as defined by the missing regex.
    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            result[k] = flat_ref[k]

    return flax.traverse_util.unflatten_dict(result, sep="/")

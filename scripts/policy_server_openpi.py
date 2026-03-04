#!/usr/bin/env python
"""OpenPI policy inference server for dual-process LIBERO evaluation.

Run in OpenPI/lamda environment (Python>=3.11, openpi deps installed).
  From repo root: python scripts/policy_server_openpi.py --checkpoint_dir <ckpt> --norm_stats_path <norm.json> [--port 5555]

Uses scripts.lavida_policy_utils (LaViDa model + tokenizer; no libero/robosuite).
Dependencies: pyzmq, msgpack, numpy (and openpi + torch). Client connects via ZMQ REQ.
"""
from __future__ import annotations

# Disable jaxtyping so tokenized_prompt shape mismatches do not block inference
import os
os.environ["JAXTYPING_DISABLE"] = "1"

import argparse
import pathlib
import sys
import time
import traceback

import msgpack
import numpy as np
import zmq

# Repo root on path so we can import scripts.lavida_policy_utils (pure module, no libero/robosuite)
OPENPI_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(OPENPI_ROOT))

import scripts.lavida_policy_utils as _policy_utils
from scripts.lavida_policy_utils import IMAGE_SIZE, build_policy, ensure_uint8_hwc


# ---------- msgpack ndarray protocol ----------
def _encode_ndarray(obj):
    if isinstance(obj, np.ndarray):
        return {"dtype": str(obj.dtype), "shape": list(obj.shape), "data": obj.tobytes()}
    if isinstance(obj, dict):
        return {k: _encode_ndarray(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_encode_ndarray(x) for x in obj]
    return obj


def _decode_ndarray(obj):
    if isinstance(obj, dict) and set(obj.keys()) == {"dtype", "shape", "data"}:
        return np.frombuffer(obj["data"], dtype=obj["dtype"]).reshape(obj["shape"]).copy()
    if isinstance(obj, dict):
        return {k: _decode_ndarray(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decode_ndarray(x) for x in obj]
    return obj


def pack_response(payload):
    return msgpack.packb(_encode_ndarray(payload), use_bin_type=True)


def unpack_request(raw):
    return _decode_ndarray(msgpack.unpackb(raw, raw=False))


# ---------- obs_dict preprocess (before policy.infer; do not touch tokenized_*) ----------
def preprocess_obs_dict(obs_dict: dict) -> None:
    """In-place: ensure observation/image and wrist_image are uint8 HWC; state is float32 (7,) or (8,); prompt str."""
    img_keys = ("observation/image", "observation/wrist_image")
    for key in img_keys:
        if key not in obs_dict:
            continue
        img = np.asarray(obs_dict[key])
        if img.ndim == 3 and img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
        if np.issubdtype(img.dtype, np.floating):
            img = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
        elif img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        obs_dict[key] = img
    if "observation/state" in obs_dict:
        s = np.asarray(obs_dict["observation/state"], dtype=np.float32).flatten()
        if s.size >= 8:
            obs_dict["observation/state"] = s[:8].copy()
        elif s.size >= 7:
            obs_dict["observation/state"] = s[:7].copy()
        else:
            obs_dict["observation/state"] = np.resize(s, 8).astype(np.float32)
    if "prompt" in obs_dict:
        obs_dict["prompt"] = str(obs_dict["prompt"])


def main():
    ap = argparse.ArgumentParser(description="OpenPI policy server (ZMQ REP)")
    ap.add_argument("--checkpoint_dir", required=True, help="Path to checkpoint dir (model.safetensors)")
    ap.add_argument("--norm_stats_path", required=True, help="Path to norm_stats JSON")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--bind", default="0.0.0.0")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--image_size", type=int, default=IMAGE_SIZE, help=f"Image size for obs (default {IMAGE_SIZE}, must match eval)")
    args = ap.parse_args()

    policy = build_policy(args.checkpoint_dir, args.norm_stats_path, args.device)
    print("[policy_server] policy_utils=", _policy_utils.__file__)
    print("[policy_server] build_policy=", build_policy)
    endpoint = f"tcp://{args.bind}:{args.port}"
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(endpoint)
    print(f"[policy_server] listening on {endpoint}")

    step_idx = 0
    while True:
        try:
            raw = socket.recv()
        except zmq.ZMQError:
            break
        try:
            req = unpack_request(raw)
            state = np.asarray(req.get("state", np.zeros(8)), dtype=np.float32)
            agentview = req.get("agentview")
            wrist = req.get("wrist")
            prompt = req.get("prompt", "")
            if agentview is None:
                agentview = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
            else:
                agentview = ensure_uint8_hwc(np.asarray(agentview))
            if wrist is None:
                wrist = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
            else:
                wrist = ensure_uint8_hwc(np.asarray(wrist))
            obs_dict = {
                "observation/state": state,
                "observation/image": agentview,
                "observation/wrist_image": wrist,
                "prompt": str(prompt),
            }
            preprocess_obs_dict(obs_dict)
            t0 = time.perf_counter()
            out = policy.infer(obs_dict)
            latency_ms = (time.perf_counter() - t0) * 1000
            actions_arr = np.asarray(out["actions"]).reshape(-1)[:7]
            actions_str = ", ".join(f"{x:.4f}" for x in actions_arr.tolist())
            print(f"[Server] Step {step_idx}: Predicted Actions = [{actions_str}]  latency_ms = {latency_ms:.2f}")
            step_idx += 1
            # policy output_transforms (Unnormalize + LiberoOutputs) use norm_stats by key: actions -> norm_stats["actions"] (7d), state -> norm_stats["state"] (8d)
            action = np.asarray(out["actions"], dtype=np.float32)
            if action.ndim >= 2:
                action = action[0]
            if action.ndim == 2:
                action = action[0]
            action = action.reshape(-1)[:7]
            if len(action) < 7:
                action = np.resize(action, 7)
            resp = {"ok": True, "action": action.astype(np.float32)}
        except Exception as e:
            resp = {
                "ok": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
        socket.send(pack_response(resp))


if __name__ == "__main__":
    main()

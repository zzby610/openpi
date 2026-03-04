#!/usr/bin/env python
"""LIBERO env client for dual-process evaluation (connects to OpenPI policy server).

Run in .venv_libero14 / Python 3.10 libero+robosuite environment.
  python scripts/env_client_libero.py --bddl_path /path/to/task.bddl [--server tcp://127.0.0.1:5555]

Start policy_server_openpi.py first in the OpenPI/lamda environment.
Dependencies: pyzmq, msgpack, numpy, libero, robosuite.
"""




from __future__ import annotations

import os, logging
_orig_fh = logging.FileHandler
def _safe_filehandler(filename, *args, **kwargs):
    if filename == "/tmp/robosuite.log":
        os.makedirs("/export/ra/zoubiyu/tmp", exist_ok=True)
        filename = "/export/ra/zoubiyu/tmp/robosuite.log"
    return _orig_fh(filename, *args, **kwargs)
logging.FileHandler = _safe_filehandler

import argparse
import os
import time
from datetime import datetime

import imageio
import msgpack
import numpy as np
import zmq

# ---------- msgpack ndarray protocol (must match server) ----------
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


def pack_request(payload):
    return msgpack.packb(_encode_ndarray(payload), use_bin_type=True)


def unpack_response(raw):
    return _decode_ndarray(msgpack.unpackb(raw, raw=False))


def ensure_uint8_hwc(img):
    img = np.asarray(img)
    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    if img.ndim == 3 and img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    return img


def extract_obs(obs, image_size: int):
    """Extract state, agentview, wrist from env obs (compatible with tuple/dict structures)."""
    actual = obs[0] if isinstance(obs, (tuple, list)) else obs
    if isinstance(actual, dict) and "observation" in actual:
        actual = actual["observation"]
    if not isinstance(actual, dict):
        actual = {}
    # state: robot0_joint_pos or first key containing 'joint_pos', else zeros(8)
    state = None
    if "robot0_joint_pos" in actual:
        state = np.asarray(actual["robot0_joint_pos"], dtype=np.float32)
    else:
        for k, v in actual.items():
            if "joint_pos" in k:
                state = np.asarray(v, dtype=np.float32)
                break
    if state is None:
        state = np.zeros(8, dtype=np.float32)
    state = np.asarray(state).reshape(-1)
    # images
    agentview = actual.get("agentview_image", np.zeros((image_size, image_size, 3)))
    wrist = actual.get("robot0_eye_in_hand_image", np.zeros((image_size, image_size, 3)))
    agentview = ensure_uint8_hwc(agentview)
    wrist = ensure_uint8_hwc(wrist)
    return state, agentview, wrist


def bddl_to_prompt(bddl_path: str) -> str:
    """Turn bddl filename into a short prompt (e.g. task_0.bddl -> 'task 0')."""
    name = os.path.basename(bddl_path).replace(".bddl", "").replace("_", " ")
    return name.strip() or "task"


def main():
    ap = argparse.ArgumentParser(description="LIBERO env client (ZMQ REQ to policy server)")
    ap.add_argument("--bddl_path", required=True, help="Full path to .bddl file")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--server", default="tcp://127.0.0.1:5555")
    ap.add_argument("--image_size", type=int, default=384)
    ap.add_argument("--success_steps", type=int, default=10)
    ap.add_argument("--prompt", default="", help="Task prompt; empty => derived from bddl filename")
    ap.add_argument("--video_dir", default="./eval_videos", help="Directory to save episode videos")
    args = ap.parse_args()

    prompt = args.prompt.strip() or bddl_to_prompt(args.bddl_path)
    bddl_path = os.path.abspath(args.bddl_path)
    if not os.path.isfile(bddl_path):
        raise FileNotFoundError(f"BDDL not found: {bddl_path}")

    from libero.libero.envs import OffScreenRenderEnv

    env = OffScreenRenderEnv(
        bddl_file_name=bddl_path,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=[args.image_size, args.image_size],
        camera_widths=[args.image_size, args.image_size],
    )

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(args.server)
    print(f"[env_client] connected to {args.server}")

    successes = 0
    bddl_stem = os.path.basename(bddl_path).replace(".bddl", "")
    for ep in range(args.episodes):
        obs = env.reset()
        frames = []
        consecutive_success = 0
        step_succeeded = None
        for step in range(args.steps):
            state, agentview, wrist = extract_obs(obs, args.image_size)
            img = np.asarray(agentview)
            frame = (img * 255).clip(0, 255).astype(np.uint8) if np.issubdtype(img.dtype, np.floating) else img.astype(np.uint8).copy()
            frames.append(frame)
            req = {
                "state": state,
                "agentview": agentview,
                "wrist": wrist,
                "prompt": prompt,
            }
            socket.send(pack_request(req))
            raw = socket.recv()
            resp = unpack_response(raw)
            if not resp.get("ok", False):
                print(resp.get("traceback", resp.get("error", "unknown error")))
                raise RuntimeError(resp.get("error", "policy server error"))
            action = np.asarray(resp["action"], dtype=np.float64).reshape(-1)[:7]
            if len(action) < 7:
                action = np.resize(action, 7)
            result = env.step(action)
            obs = result[0]
            reward = result[1] if len(result) > 1 else 0.0
            done = result[2] if len(result) > 2 else False
            info = result[3] if len(result) > 3 else {}
            if info.get("success", False):
                consecutive_success += 1
                if step_succeeded is None:
                    step_succeeded = step
                if consecutive_success >= args.success_steps:
                    successes += 1
                    print(f"Episode {ep + 1}: success at step {step_succeeded} (held {args.success_steps})")
                    break
            else:
                consecutive_success = 0
                step_succeeded = None
            if done:
                print(f"Episode {ep + 1}: done at step {step}, no success")
                break
        else:
            if step_succeeded is None:
                print(f"Episode {ep + 1}: max steps {args.steps}, no success")
        if frames:
            os.makedirs(args.video_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_path = os.path.join(
                args.video_dir,
                f"{bddl_stem}_ep{ep}_{timestamp}.mp4",
            )
            try:
                imageio.mimsave(video_path, frames, fps=20)
                print(f"Video saved to: {video_path}")
            except Exception as e:
                print(f"[env_client] Failed to save video to {video_path}: {e}")
    env.close()

    sr = successes / args.episodes
    print(f"SR = {successes}/{args.episodes} = {sr}")


if __name__ == "__main__":
    main()

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
import robosuite.utils.transform_utils as T

import imageio
import msgpack
import numpy as np
import zmq
import robosuite.utils.transform_utils as T


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


def extract_obs(obs, image_size):
    actual = obs[0] if isinstance(obs, (tuple, list)) else obs
    if isinstance(actual, dict) and "observation" in actual:
        actual = actual["observation"]

    # --- 核心修复：提取末端笛卡尔坐标，而不是关节角 ---
    if "robot0_eef_pos" in actual:
        # 1. 提取 3D 位置 (X, Y, Z)
        eef_pos = actual["robot0_eef_pos"]
        
        # 提取旋转并转换为轴角 (Axis-Angle)
        eef_quat = actual["robot0_eef_quat"]
        eef_axis_angle = T.quat2axisangle(eef_quat) # <--- 就是这句救命的代码
        
        # 提取 2D 夹爪状态
        gripper = actual.get("robot0_gripper_qpos", np.zeros(2))
        
        # 拼接成 8 维
        state = np.concatenate([eef_pos, eef_axis_angle, gripper]).astype(np.float32)
    else:
        state = np.zeros(8, dtype=np.float32)

    agentview = actual.get("agentview_image", np.zeros((image_size, image_size, 3), dtype=np.uint8))
    wrist = actual.get("robot0_eye_in_hand_image", np.zeros((image_size, image_size, 3), dtype=np.uint8))

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
    from libero.libero import benchmark

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
    chunk_use = 4  # use first N steps of each chunk for open-loop execution

    # Load official fixed init_states for this task (align with Libero benchmark / training data).
    task_suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task_name = bddl_stem
    all_task_names = task_suite.get_task_names()
    try:
        task_id = all_task_names.index(task_name)
    except ValueError:
        task_id = None
    init_states = None
    if task_id is not None:
        init_states = task_suite.get_task_init_states(task_id)
        if hasattr(init_states, "numpy"):
            init_states = init_states.numpy()
    if init_states is not None and len(init_states) == 0:
        init_states = None

    for ep in range(args.episodes):
        obs = env.reset()
        if init_states is not None and len(init_states) > 0:
            init_state = init_states[ep % len(init_states)]
            if hasattr(init_state, "numpy"):
                init_state = init_state.numpy()
            obs = env.set_init_state(init_state)
        for _ in range(10):
            dummy_action = np.zeros(7)
            obs, _, _, _ = env.step(dummy_action)
        frames = []
        action_buffer = []
        consecutive_success = 0
        step_succeeded = None
        for step in range(args.steps):
            state, agentview, wrist = extract_obs(obs, args.image_size)

            if step < 5:
                print("STATE:", state)
                print("STATE MAX:", np.abs(state).max())
            img = np.asarray(agentview)
            frame = (img * 255).clip(0, 255).astype(np.uint8) if np.issubdtype(img.dtype, np.floating) else img.astype(np.uint8).copy()
            frames.append(frame)

            # Request new action chunk only when buffer is empty (action chunking / open-loop).
            if len(action_buffer) == 0:
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
                chunk = np.asarray(resp["action"], dtype=np.float64)
                print("CHUNK SHAPE:", chunk.shape)
                print("CHUNK SAMPLE:", chunk[0])
                if chunk.ndim == 1:
                    chunk = chunk.reshape(1, -1)
                n_use = min(chunk_use, len(chunk))
                for i in range(n_use):
                    row = chunk[i].reshape(-1)[:7]
                    if len(row) < 7:
                        row = np.resize(row, 7)
                    action_buffer.append(row.astype(np.float64))

            if len(action_buffer) == 0:
                action = np.zeros(7, dtype=np.float64)
            else:
                action = np.asarray(action_buffer.pop(0), dtype=np.float64).reshape(-1)[:7]

            if step < 5:
                print("ACTION:", action)
                print("ACTION MAX:", np.abs(action).max())
            if len(action) < 7:
                action = np.resize(action, 7)
            # Gripper binarization: index 6 -> 1.0 (grip) or -1.0 (release) for reliable rigid contact
            action[6] = 1.0 if action[6] > 0.8 else -1.0

            # LIBERO expects actions in [-1, 1]. Unbounded model output can yield 4.0+ after unnormalize → 乱飞.
            # Clip to safe range before env.step (only first 6 dims; dim 6 is already ±1).
            action_before_clip = action.copy()
            action[:6] = np.clip(action[:6], -1.0, 1.0)
            if step < 10 and np.any(np.abs(action_before_clip[:6]) > 1.0):
                print(f"[clip] step {step}: pre-clip max |action[:6]| = {np.abs(action_before_clip[:6]).max():.3f} -> post-clip max = {np.abs(action[:6]).max():.3f}")

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

#!/usr/bin/env python
"""Evaluate a LaViDa policy on LIBERO-Spatial tasks.

Compatible with robosuite==1.5.2 + mujoco==3.x + libero.

Verify:
  python scripts/eval_lavida_libero.py \
      --checkpoint_dir <ckpt> --norm_stats_path <norm.json> --n_episodes 1
"""
from __future__ import annotations

import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import sys
import types
import argparse
import json
import logging
import pathlib

import numpy as np
import torch

# Redirect /tmp/robosuite.log to writable path (before any import robosuite)
_robosuite_log_redirect = "/export/ra/zoubiyu/tmp/robosuite.log"
_orig_file_handler = logging.FileHandler

def _patched_file_handler(filename, *args, **kwargs):
    if filename == "/tmp/robosuite.log":
        os.makedirs(os.path.dirname(_robosuite_log_redirect), exist_ok=True)
        filename = _robosuite_log_redirect
    return _orig_file_handler(filename, *args, **kwargs)

logging.FileHandler = _patched_file_handler

# ======================================================================
# Robosuite 1.5.x ↔ LIBERO compatibility patch
# ======================================================================
def patch_robosuite_for_libero():
    """Must run BEFORE importing libero.libero.envs."""
    import inspect
    import robosuite

    # --- (A) load_controller_config fallback ---
    if not hasattr(robosuite, "load_controller_config"):
        found = False
        for p in [
            "robosuite.controllers.controller_factory",
            "robosuite.controllers",
            "robosuite.utils.control_utils",
        ]:
            try:
                mod = __import__(p, fromlist=["load_controller_config"])
                func = getattr(mod, "load_controller_config", None)
                if func:
                    robosuite.load_controller_config = func
                    found = True
                    break
            except Exception:
                continue
        if not found:

            def _manual_load_config(default_controller="OSC_POSE"):
                ctrl_map = {
                    "IK_POSE": "inverse_kinematics",
                    "OSC_POSE": "osc_pose",
                    "OSC_POSITION": "osc_position",
                    "JOINT_VELOCITY": "joint_velocity",
                    "JOINT_TORQUE": "joint_torque",
                }
                cfg_name = ctrl_map.get(default_controller, "osc_pose")
                cfg_path = os.path.join(
                    os.path.dirname(robosuite.__file__), "controllers", "config", f"{cfg_name}.json"
                )
                if os.path.exists(cfg_path):
                    with open(cfg_path) as f:
                        return json.load(f)
                return {
                    "type": "OSC_POSE",
                    "input_max": 1, "input_min": -1,
                    "output_max": 0.05, "output_min": -0.05,
                    "kp": 150, "damping_ratio": 1, "impedance_mode": "fixed",
                    "kp_limits": [0, 300], "damping_ratio_limits": [0, 10],
                    "control_delta": True, "uncouple_pos_ori": False,
                    "control_ori": True, "interpolation": None, "ramp_ratio": 0.2,
                }

            robosuite.load_controller_config = _manual_load_config

    # --- (B) Fake modules that LIBERO imports from robosuite <=1.4 ---
    import robosuite.environments.manipulation.manipulation_env as menv
    if "robosuite.environments.manipulation" not in sys.modules:
        import robosuite.environments.manipulation as _manip_pkg
        sys.modules["robosuite.environments.manipulation"] = _manip_pkg

    name = "robosuite.environments.manipulation.single_arm_env"
    if name not in sys.modules:
        m = types.ModuleType(name)
        m.SingleArmEnv = menv.ManipulationEnv
        sys.modules[name] = m
        print("[patch] injected robosuite.environments.manipulation.single_arm_env.SingleArmEnv -> ManipulationEnv")

    from robosuite.robots.fixed_base_robot import FixedBaseRobot
    name = "robosuite.robots.single_arm"
    if name not in sys.modules:
        m = types.ModuleType(name)
        m.SingleArm = FixedBaseRobot
        sys.modules[name] = m
        print("[patch] injected robosuite.robots.single_arm.SingleArm -> FixedBaseRobot")

    # --- (C) Patch ManipulationEnv.__init__: drop mount_types (LIBERO adds it, robosuite 1.5 rejects it) ---
    _orig_manip_init = menv.ManipulationEnv.__init__
    _LIBERO_ONLY_KWARGS = {"mount_types", "mount_names"}

    def _patched_manip_init(self, *args, **kwargs):
        dropped = {k for k in _LIBERO_ONLY_KWARGS if k in kwargs}
        for k in dropped:
            del kwargs[k]
        if dropped:
            print(f"[patch] Dropped LIBERO-only kwargs from ManipulationEnv: {dropped}")
        return _orig_manip_init(self, *args, **kwargs)

    menv.ManipulationEnv.__init__ = _patched_manip_init

    # --- (D2) Ensure REGISTERED_ROBOTS entries have .arms (e.g. MountedPanda) ---
    from robosuite.robots.robot import REGISTERED_ROBOTS
    for key, cls in list(REGISTERED_ROBOTS.items()):
        if not isinstance(cls, type):
            continue
        if hasattr(cls, "arms"):
            continue
        setattr(cls, "arms", ["right"])
        if cls.__name__ == "MountedPanda":
            print(f"[patch] Set {key}(MountedPanda).arms = ['right']")
        else:
            print(f"[patch] Set {key}({cls.__name__}).arms = ['right'] (missing)")

    # --- (D3) Patch Robot.__init__ to fill missing .arms at instantiation time ---
    from robosuite.robots.robot import Robot as RobotBase
    _orig_robot_init = RobotBase.__init__

    def _patched_robot_init(self, *args, **kwargs):
        robot_type = kwargs.get("robot_type")
        if robot_type is None and len(args) > 0:
            robot_type = args[0]
        if robot_type is not None:
            from robosuite.robots.robot import REGISTERED_ROBOTS as _REG
            if robot_type in _REG:
                cls = _REG[robot_type]
                if isinstance(cls, type) and not hasattr(cls, "arms"):
                    setattr(cls, "arms", ["right"])
                    print(f"[patch] Filled missing arms for {robot_type} ({cls.__name__})")
        return _orig_robot_init(self, *args, **kwargs)

    RobotBase.__init__ = _patched_robot_init
    print("[patch] Patched robosuite.robots.robot.Robot.__init__ to fill missing arms")

    # --- (E_base) Override ManipulatorModel.default_base (NotImplementedError) ---
    import robosuite.models.bases as bases_mod
    candidate_dicts = []
    for _k, v in bases_mod.__dict__.items():
        if not isinstance(v, dict):
            continue
        if len(v) == 0:
            continue
        if not all(isinstance(key, str) for key in v.keys()):
            continue
        candidate_dicts.append((len(v), v))
    if not candidate_dicts:
        print("bases_mod.__dict__.keys() (first 50):", list(bases_mod.__dict__.keys())[:50])
        raise RuntimeError("Cannot locate base registry in robosuite.models.bases")
    candidate_dicts.sort(key=lambda x: -x[0])
    base_registry = candidate_dicts[0][1]
    base_key = next(iter(base_registry.keys()))
    import robosuite.models.robots.manipulators.manipulator_model as mm

    def _default_base_getter(self):
        return base_key

    mm.ManipulatorModel.default_base = property(_default_base_getter)
    print(f"[patch] Overrode ManipulatorModel.default_base -> '{base_key}' from bases registry")

    # (D)(E)(F) default_gripper / grippers patches disabled to avoid
    # read-only property and shape mismatches; keep minimal patch for reset/step only.

# Run 1.5.x patches only when robosuite >= 1.5; 1.4.x has no fixed_base_robot etc.
import robosuite as _rs
_rs_ver = getattr(_rs, "__version__", "0.0.0")
_rs_parts = _rs_ver.split(".")[:2]
try:
    _rs_major = int(_rs_parts[0])
    _rs_minor = int(_rs_parts[1]) if len(_rs_parts) > 1 else 0
except (ValueError, IndexError):
    _rs_major, _rs_minor = 0, 0
if (_rs_major, _rs_minor) >= (1, 5):
    patch_robosuite_for_libero()
else:
    print(f"[patch] robosuite {_rs_ver} detected, skipping 1.5.x compatibility patches")

# ======================================================================
# LaViDa policy (from pure module; no libero/robosuite here)
# ======================================================================
OPENPI_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(OPENPI_ROOT))
from scripts.lavida_policy_utils import (
    IMAGE_SIZE,
    MAX_TOKEN_LEN,
    PadTokenizedPrompt,
    build_policy,
    ensure_uint8_hwc,
    load_norm_stats,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")


def _to_bddl_filename(bddl_value: str) -> str:
    name = os.path.basename(bddl_value.strip())
    if not name.endswith(".bddl"):
        name = name.replace(" ", "_").replace("-", "_") + ".bddl"
    else:
        name = name.replace(" ", "_").replace("-", "_")
    return name


def get_libero_tasks():
    import libero.libero.benchmark as benchmark

    task_suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    num_tasks = task_suite.get_num_tasks()
    all_task_names = task_suite.get_task_names()
    tasks = []
    for i in range(num_tasks):
        task_info = task_suite.get_task(i)
        if isinstance(task_info, (list, tuple)):
            actual_obj, bddl_raw = task_info[0], task_info[1]
        else:
            actual_obj = task_info
            bddl_raw = getattr(task_info, "bddl_file", all_task_names[i])
        bddl_filename = _to_bddl_filename(str(bddl_raw))
        problem_info = getattr(actual_obj, "problem_info", None) or {}
        if not isinstance(problem_info, dict):
            problem_info = {}
        tasks.append({
            "task_id": i,
            "name": all_task_names[i],
            "bddl": bddl_filename,
            "robot": problem_info.get("robot_name", "Panda"),
            "controller": problem_info.get("controller", "OSC_POSE"),
        })
    return task_suite, tasks


def _save_video(frames, path):
    import imageio

    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(path), frames, fps=20)


def run_episode(env, policy, task_name, max_steps, success_steps, video_path, init_state=None):
    obs = env.reset()
    if init_state is not None:
        try:
            env.set_init_state(init_state)
            obs = env.reset()
        except Exception as e:
            logging.warning("set_init_state / reset after init_state failed: %s", e)
    frames = []
    consecutive_success, step = 0, 0
    while step < max_steps:
        actual_obs = obs[0] if isinstance(obs, (tuple, list)) else obs
        if isinstance(actual_obs, dict) and "observation" in actual_obs:
            actual_obs = actual_obs["observation"]
        obs_dict = {
            "observation/state": np.asarray(actual_obs.get("robot0_joint_pos", np.zeros(8))),
            "observation/image": ensure_uint8_hwc(
                actual_obs.get("agentview_image", np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3)))
            ),
            "observation/wrist_image": ensure_uint8_hwc(
                actual_obs.get("robot0_eye_in_hand_image", np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3)))
            ),
            "prompt": str(task_name),
        }
        out = policy.infer(obs_dict)
        action = np.asarray(out["actions"])[0]
        if action.ndim == 2:
            action = action[0]
        obs, reward, done, info = env.step(action)[:4]
        if video_path:
            frames.append(obs_dict["observation/image"])
        if info.get("success", False):
            consecutive_success += 1
            if consecutive_success >= success_steps:
                if video_path:
                    _save_video(frames, video_path)
                return 1
        else:
            consecutive_success = 0
        if done:
            break
        step += 1
    if video_path:
        _save_video(frames, video_path)
    return 0


# ======================================================================
# Main
# ======================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--norm_stats_path", type=str, required=True)
    parser.add_argument("--video_dir", type=str, default="eval_outputs/videos")
    parser.add_argument("--n_episodes", type=int, default=20)
    args = parser.parse_args()

    device = "cuda"
    policy = build_policy(args.checkpoint_dir, args.norm_stats_path, device)
    task_suite, tasks = get_libero_tasks()

    from libero.libero.envs import OffScreenRenderEnv
    import libero.libero as libero_module

    libero_path = os.path.dirname(libero_module.__file__)
    bddl_root = os.path.join(libero_path, "bddl_files")
    bddl_spatial = os.path.join(bddl_root, "libero_spatial")

    results = {}
    for task in tasks:
        task_name = task["name"]
        task_id = task["task_id"]
        full_bddl_path = os.path.join(bddl_spatial, task["bddl"])
        if not os.path.exists(full_bddl_path):
            full_bddl_path = os.path.join(bddl_root, task["bddl"])
        if not os.path.exists(full_bddl_path):
            print(f"[warning] BDDL not found for '{task_name}', skipping.")
            continue

        print(f"Task: {task_name}")
        print(f"  BDDL: {full_bddl_path}")

        init_states = None
        try:
            init_states = task_suite.get_task_init_states(task_id)
            if hasattr(init_states, "numpy"):
                init_states = init_states.numpy()
        except Exception as e:
            logging.warning("get_task_init_states(%s) failed: %s", task_id, e)

        # Minimal env args: bddl + render/camera only (no robots/controller/render_gpu_device_id)
        env = OffScreenRenderEnv(
            bddl_file_name=full_bddl_path,
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            camera_names=["agentview", "robot0_eye_in_hand"],
            camera_heights=[IMAGE_SIZE, IMAGE_SIZE],
            camera_widths=[IMAGE_SIZE, IMAGE_SIZE],
        )

        successes = 0
        for ep in range(args.n_episodes):
            v_path = os.path.join(args.video_dir, f"{task_name}_ep{ep}.mp4")
            init_state = None
            if init_states is not None and len(init_states) > 0:
                idx = ep % len(init_states)
                init_state = init_states[idx]
                if hasattr(init_state, "numpy"):
                    init_state = init_state.numpy()
            successes += run_episode(env, policy, task_name, 600, 10, v_path, init_state=init_state)
        env.close()

        sr = successes / args.n_episodes
        results[task_name] = sr
        print(f"  SR: {sr}")

    if results:
        print(f"Final Average SR: {np.mean(list(results.values()))}")


if __name__ == "__main__":
    main()

# Quick self-check (run from repo root):
#   python -c "import scripts.eval_lavida_libero as s; import sys; print('single_arm_env' in sys.modules); import robosuite.environments.manipulation.single_arm_env as x; print(x.SingleArmEnv)"

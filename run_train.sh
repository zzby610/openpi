#!/usr/bin/env bash
# LaMDA + LIBERO 极速过拟合 / 小规模验证训练
# 使用多卡 DDP（torchrun），挂载 train_lamda_libero 配置
#
# 首次训练前请先计算 norm stats（否则会报错缺少 norm_stats）:
#   PYTHONPATH=src:$LAMDA_REPO:$LEROBOT_SRC python scripts/compute_norm_stats.py train_lamda_libero
# 并设置 HF_LEROBOT_HOME 与本次一致。

set -e

# -----------------------------------------------------------------------------
# 环境与路径（按需修改）
# -----------------------------------------------------------------------------
OPENPI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$OPENPI_ROOT"
# 本地 LIBERO LeRobot 数据集：LeRobot 通过 HF_LEROBOT_HOME/repo_id 找数据
# 若数据在 /data/datasets/biyuz/libero_lerobot，则 HF_LEROBOT_HOME 设为父目录，config 中 repo_id=libero_lerobot
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/data/datasets/biyuz}"
# LaMDA 源码路径（用于 import diffusion.lamda 等）
LAMDA_REPO="${LAMDA_REPO:-/export/ra/zoubiyu/repo/LaMDA/LaMDA}"
# 可选：LeRobot 源码（若用本地 lerobot_repo）
LEROBOT_SRC="${LEROBOT_SRC:-$OPENPI_ROOT/lerobot_repo/src}"

export PYTHONPATH="${OPENPI_ROOT}/src:${LAMDA_REPO}:${LEROBOT_SRC}:${PYTHONPATH:-}"

# 配置名（对应 config.py 中 TrainConfig name）
CONFIG_NAME="${CONFIG_NAME:-train_lamda_libero}"
EXP_NAME="${EXP_NAME:-lamda_libero_overfit}"
# 使用的 GPU 数量（不设则用当前可见 GPU 数）
NUM_GPUS="${NUM_GPUS:-}"

# -----------------------------------------------------------------------------
# 多卡 DDP 启动
# -----------------------------------------------------------------------------
if [ -n "$NUM_GPUS" ] && [ "$NUM_GPUS" -gt 1 ]; then
  echo "Using torchrun with $NUM_GPUS GPUs (config=$CONFIG_NAME, exp=$EXP_NAME)"
  exec torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="$NUM_GPUS" \
    scripts/train_pytorch.py \
    "$CONFIG_NAME" \
    --exp_name "$EXP_NAME" \
    "$@"
else
  echo "Using single process (config=$CONFIG_NAME, exp=$EXP_NAME). Set NUM_GPUS=2 (or more) for multi-GPU."
  exec python scripts/train_pytorch.py \
    "$CONFIG_NAME" \
    --exp_name "$EXP_NAME" \
    "$@"
fi

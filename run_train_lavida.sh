#!/usr/bin/env bash
# LaViDa + LIBERO 正式训练（VLM 冻结，只训动作头）
# 使用多卡 DDP（torchrun），挂载 train_lavida_libero 配置
#
# 首次训练前请先计算 norm stats（否则会报错缺少 norm_stats）:
#   PYTHONPATH=src:$LAVIDA_REPO:$LEROBOT_SRC python scripts/compute_norm_stats.py train_lavida_libero
# 并设置 HF_LEROBOT_HOME 与本次一致。

set -e

# -----------------------------------------------------------------------------
# 环境与路径（按需修改）
# -----------------------------------------------------------------------------
OPENPI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$OPENPI_ROOT"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/data/datasets/biyuz}"
# LaViDa 源码路径
LAVIDA_REPO="${LAVIDA_REPO:-/export/ra/zoubiyu/repo/LaViDa}"
LEROBOT_SRC="${LEROBOT_SRC:-$OPENPI_ROOT/lerobot_repo/src}"

# 关键点：要把 LaViDa 的路径放在最前面，并且还要把 LaViDa/llava 也加进去
export PYTHONPATH=src:/export/ra/zoubiyu/repo/LaViDa:/export/ra/zoubiyu/repo/LaMDA/LaMDA:$PYTHONPATH

CONFIG_NAME="${CONFIG_NAME:-train_lavida_libero}"
EXP_NAME="${EXP_NAME:-lavida_libero}"
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

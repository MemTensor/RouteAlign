#!/bin/bash
# router_21K_Qwen3-0.6B learned_fusion 复现脚本
# 结果: Dataset-Avg 68.64% / Sample-Avg 63.07% / Total Cost $110.42
set -e

# 本目录: data/router_21K_Qwen3-0.6B_learned_fusion
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"   # 仓库根目录
cd "$ROOT"

PYTHON="${PYTHON:-python}"

"$PYTHON" -m adapter.evaluate_learned_fusion \
  --test "$HERE/test.jsonl" \
  --routing-records "$HERE/test_task_profiles_cache.jsonl" \
  --model-profiles "$HERE/model_profiles_data_driven.json" \
  --judge-profiles "$HERE/model_profiles_judge.json" \
  --model-pool "$HERE/model_pool.yaml" \
  --config "$HERE/demo_router_config.yaml" \
  --train-path "$HERE/train.jsonl" \
  --train-routing-records "$HERE/train_task_profiles.jsonl" \
  --train-task-profiles "$HERE/train_task_profiles.jsonl" \
  --output-dir "$HERE/output" \
  --knn-k 150 --knn-beta 5.0 --knn-cap --ridge-alpha 1.0 --cost-lambda 0.4 \
  --train-emb-cache "$HERE/emb_train.npy" \
  --eval-emb-cache "$HERE/emb_test.npy"

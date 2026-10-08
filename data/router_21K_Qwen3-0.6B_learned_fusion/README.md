# Reproduction bundle: router_21K_Qwen3-0.6B shared scorer

Self-contained inputs and expected outputs for the main RouteAlign service-selection result on
LLMRouterBench: Dataset-Avg 68.64%, Sample-Avg 63.07%, generation cost $110.42.

Run from the repository root:

```bash
bash data/router_21K_Qwen3-0.6B_learned_fusion/run.sh
```

The evaluation is fully offline: requirement profiles, model profiles, and
query embeddings are read from the caches below, and no LLM endpoint is
contacted. Results are written to `output/` and can be diffed against the
expected files already present there.

## Files

| File | Description |
|---|---|
| `train.jsonl` | Training split, 8,710 requests with recorded scores, usages, and costs for all 13 candidates |
| `test.jsonl` | Test split, 3,738 requests with the same record schema |
| `train_task_profiles.jsonl` | Requirement profiles of the training requests, predicted by the fine-tuned Qwen3-0.6B profiler |
| `test_task_profiles_cache.jsonl` | Requirement profiles of the test requests, same profiler |
| `model_profiles_data_driven.json` | Outcome profiles aggregated from the training split under profiler-assigned requirement weights |
| `model_profiles_judge.json` | Response-assessment profiles from rubric-based scoring, an independent evidence channel |
| `emb_train.npy` / `emb_test.npy` | L2-normalized 384-dimensional query embeddings for neighborhood retrieval |
| `model_pool.yaml` / `demo_router_config.yaml` | Candidate service attributes and router configuration used by the reported run |
| `run.sh` | One-command reproduction; writes `eval_result.json`, `routing_records.jsonl`, and `routing_analysis.json` to `output/` |
| `output/` | Expected outputs of the reported run |

Core parameters: `--knn-k 150 --knn-beta 5.0 --knn-cap --ridge-alpha 1.0 --cost-lambda 0.4`.

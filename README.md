<h1 align="center">
    RouteAlign: From Model Selection to Requirement-Aware Routing
</h1>

RouteAlign is an LLM routing framework built on requirement-profile alignment. A compact profiler represents each request as a mixture over ten capability dimensions and a difficulty level. The same requirement coordinates organize historical outcomes into difficulty-conditioned model profiles offline, and condition the aggregation of local evidence from semantically related requests online. A shared scorer compares all feasible candidates through fixed feature definitions under service constraints and a generation-cost preference, so model suitability becomes a relation between what a request demands and what a candidate has measurably delivered.

## 🎯 Who Should Pay Attention

* **LLM serving engineers:** Candidate pools keep changing as providers release new versions. RouteAlign updates model evidence without refitting the scorer, and its coefficients trained on an initial seven-model pool remain valid through additions, a version replacement, a removal, and a price change.
* **Routing researchers:** Requirement coordinates give global measurement and local retrieval a common basis. Ablations with scorer refitting quantify what each evidence channel contributes, and the feature interface makes every estimate inspectable.
* **Benchmark practitioners:** The complete LLMRouterBench evaluation ships with cached requirement profiles, model profiles, and query embeddings. One command reproduces the reported result exactly, with no LLM endpoint involved.
* **Cost-sensitive deployments:** A single coefficient λ exposes the quality-cost trade-off as an explicit deployment preference. The default configuration spends 11.5% less than its cost-agnostic variant while attaining higher Dataset-Avg.

## ✨ Core Contributions

1. **Requirement-profile alignment over shared coordinates.**
   Requests and model measurements live on the same ten capability dimensions and four difficulty levels. A fine-tuned Qwen3-0.6B profiler predicts capability weights and difficulty from request text with deterministic decoding, supervised by about 21,000 GPT-5.4-annotated requirement profiles. Because the profiler output describes task demands rather than candidate identities, request annotations and embeddings stay reusable as the pool evolves.

2. **Difficulty-conditioned model profiles from two channels.**
   Outcome profiles aggregate historical task scores into capability-by-difficulty cells, weighting each observation by the requirement weights of its request, so failures count in proportion to what the task required. A second channel aggregates rubric-based response assessments on its native scale. Conditioning on difficulty preserves performance differences that an aggregate model score would average together: two models tied on easy requests separate when the same capabilities are demanded at higher difficulty.

3. **Capability-conditioned local evidence and a shared scorer.**
   For each request, a frozen encoder retrieves 150 neighboring training queries. Their outcomes, weighted by semantic proximity and by their own requirement weights, yield per-capability local estimates that the incoming capability mixture recomposes. A ridge regression scorer combines global compatibility, local evidence, conditional priors, and relative comparisons into ten features shared across candidates, and selection maximizes estimated quality minus λ times estimated cost. Fitting takes seconds on CPU and adds no learned per-model parameters.

4. **Reuse under pool changes through a fixed evidence interface.**
   Model identity indexes evidence without defining a learned output class. Adding or replacing a candidate updates its profiles, costs, and local records while request annotations and scorer coefficients stay fixed. Across nine evaluated pool configurations, RouteAlign-Frozen attains the highest Dataset-Avg in every configuration and a mean of 64.61%, 4.41 points above the strongest baseline.

## 📊 Results

Evaluation covers 8,710 training and 3,738 test requests from ten LLMRouterBench datasets (AIME, ArenaHard, GPQA, HLE, LiveCodeBench, LiveMathBench, MMLU-Pro, SimpleQA, SWE-bench, τ²-bench) with 13 candidate LLMs, split by dataset and prompt at seed 42 with a 70% training fraction.

| Policy | Sample-Avg (%) | Dataset-Avg (%) | Cost ($) |
|---|---|---|---|
| **RouteAlign** | **63.07** | **68.64** | 110.42 |
| RouteLLM (strongest competing router) | 61.69 | 66.42 | 109.27 |
| RouterDC | 58.56 | 65.24 | 122.62 |
| AvengersPro | 58.91 | 63.63 | 61.35 |
| GPT-5 (highest Dataset-Avg among candidates) | 58.37 | 65.11 | 124.82 |
| Gemini 2.5 Pro (highest Sample-Avg among candidates) | 59.18 | 59.55 | 233.97 |

* **Routing quality.** RouteAlign improves the strongest competing router by 1.38 Sample-Avg and 2.22 Dataset-Avg points, at 1.1% additional cost. It surpasses the strongest individual candidate under each metric: +4.70 Sample-Avg and +3.53 Dataset-Avg over GPT-5 at 11.5% lower generation cost.
* **Model complementarity.** GPT-5 receives 1,852 of the 3,738 selections; the rest go to twelve alternatives. Request-specific allocation exceeds what any single candidate delivers, at lower cost than committing to the best one.
* **Evidence ablations.** Removing all local quality evidence costs 3.12 Dataset-Avg points even after the scorer is refitted, the largest decrease among the evaluated removals. The cost preference moves expenditure from $124.82 to $110.42 while adding 0.37 Dataset-Avg points relative to λ = 0.
* **Routing overhead.** With requirement profiles and embeddings cached, routing 3,738 requests takes about 119 seconds, or 31.9 ms per request. A serving measurement with the 0.6B profiler in the loop processes 51.2 requests per second at concurrency 32 with 0.60 s median latency.

A supplementary pool-evolution study fits the scorer on the initial seven candidates and keeps its coefficients fixed through five additions, one version replacement, one removal, and one price change. RouteAlign-Frozen leads all nine configurations.

## 🛠️ Quick Start

### Installation

Python 3.10 or later.

```bash
pip install -r requirements.txt
```

PyTorch and Transformers are only needed to regenerate query embeddings from a local profiler checkpoint; the reproduction bundle ships with the caches already computed.

### Reproduce the main result

```bash
bash data/router_21K_Qwen3-0.6B_learned_fusion/run.sh
```

The bundle is self-contained and fully offline: requirement profiles, both model-profile channels, and query embeddings are read from cache, and ridge fitting plus scoring take seconds. Expected output, also shipped under `output/` for direct comparison:

```
Dataset-Avg accuracy: 68.64%
Sample-Avg accuracy:  63.07%
Total cost:           $110.42
```

### Full pipeline from raw benchmark data

The stages below regenerate every artifact in the bundle. Steps that call LLM endpoints resume from checkpoints and read credentials from environment variables referenced in the configs.

```bash
# 1. Convert LLMRouterBench records and split train/test (requires the
#    LLMRouterBench package; the prepared splits already ship in the bundle)
python -m adapter.data_prep --config config/adapter_config.yaml

# 2. Annotate question metadata and capability weights with a strong model
python -m adapter.build_question_bank --config config/adapter_config.yaml

# 3. Grade candidate responses and aggregate model profiles
python -m adapter.run_offline --config config/adapter_config.yaml

# 4. Pure profile-based routing evaluation on the test split
python -m adapter.evaluate --config config/adapter_config.yaml
```

The learned-fusion path used for the reported numbers continues with:

```bash
# 5. Requirement profiles for the training split
python -m adapter.profile_train_tasks --config config/adapter_config.yaml \
    --train data/seed42_split0.7/train.jsonl \
    --output data/seed42_split0.7/train_task_profiles.jsonl

# 6. Outcome profiles aggregated from training records
python -m adapter.build_profile_from_records \
    --train data/seed42_split0.7/train.jsonl \
    --task-profiles data/seed42_split0.7/train_task_profiles.jsonl \
    --model-pool config/model_pool.yaml \
    --output data/seed42_split0.7/model_profiles_data_driven.json

# 7. Fusion scoring and evaluation (see the bundle run.sh for all arguments)
python -m adapter.evaluate_learned_fusion --config config/demo_router_config.yaml \
    --knn-k 150 --knn-beta 5.0 --knn-cap --ridge-alpha 1.0 --cost-lambda 0.4
```

### Fine-tuned requirement profilers

The registry in `config/trained_models.yaml` covers six local checkpoints (SFT and GRPO variants at 0.6B, 1.7B, and 4B). Edit `model_path` entries to point at your checkpoints, then:

```bash
bash scripts/serve_trained_models.sh start     # one sglang service per GPU
python -m adapter.run_trained_eval             # profile-based routing per checkpoint
python -m adapter.run_trained_fusion           # full fusion pipeline per checkpoint
```

Both drivers support `--models` for subsets, `--max-samples-per-dataset` for smoke tests, and `--summary-only` for aggregation.

### Routing engine as a library

```python
from src import DemoRouterConfig, RoutingEngine

engine = RoutingEngine(DemoRouterConfig.load("config/demo_router_config.yaml"))
decision = engine.route("Compare the revenue trends across these three reports")
result = engine.answer("Compare the revenue trends across these three reports")
```

The engine requires a reachable requirement-profiler endpoint (a local checkpoint served by sglang, or any OpenAI-compatible API configured under `task_profile_model`) and prepared model profiles. `route` returns only the decision; `answer` additionally calls the selected candidate, which requires live endpoints in `config/model_pool.yaml`.

## 📁 Repository Structure

```
├── src/                             # Routing framework
│   ├── capabilities.py              # Capability dimensions, difficulty levels, tiers
│   ├── schemas.py                   # Request, grading, profile, and decision structures
│   ├── prompts.py                   # Grading and requirement-profiler prompts
│   ├── llm_client.py                # OpenAI-compatible client with exponential backoff
│   ├── question_bank.py             # Plug-in question-bank loading
│   ├── model_pool.py                # Candidate service attributes
│   ├── grading.py                   # Capability grading of candidate responses
│   ├── profiling.py                 # Outcome-profile aggregation, confidence, tiers
│   ├── task_profiler.py             # Online requirement prediction
│   ├── request_features.py          # Request metadata detection
│   ├── router.py                    # Feasibility filter and compatibility scoring
│   ├── offline.py                   # Offline measurement pipeline with resume
│   ├── engine.py                    # Online routing engine
│   └── config.py                    # Configuration loading
├── adapter/                         # LLMRouterBench adapter and evaluation pipeline
│   ├── data_prep.py                 # Benchmark conversion, split, stratified sampling
│   ├── build_question_bank.py       # Metadata and capability-weight annotation
│   ├── gen_model_pool.py            # Model-pool template with price estimation
│   ├── run_offline.py               # Offline grading and profiling driver
│   ├── evaluate.py                  # Test-set routing evaluation
│   ├── profile_train_tasks.py       # Requirement profiles for training requests
│   ├── build_profile_from_records.py # Outcome profiles from training records
│   ├── compute_router_embeddings.py  # Query embedding caches from the profiler
│   ├── knn_priors.py                # Neighborhood statistics and local estimates
│   ├── evaluate_with_cached_tasks.py # Routing from cached requirement profiles
│   ├── evaluate_strategy_map.py      # Conditional priors and difficulty strata
│   ├── evaluate_learned_fusion.py    # Shared scorer fitting and evaluation
│   ├── run_trained_eval.py          # Batch evaluation of fine-tuned profilers
│   └── run_trained_fusion.py        # Full fusion pipeline per profiler checkpoint
├── config/                          # Router, adapter, pool, and per-profiler configs
├── scripts/serve_trained_models.sh  # sglang serving for local profiler checkpoints
└── data/router_21K_Qwen3-0.6B_learned_fusion/  # Reproduction bundle with expected outputs
```

## ⚙️ Configuration Notes

* Credentials are never stored in configs. Any `api_key` or `base_url` written in all capitals is resolved from the environment at runtime, so set `GRADER_API_KEY`, `METADATA_API_KEY`, and `MODEL_ENDPOINT_API_KEY` before running stages that call external endpoints.
* `config/model_pool.yaml` carries service attributes for the thirteen evaluated candidates. Prices are least-squares estimates from benchmark usage records and latency fields are placeholders; verify both against your provider before live routing.
* The requirement schema is defined once in `src/capabilities.py` and shared by the grader, the profiler, and the question-bank annotator, which keeps the three annotation procedures consistent by construction.

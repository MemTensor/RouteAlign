<h1 align="center">
    RouteAlign: Aligning Task Requirements with Model Profiles for LLM Service Selection
</h1>

Selecting a large language model service requires estimating how measured model performance transfers to the demands of a new task. Aggregate scores average over heterogeneous task demands, while semantic similarity alone can overlook differences in the capabilities needed for successful completion. RouteAlign makes task requirements the common basis for constructing model profiles and selecting services. A compact profiler predicts capability weights and a difficulty level for each request. Historical requirements determine how observed outcomes contribute to difficulty-conditioned model profiles and to capability-conditioned local estimates; incoming requirements determine how those measurements inform the current decision. A shared quality scorer integrates evidence across these scopes, service metadata determines eligibility, and a generation-cost preference controls selection among eligible services.

## 🎯 Who Should Pay Attention

* **LLM service platform engineers:** Candidate catalogs change as models are introduced, updated, repriced, and retired. RouteAlign separates evidence maintenance from scorer fitting, so an updated catalog requires fresh candidate measurements but no new training run, and the same feature schema describes every candidate regardless of when it joined.
* **Routing and service-selection researchers:** Requirement coordinates give global measurement and local retrieval a shared basis. The effective-weight form of local alignment makes explicit how each neighboring outcome is reweighted, and feature ablations with scorer refitting quantify the predictive contribution of every evidence group.
* **Benchmark practitioners:** The complete LLMRouterBench evaluation ships with cached requirement profiles, model profiles, and query embeddings. One command reproduces the reported operating point exactly, without contacting any LLM endpoint.
* **Cost-aware applications:** The coefficient λ is an explicit application-level control rather than a fitted parameter. The default configuration reduces generation expenditure by 11.5% relative to its cost-agnostic variant while attaining higher Dataset-Avg, and changing λ shifts the operating point without refitting the quality estimator.

## ✨ Core Contributions

1. **A requirement-aware formulation of LLM service selection.**
   Task competence, service eligibility, and generation-cost preference are separated explicitly. Ten capability coordinates and four difficulty levels describe what successful completion requires; service metadata such as context capacity, supported modalities, and tool access determines admissibility; the coefficient λ expresses the application preference. A capability weight influences the quality comparison among eligible services, while an explicit execution requirement decides eligibility itself, so high estimated quality cannot compensate for a known violation of an execution condition.

2. **Requirement-profile alignment as a measurement and selection mechanism.**
   The same requirement coordinates organize evidence on both sides of the comparison. Historical capability weights distribute observed outcomes across capability-by-difficulty profile cells, and reweight neighboring outcomes within each retrieved neighborhood; incoming weights read the global profile at the predicted difficulty and recompose per-capability local estimates. A shared ridge scorer integrates both scopes of evidence with conditional priors and relative candidate comparisons through ten candidate-agnostic features, and selection maximizes estimated quality minus λ times estimated generation cost. Model identity indexes the evidence without defining a learned output class.

3. **An empirical assessment from selection value to evidence maintenance.**
   Four research questions cover selection quality and generation expenditure, the contribution of each evidence group, execution overhead with and without requirement prediction, and scorer reuse under controlled catalog changes. Across nine catalog configurations, coefficients fitted on the initial seven candidates remain effective through five additions, one version replacement, one removal, and one repricing, provided candidate evidence is refreshed.

## 📊 Results

Evaluation uses the recorded requests, responses, task scores, and generation costs of LLMRouterBench: ten datasets (AIME, ArenaHard, GPQA, HLE, LiveCodeBench, LiveMathBench, MMLU-Pro, SimpleQA, SWE-bench, τ²-bench) with 13 candidate LLM services, partitioned by dataset and prompt at seed 42 with a 70% training fraction, giving 8,710 training and 3,738 test requests.

| Policy | Sample-Avg (%) | Dataset-Avg (%) | Cost ($) |
|---|---|---|---|
| **RouteAlign** | **63.07** | **68.64** | 110.42 |
| RouteLLM (strongest competing policy) | 61.69 | 66.42 | 109.27 |
| RouterDC | 58.56 | 65.24 | 122.62 |
| AvengersPro | 58.91 | 63.63 | 61.35 |
| GPT-5 (highest Dataset-Avg among candidates) | 58.37 | 65.11 | 124.82 |
| Gemini 2.5 Pro (highest Sample-Avg among candidates) | 59.18 | 59.55 | 233.97 |

* **RQ1, selection value.** RouteAlign improves on the strongest competing policy by 1.38 Sample-Avg and 2.22 Dataset-Avg points at 1.1% higher generation cost. It attains the highest per-dataset score, including ties, on seven of the ten datasets; the largest single gain over RouteLLM is on SWE-bench, from 16.67% to 33.33%. Against direct invocation, RouteAlign improves on GPT-5 by 4.70 Sample-Avg and 3.53 Dataset-Avg points over the same 3,738 requests while reducing expenditure by 11.5%, and exceeds the strongest individual Sample-Avg by 3.89 points. GPT-5 serves 1,852 selections; the remaining 1,886 requests go to twelve alternatives, so the gain over a strong default is realized through allocation across the catalog.
* **RQ2, evidence contribution.** Removing all local quality evidence causes the largest decrease, 3.12 Dataset-Avg and 1.66 Sample-Avg points, even after the scorer is refitted. Requirement-conditioned estimates overlap: their individual removals trade small changes between the two averages rather than degrading both. Setting λ = 0 raises expenditure from $110.42 to $124.82, so the default preference saves 11.5% while adding 0.37 Dataset-Avg points relative to the cost-agnostic variant.
* **RQ3, execution overhead.** With requirement profiles and embeddings cached, selection takes about 119 seconds over 3,738 requests, or 31.9 ms per request. A separate endpoint that also runs the 0.6B requirement profiler serves 51.2 requests per second at concurrency 32, with 0.60 s median and 0.74 s 95th-percentile latency and no observed errors. RouteAlign exchanges additional evidence aggregation for its quality gains; it is not the lowest-overhead policy in the comparison.
* **RQ4, scorer reuse under catalog changes.** RouteAlign-Frozen, fitted only on the initial seven-candidate catalog, keeps the highest Dataset-Avg in all nine update configurations, with an equally weighted mean of 64.61% against 60.20% for the strongest baseline. Admitting GPT-5 raises Dataset-Avg from 64.18% to 68.55% with no coefficient update; removing it costs 4.52 points; halving the price of Gemini 2.5 Flash changes the score by 0.04 points.

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

# 4. Pure profile-based selection evaluation on the test split
python -m adapter.evaluate --config config/adapter_config.yaml
```

The shared-scorer path used for the reported numbers continues with:

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

# 7. Scorer fitting and evaluation (see the bundle run.sh for all arguments)
python -m adapter.evaluate_learned_fusion --config config/demo_router_config.yaml \
    --knn-k 150 --knn-beta 5.0 --knn-cap --ridge-alpha 1.0 --cost-lambda 0.4
```

### Fine-tuned requirement profilers

The registry in `config/trained_models.yaml` covers six local checkpoints (SFT and GRPO variants at 0.6B, 1.7B, and 4B). Edit `model_path` entries to point at your checkpoints, then:

```bash
bash scripts/serve_trained_models.sh start     # one sglang service per GPU
python -m adapter.run_trained_eval             # profile-based selection per checkpoint
python -m adapter.run_trained_fusion           # full shared-scorer pipeline per checkpoint
```

Both drivers support `--models` for subsets, `--max-samples-per-dataset` for smoke tests, and `--summary-only` for aggregation.

### Selection engine as a library

```python
from src import DemoRouterConfig, RoutingEngine

engine = RoutingEngine(DemoRouterConfig.load("config/demo_router_config.yaml"))
decision = engine.route("Compare the revenue trends across these three reports")
result = engine.answer("Compare the revenue trends across these three reports")
```

The engine requires a reachable requirement-profiler endpoint (a local checkpoint served by sglang, or any OpenAI-compatible API configured under `task_profile_model`) and prepared model profiles. `route` returns only the decision; `answer` additionally calls the selected candidate, which requires live endpoints in `config/model_pool.yaml`.

## 📁 Repository Structure

```
├── src/                             # RouteAlign framework
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
│   ├── router.py                    # Eligibility filter and compatibility scoring
│   ├── offline.py                   # Offline measurement pipeline with resume
│   ├── engine.py                    # Online selection engine
│   └── config.py                    # Configuration loading
├── adapter/                         # LLMRouterBench adapter and evaluation pipeline
│   ├── data_prep.py                 # Benchmark conversion, split, stratified sampling
│   ├── build_question_bank.py       # Metadata and capability-weight annotation
│   ├── gen_model_pool.py            # Candidate-registry template with price estimation
│   ├── run_offline.py               # Offline grading and profiling driver
│   ├── evaluate.py                  # Test-set selection evaluation
│   ├── profile_train_tasks.py       # Requirement profiles for training requests
│   ├── build_profile_from_records.py # Outcome profiles from training records
│   ├── compute_router_embeddings.py  # Query embedding caches from the profiler
│   ├── knn_priors.py                # Neighborhood statistics and local estimates
│   ├── evaluate_with_cached_tasks.py # Selection from cached requirement profiles
│   ├── evaluate_strategy_map.py      # Conditional priors and difficulty strata
│   ├── evaluate_learned_fusion.py    # Shared scorer fitting and evaluation
│   ├── run_trained_eval.py          # Batch evaluation of fine-tuned profilers
│   └── run_trained_fusion.py        # Full shared-scorer pipeline per checkpoint
├── config/                          # Framework, adapter, catalog, and per-profiler configs
├── scripts/serve_trained_models.sh  # sglang serving for local profiler checkpoints
└── data/router_21K_Qwen3-0.6B_learned_fusion/  # Reproduction bundle with expected outputs
```

## ⚙️ Configuration Notes

* Credentials are never stored in configs. Any `api_key` or `base_url` written in all capitals is resolved from the environment at runtime, so set `GRADER_API_KEY`, `METADATA_API_KEY`, and `MODEL_ENDPOINT_API_KEY` before running stages that call external endpoints.
* `config/model_pool.yaml` carries service attributes for the thirteen evaluated candidates. Prices are least-squares estimates from benchmark usage records and latency fields are placeholders; verify both against your provider before live selection.
* The requirement schema is defined once in `src/capabilities.py` and shared by the grader, the profiler, and the question-bank annotator, which keeps the three annotation procedures consistent by construction.

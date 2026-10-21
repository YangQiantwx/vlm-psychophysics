# Classic Precision

Lossless T2 luminance-increment and T3 orientation-discrimination stimuli, with finite four-model evaluation. The source signal, official preprocessing and BF16 input are measured separately.

**Current state:** the implementation is frozen and all 22 local tests passed, including incremental readiness and CPU pipeline fault handling. Full runtime acceptance remains pending. Historical input paths passed eight real GPU qualification calls. This 3,504-video / four-model panel has not started; its CPU calibration and real worker first-two checks remain runtime acceptance steps. The independent review record is linked below.

- [Protocol](EXPERIMENT_PLAN.md): exact stimuli, denominators, model settings and acceptance criteria.
- [Independent validation](validation/code_review.json): frozen source hashes, checks and runtime limits.
- [Experiment log](report.md): completed smoke findings and why the protocol changed.
- `classic_precision/protocol.json`: pinned scientific configuration and model revisions.
- `classic_precision/`: rendering, validation, RGB48 input, inference, durable storage, scheduling and analysis.
- `tests/classic_precision/`: meaningful source, codec, restart and scoring checks; no model downloads.

| Task | Conditions | Videos/model |
| --- | --- | ---: |
| T2 main | 13 increments × 3 backgrounds × 24 onsets × 2 sides; 500 ms | 1,872 |
| T2 control | 3 increments × 3 backgrounds × 24 onsets × 2 sides; 167 ms | 432 |
| T3 | 12 angles × 24 onsets × 2 signs × 2 sides; plus 90° deduplicated | 1,200 |
| Total | Same 3,504 videos for each of four models | 14,016 planned answers |

A failed unfinished request may be retried once. Completed raw answers are terminal, including incorrect or malformed answers. Attempts and retries are counted separately from unique scientific rows.

## Environment

Use an existing Python 3.11+ environment. CPU construction dependencies are in `pyproject.toml`; no command creates an environment or upgrades Torch/CUDA. On the established workstations, use the existing `qy` for construction and the already qualified inference SDK for model work. Do not install a second conflicting SDK merely to match a generic requirements list.

The previously qualified Linux SDK recorded torch 2.13.0, transformers 5.17.0, numpy 2.3.5, av 17.1.0 and accelerate 1.12.0. vLLM 0.31.0 and the official `transformers-cosmos3` package are required by their respective loaders; exact installed versions/source identities are written to each run. Cosmos requires `--cosmos-package /path/to/transformers-cosmos3` from the existing official framework checkout. Model licenses/access remain those of the upstream repositories. The qualified Cosmos shim is from [NVIDIA/cosmos-framework](https://github.com/NVIDIA/cosmos-framework/tree/0fa3ba479a7c1c5be28f81fe084b9ea544d697c9/packages/transformers-cosmos3), commit `0fa3ba479a7c1c5be28f81fe084b9ea544d697c9`. Reuse that package or check out that exact revision; do not substitute an unpinned fork.

For the reviewed `04a2d52` runner with vLLM hybrid KV groups, the optional `runtime_compat/` layer reads capacity from vLLM's initialized, group-aware native API. Enable it only for the owned run with `CLASSIC_PRECISION_KV_COMPAT=1` and prepend `runtime_compat` plus this repository to `PYTHONPATH`. The original runner must match its pinned SHA or startup exits with code 78. This changes only `_kv_capacity`; the existing 65,536-token gate, scientific inputs, generation and calibration remain unchanged. Capacity receipts record the compatibility layer, original runner and installed native helper SHA values. Verify the installed SDK separately and require actual worker first-two acceptance. The existing local-function RPC probes also require `VLLM_ALLOW_INSECURE_SERIALIZATION=1` in these trusted local processes; do not apply it globally or expose an untrusted RPC service.

```bash
python -B -m pytest -q -p no:cacheprovider tests/classic_precision
python -B -m classic_precision --help
```

Commands run from this repository root (or after installing this package into an approved existing environment). Set these task-specific paths:

```bash
RUN=/absolute/path/to/new-classic-run
MODELS=/absolute/path/to/model-snapshots
COSMOS_PACKAGE=/absolute/path/to/transformers-cosmos3
INFERENCE_PYTHON=/absolute/path/to/existing/inference/python
```

## Integrated pipeline

`pipeline` connects the same independently callable stages. Run it as an ordinary detached process on each workstation. On the producing host, use the existing CPU environment for the coordinator and the qualified inference interpreter for model work:

```bash
python -B -m classic_precision manifest --run "$RUN"
python -B -m classic_precision pipeline --run "$RUN" --model-root "$MODELS" \
  --produce --gpu-shards 0:0 1:1 --shard-count 3 \
  --worker-python "$INFERENCE_PYTHON" --cosmos-package "$COSMOS_PACKAGE" \
  --cpu-memory-gib 32 --ram-reserve-gib 4
```

On the receiving host, use `--gpu-shards 0:2` and omit `--produce`. Supply the same immutable manifest and mirror verified inputs using `sync-ready` below. For direct host-to-host copying, the producing pipeline can instead start the mirror with `--sync-destination-root`, `--destination-ssh`, `--destination-python` and `--destination-code-root`. Keep only one producer and one destination mirror for each run. A pipeline's `COMPLETE` applies to its assigned local GPU shards; the final global summary still requires all 14,016 answers from both hosts.

By default, the CPU ceiling follows the process's available cores, with two reserved for control and inference. `--cpu-max-workers` can lower that ceiling. Admission adapts to available RAM and the ready backlog. Calibration inputs receive priority; enough buffered inputs reduce production concurrency, but do not stop eventual full generation. Codec/BLAS threads are capped to prevent each worker spawning another full thread pool. A shortage of memory or input is a visible waiting state, not a model attempt.

Each GPU loads one model at a time and processes one independent video request at a time. Its startup buffer defaults to 100 locally verified videos, together with the model's completed calibration. After loading, it consumes further ready videos without reloading. This is streaming execution, not a batch of 100 videos in one context. The manifest still fixes all 3,504 videos and all 14,016 planned answers.

## Independent generation and validation

```bash
python -B -m classic_precision manifest --run "$RUN"
python -B -m classic_precision generate --run "$RUN" --shard-count 3
python -B -m classic_precision validate --run "$RUN" --require-complete
```

Run generation in its own ordinary detached process while the calibration and GPU supervisors below run concurrently. Generation prioritizes the 128 unique-state representatives and each shard's first T2/T3 qualification inputs. Each completed video is decoded and checked before its atomic sidecar and durable ready record are published. The complete 3,504-row manifest is frozen at the start; a partial file is never a ready input.

`validate` is also available independently for a deliberate recheck of existing files. Generation already validates each video before publication, so a second whole-panel validation is not required merely to begin inference. Without `--require-complete`, validation reports partial coverage explicitly; it never certifies missing files as complete.

Copy the immutable manifest to the second workstation once. Incrementally transfer only newly verified ready videos with checksums and atomic destination publication. The destination publishes a row only after its own file is complete; its ready journal can temporarily lag behind the source. It must contain the same immutable per-row records, not a repeatedly overwritten snapshot of a growing source journal. No full media download to the Mac is required. Run the finite mirror as an ordinary process alongside generation, preferably on the source workstation:

```bash
python -B -m classic_precision sync-ready --manifest "$RUN/manifest.json" \
  --source-root "$RUN" --destination-root "$DEST_RUN" \
  --destination-ssh "$DEST_SSH" --destination-python "$DEST_PYTHON" \
  --destination-code-root "$DEST_CODE" --poll-seconds 10
```

The destination variables specify its run directory, an existing authorized SSH command, approved Python interpreter and this deployed codebase. If direct workstation SSH is unavailable, run the mirror on the Mac with `--source-ssh`, `--source-python` and `--source-code-root` as well. This relays bytes in memory and requires the Mac and mirror process to remain running until transfer completes. The worker queues themselves run on the workstations. Do not start two mirrors for the same destination.

Each model loads its own pinned official processor inside its own calibration/inference worker; no shared cross-model processor or extra processor network service is used. Only RGB48 decoding and the lossless FP32 array entry are common.

For each model, use its official processor to measure the 128 unique states on CPU as soon as their representative videos are ready, while the remaining videos continue to generate. This loads no model weights and makes no VLM calls:

```bash
python -B -m classic_precision calibrate --manifest "$RUN/manifest.json"   --model nemo --model-root "$MODELS" --output "$RUN/calibration" --input-poll-seconds 10
```

Repeat `--model qwen`, `cosmos` (with `--cosmos-package "$COSMOS_PACKAGE"`), and `gemma` as separate CPU processes. These wait for their fixed representative inputs; no full-panel completion is required. Keep the calibration receipts. Pre-BF16 uint16 recovery must be exact; post-BF16 signal errors are reported, not used to rewrite labels.

## Model snapshots

Reuse existing pinned snapshot directories. Download only missing approved models:

```bash
python -B -m classic_precision download --model-root "$MODELS" --models nemo qwen cosmos gemma
```

`protocol.json` pins repository revisions and relative snapshot locations. The package does not download weights during inference. Actual model config/tokenizer/processor source hashes and weight-shard sizes are retained without repeatedly hashing all weights.

## Three GPUs

Use one supervisor per GPU, with distinct onset shards. Fourier GPU0/1 use shards0/1; Radon GPU0 uses shard2. Each model shard has1,168 videos (768T2+400T3). Run each command in an ordinary detached session on its assigned host:

```bash
python -B -m classic_precision supervise --manifest "$RUN/manifest.json"   --output "$RUN/outputs" --model-root "$MODELS"   --cosmos-package "$COSMOS_PACKAGE" --gpu 0 --shard-index 0 --start-ready 100 --input-poll-seconds 10 --calibration-root "$RUN/calibration"
```

Use `--gpu 1 --shard-index 1` on the second Fourier card and `--gpu 0 --shard-index 2` on Radon. Each supervisor selects a fitting unfinished model from real free VRAM/RAM, performs a finite queue, releases its model and proceeds. Foreign processes are protected. A shared card is allowed; no fitting remaining model means resource waiting, not a fake GPU workload.

A worker starts once its model checkpoint and 128-state calibration are ready and its shard has 100 locally verified inputs (or generation is already complete). It loads the model once and consumes later ready rows without reloading. Waiting for an input never begins a model attempt. Each worker qualifies a predetermined T2/T3 pair for input integrity, native OFF, actual GPU tensor geometry, context and exact single-label output. Correctness is not an acceptance criterion. After technical qualification it continues automatically. No arbitrary whole-run wall deadline; no retry of completed answers. A missing answer has at most two persisted attempts in total. Real configuration/technical failures are surfaced rather than retried indefinitely.

For an explicitly scoped single model/shard, use `run` with the same arguments plus `--model nemo`. All requests have independent contexts. The source code uses actual vLLM choice constraints or the HF equivalent token-prefix/EOS grammar.

## Monitor and summarize

After all three workers start normally, one bounded check per hour is sufficient. The supervisors continue without an LLM. Copy small output journals/receipts from both hosts into distinct local directories; keep raw/token IDs. `status` reads journals without querying GPUs:

```bash
python -B -m classic_precision status --manifest "$RUN/manifest.json"   --inputs /path/to/fourier-outputs /path/to/radon-outputs
python -B -m classic_precision summarize --manifest "$RUN/manifest.json"   --inputs /path/to/fourier-outputs /path/to/radon-outputs   --output "$RUN/analysis" --require-complete --plot
```

Summary rejects duplicated answers, incompatible systems and unresolved technical outputs. It preserves failures, counts48/96 denominators, side/sign/onset effects, direction-pair success,432 matched duration pairs and model-load/call cost. Curves show measured points, not forced-monotone fits. Onset-block intervals are conditional on these fixed stimuli.

## Historical records

The research workspace retains `archive_index.json` and `archive/history.py` for verified compact formal trial records and retired source. Restore into a fresh destination before using historical analyzers. Historical smoke summaries belong in `report.md`; videos, credentials, model weights and raw reasoning do not belong in this code repository. The T4 originals and recent precision/input qualification evidence remain in the research workspace.

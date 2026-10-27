# Classic Precision

Controlled visual psychophysics with lossless RGB16 source videos. Source accuracy, framework preprocessing and model outputs are separate measurements. [Protocol](EXPERIMENT_PLAN.md) defines stimuli and numerical scope; [findings](report.md) retains observations and negative results. Local operational decisions are in the benchmark registry and project `LATEST.md`.

## Choose the protocol first

| Protocol | Input contract | Evidence |
|---|---|---|
| `controlled-rgb16` | RGB48 decode → FP32 `code*255/65535`; frozen per-model adapters and geometry | Complete Qwen/Nemo/Cosmos/Gemma T1–T3 curves |
| `native-default` | Official file uint8 reader; all180 frames; official default spatial processing and numerical order | Selected-cell deployment sensitivity |
| `native-original` | Same official uint8/all180 route; only supported native spatial configuration changes | Qwen/Cosmos/Nemo supported; Gemma native video unsupported in pinned vLLM |

Keep versions separate. The highest-scoring protocol at each cell is not a valid composite main curve. No completed comparison isolates only uint8 versus FP32.

The current panel has **T1 3,840 + T2 3,744 + T3 1,776 = 9,360 unique videos**. Current Qwen (QSA-corrected), Nemo, Cosmos and Gemma each completed the panel using their qualified official mixed NVFP4 checkpoint. Cosmos NVFP4 is1,743/3,840;3,112/3,744;1,673/1,776 and independently repeated144 T3 inputs with full-token/answer/finish agreement. The earlier Cosmos BF16 curve remains a historical reference, not part of the current NVFP4 panel.

Gemma O0 passed the same24 complete input/output cases on three cards across two hosts. Its controlled-input panel is now complete: **1,920/3,840 T1, 2,145/3,744 T2 and1,447/1,776 T3**. The original3,145 answers and later6,215 unique answers were merged only after finite shard closure; no old worker is active. All four models now have456 single-image controls plus960 ordered T1 controls each. The final Gemma static gap used three finite shards324/318/318; every assigned ID, durable raw/PASS binding, real wait0, owned-process exit and original-lock release passed. Old configurations and diagnostic answers remain separate.

## Inspect a configuration without starting inference

```bash
python -B -m classic_precision.reproduce list
python -B -m classic_precision.reproduce show --model qwen --protocol native-original \
  --model-dir /absolute/pinned-checkpoint --data-root /absolute/videos \
  --output qwen-native-original.json
```

This CPU-only entry exports pinned settings and classifies current versus historical code. It does not download weights, import Torch, initialize CUDA or launch a model. It refuses unsupported Gemma/no-resize and refuses to replace a controlled-input profile with generic native settings. The export is not a complete GPU job.

For actual native inference, use the corresponding sealed `native_path_smoke` plan. Preserve exact IDs/prompts, checkpoint/tokenizer revisions, model assets, SDK identities and generation controls. Bind local paths explicitly rather than copying another workstation's absolute paths. Query plan checks with:

```bash
python -B -m classic_precision.native_path_smoke --help
python -B -m classic_precision.native_path_smoke pair-check --plan /absolute/sealed-pair.json
```

The finite worker records INTENT → original raw answer → byte-bound PASS, with no automatic retry. Original physical GPU leases and real worker waits/engine closure control transitions. A wrong answer remains a completed result. Generic clean-machine launch/lease bootstrap is still a release gap, not something the configuration exporter claims to solve.

## Model-specific native requirements

- **Qwen:** existing scoped QSA deterministic selector, installed in the worker; retain actual `state=applied` evidence. A fixed seed/temperature alone is not a cross-card guarantee. Default/original spatial variants use `do_resize=True/False` at global and request scope.
- **Cosmos:** native ModelOpt/FlashInfer NVFP4; same resize settings as Qwen. The older BF16 HF/SDPA deployment is a separate reference.
- **Nemo:** original-size mode uses native vision-config fields `video_target_num_patches=1536` and `video_maintain_aspect_ratio=True`, through `native_path_smoke.nemo_original_geometry`. It does not accept Qwen's `do_resize` switch. The callback sets configuration, not pixels.
- **Gemma:** fixed vLLM0.31 video processing forces70 soft tokens/frame. Original512×768 yields1536 patches, incompatible with its original pooling/prompt-count path. No valid native original-size video speed is available; do not substitute the7.81s default-resize result or the controlled280-token/padded route.

All four native routes request reader indices0…179 and disable processor frame sampling. Verify both boundaries: requesting180 only at the reader was insufficient for Qwen/Cosmos. Actual native outputs are448×288 for default Qwen/Cosmos and512×512 for default Nemo; original-size geometry is768×512. Source size is **W×H768×512**.

Pinned native processors return FP32 for Gemma/Qwen/Cosmos and BF16 for Nemo; vLLM's common postprocessor converts floating values to model BF16. Gemma then applies internal `2*(x-.5)`. Our retained Gemma bridge explicitly casts before this scaling too, while its input/geometry differs. Direct HF FP32 input may scale before casting. Neither order is described as universally invalid. The two historical static151 configurations changed18 answers (102 versus100 correct) and remain separate evidence.

Static RGB16-to-FP32 ingress does **not** mean every model skips native resizing. Qwen/Cosmos retain768×512 with native resize disabled; Nemo retains source geometry with its qualified float-input adapter; Gemma single/ordered images retain native image resizing,768×512→960×624, before its BF16 visual-input bridge. Gemma controlled video instead retains the effective768×512 source and adds16 background rows for its native pooling grid. Static T1 uses two initial phases and ordered2/4-frame inputs; T2/T3 use152 unique event images across three frozen prompts. These controls test available spatial information, not an identical image/video processor or a uniquely isolated temporal mechanism.

## Dataset and source validation

Use the release's portable `dataset-index.json`, preserving IDs, prompts, LEFT/RIGHT truth, nominal parameters, file/decoded-stream SHA and audit identities. The complete9,360 source videos are public at [DrQY/vlm-psychophysics-classic-t1-t3](https://huggingface.co/datasets/DrQY/vlm-psychophysics-classic-t1-t3), v0.3, immutable commit `a28b0add9ee7621191b453c03d9a69d60e30f83e`; all remote file SHA256 identities match, and the original v0.1 four-family download and the v0.2 three-template download/readout both passed.

```python
from classic_precision.dataset import load_index, read_sample, rgb16_to_fp32
index = load_index("dataset-index.json")
codes, pts, receipt = read_sample(index, "t1rgb16_i00_sf0p5_m0p05_left", "data")
frames = rgb16_to_fp32(codes)  # controlled-input route only
```

The reader returns `uint16 [180,512,768,3]` in RGB order and verifies source integrity. This example is not the native uint8 route. The single complete `validation/source-qa.jsonl` supplies nominal/measured values, errors and their units. `metadata.jsonl` supports the standard HF table; `dataset-index.json` supports stable-ID lookup and integrity checks, so they serve distinct loader interfaces. Its audit-row SHA identifies each actual canonical audit row, not the common audit configuration.

**All9,360 videos /1,684,800 frames passed source QA**, with zero decoded RGB-code differences and no failed IDs. Maximum PTS rounding is0.333ms; onset/exposure frame membership is exact. T1 modulation error≤4.730e−6;T2 core ΔY error≤2.2641e−5;T3 PCA error≤5.3904e−5°. This does not certify unchanged model-side geometry/luminance or physical display calibration. T2/T3 share24 onsets `18+5k`,k0…23; these are timing repetitions, not24 independent scenes.

`python -m classic_precision.dataset --help` exposes bounded index/QA export and reading. Use `fetch --repo DrQY/vlm-psychophysics-classic-t1-t3 --revision a28b0add9ee7621191b453c03d9a69d60e30f83e --id ...` for the published source release. The legacy root CLI `python -m classic_precision --help` reconstructs only the original3,504 T2/T3 videos, not the current full9,360; do not silently use its base manifest for the final panel.

The workstations also expose `datasets/classic-t1-t3-v0.3/` with the same index and9,360 readable symlinks to retained originals. Use that directory as `data_root`; its `reader/` is the byte-verified public snapshot. Original active catalogs/paths remain unchanged. Readable names contain conditions/labels: pass the benchmark question and media to models, keeping filenames, parameters and truth out of model text.

### Readable names and stable IDs

Public videos are grouped in `T1/`, `T2/`, and `T3/`, with names such as `T1_left_sf4_contrast0.6_trial01.mkv`, `T2_right_bg0.05_delta0.15_10frames_trial01.mkv` and `T3_verticalLeft_tiltRight_0.7deg_trial01.mkv`. T3 tilt names describe the upper end;90° uses `horizontal`. Trial01–24 comes from the original phase/onset index. Stable sample IDs stay in the index and join existing QA/predictions; filenames carry no random suffix.

`classic_precision.dataset_names` is the single exporter for naming policy `classic-readable-v1`. It emits a new index and an old/new path map, rejecting collisions or inconsistent truth/trial metadata. `dataset.load_index/read_sample/fetch_sample` already resolves the index path without parsing filenames. Neither naming nor path lookup changes a model's reader/dtype/resize protocol. Frozen active workstation catalogs/profiles keep their original paths; new releases use the versioned index. Ten targeted naming/layout tests and all9,360 mappings passed; no video was rendered or re-encoded. Old publicv0.1 remains pinned in its tag. The pre-existing general candidate wheel is not claimed to contain this later naming delta; the public reader snapshot does.

## Code by purpose

| Purpose | Maintained modules |
|---|---|
| Dataset/source QA | `dataset`, `dataset_names`, `stimuli`, `source_audit`, `source_audit_t1` |
| Native deployment sensitivity | `native_path_smoke`; configuration/navigation: `reproduce`, `reproduction.json` |
| Controlled Cosmos NVFP4 | `nvfp4_cosmos_official`, `nvfp4_cosmos_classic`, `nvfp4_cosmos_t3` |
| Controlled Gemma/continuation | `gemma_nvfp4_continuation`, `gemma_speed`, `gemma_production`, `nvfp4_gemma` |
| Current static controls | `static_current`, `multi_current`, `gemma_static_current`, `gemma_multi_current` |
| Persistence/analysis/replay | `storage`, `summary`, `replay_audit` |
| Runtime compatibility | run-bound `runtime_compat`; QSA helper is `classic_precision_qsa_determinism.py` |

Adapter/minimal-config/batch-invariant/early-stop/parallel modules retain historical scientific code; they are not default reproduction routes. Frozen source paths/hashes are not renamed for appearance. Runtime, tests and validation evidence remain separate. `reproduce list` gives the code map without importing historical engines.

## Fixed controls, speed and safe pause

The qualified stack is Torch2.13.0, Transformers5.17.0, vLLM0.31.0, numpy2.3.5 and FlashInfer0.7.0.post1 with matching cu130/SM120f support. Use an existing approved environment; do not upgrade shared SDKs merely to replay a result. Exact package/source identities are run-specific. No input route makes all weight, activation and KV precision identical across architectures.

Generation is thinking OFF, temperature0,top_p1,seed0,greedy,context65536,output cap16384,repetition penalty1 and LEFT/RIGHT constrained with native EOS. Independent contexts and complete terminal token IDs are retained. CPU offload is placement, not an extra quantization step. Qwen native uses24GiB expert offload/BF16KV4GiB; NemoKV2GiB; CosmosNVFP4KV17GiB; GemmaO0 offload0/KV32GiB. FlashInfer dummy autotuning is disabled in these qualified profiles.

The same-input Radon Gemma speed search was native-kernel32.039s,O0 26.742s,compiled-O0 26.621s (ordinary request medians). Native kernels here still used the controlled180-frame input bridge. They are not the later official uint8/default-resize route. O0 was selected; its three-card24-case medians were21.430/21.885/26.742s. Compilation saved0.12s/request but added26.35s loading. These finite results do not establish an absolute speed limit.

Pause an active Gemma worker with `python -m classic_precision.gemma_nvfp4_continuation pause --profile EXACT_PROFILE.json`, after setting the controller's global pause marker when it could launch a successor. The current request is persisted before engine shutdown and original-lock release. CLI return alone is not closure proof. Reconcile real raw/intents/waits before any authorized resume; use a fresh profile/output and only proven never-attempted IDs. UNCERTAIN is never automatically retried. Actual safe pause and explicit GPU resume1 were demonstrated: each resumed card's first2 matched the newly bound source/CPU references, actual BF16 GPU input/position/projection hashes and full prompt. This finite validation does not guarantee every untested input/hardware combination.

## Append-only refinement

The next bounded extension covers both late transitions, using the same new grid for all four current models:

| Task | New levels | New videos | Four-model answers |
|---|---|---:|---:|
| T2 | ΔY .16/.17/.18/.19/.25/.30/.35 | 2,016 | 8,064 |
| T3 | Complete1.0–2.5° in0.1° steps;12 missing levels | 1,152 | 4,608 |
| Total | | 3,168 | 12,672 |

T2 retains three backgrounds, two durations,24 onsets and paired sides. T3 retains24 onsets, both tilt signs and both reference sides. The earlier proposed .225 increment was removed before any rendering/inference. Existing1.75° and2.25° answers remain; no old IDs or levels are removed or rerun. The regular1.0–2.5° grid reuses1.0/1.5/2.0/2.5° and adds1.1/1.2/1.3/1.4/1.6/1.7/1.8/1.9/2.1/2.2/2.3/2.4°. Together with existing0.5–1.0°, this gives0.1° steps over0.5–2.5°; retain1.75°/2.25° as additional measured points. Plot numeric angles on a numeric axis, never equally spaced category labels. These are finite curve refinements, not a measured precise JND.

```bash
python -B -m classic_precision.refinement plan \
  --index /data/classic-t1-t3-v0.3/dataset-index.json \
  --output /data/plans/classic-refinement
```

This produces `manifest.json`, `plan.json` and `video-names.csv`, with zero model calls. Source-generation manifests intentionally contain no model launch defaults. Bind separate finite per-model NVFP4 profiles after new-source acceptance. Historical Cosmos/Gemma BF16 defaults must not be used. Existing IDs, prompts, source bytes and answers remain immutable.

After rendering and auditing only the new sources, `refinement extend-index --base-index ... --new-index ... --output ...` validates an append-only12,528-video index. It rejects overlapping IDs, path collisions and missing source evidence. The `dataset` loader resolves stable IDs to readable names; inference results join on ID. Never publish unrendered entries with invented byte hashes.

The finite implementation uses `refinement_execution source --manifest ... --root ... --output ... --workers 8` for the new media and independent source audit. It reuses the existing RGB16 renderer, FFV1 encoder and all-frame audit without the old scheduler's fixed128-state assumption. `refinement_dispatch sources --root ROOT` binds the actual accepted media, appends the index and freezes the common four-model allocation; it does not upload or load models.

`refinement_dispatch seal --root ROOT --host fourier|radon` binds the existing qualified checkpoints and physical leases. Each card's finite `refinement_dispatch queue --root ROOT --card fourier0|fourier1|radon0` follows its frozen two-stage order: Qwen→Gemma on Fourier0, Gemma→Cosmos on Fourier1, and Nemo→Gemma on Radon0. Real worker waits and owned-engine exit precede the next stage. These commands use the experiment hosts' retained operator profiles; they are not portable machine provisioning. GPU acceptance is established by actual run receipts, not CPU tests.

For this refinement, use `refinement_worker pause --profile EXACT_PROFILE.json` for each active worker and each queued successor. The active request drains to durable raw before shutdown. A paused predecessor holds its queue. An errored worker leaves its claimed IDs held with no automatic retry; reconcile raw, technical passes, actual waits and the original lease before creating an explicitly authorized continuation. Four focused CPU checks cover the finite allocation, intact LR/sign units, request-boundary pause and retention of raw/claim after a post-output error.

## Code release and validation scope

This branch contains the current source adapters, finite workers, static controls, reader/naming tools and bounded refinement planner. Portable CPU dataset/naming/refinement tests and configuration/help entrypoints are checked from a clean source export. Tests that bind private historical operator receipts require those experiment files; they are not clean-machine smoke tests. The qualified CUDA SDK and actual GPU results remain tied to their original run profiles. A new machine still needs explicit model/data paths and physical-lease/profile binding; this source publication does not claim a fresh GPU reproduction on a clean machine.

The source dataset/read-only reader is public at the pinned v0.3 commit above and passed anonymous download acceptance. New refinement media and results have not been published. A reuse license remains unspecified. The poor/middle/good complete-cell GPU audit is a future check, not completed reproduction.

# Current stimulus and model protocol

The fixed input release is [DrQY/vlm-psychophysics at `13e9cf67`](https://huggingface.co/datasets/DrQY/vlm-psychophysics/tree/13e9cf67bb0b63be562ed7b7a48668fa7b7bca22). The [dataset index](https://huggingface.co/datasets/DrQY/vlm-psychophysics/blob/13e9cf67bb0b63be562ed7b7a48668fa7b7bca22/dataset-index.json) binds the video and static manifests. Source lineage, released media hashes, exact prompts, ground truth and ordered input paths are recorded per trial. Historical configurations and results remain in the [prior protocol](https://github.com/YangQiantwx/vlm-psychophysics/blob/e9be5fd206cd7d0307528b94f5192375e687a1f3/EXPERIMENT_PLAN.md) and [research report](report.md).

## Stimuli

All videos are 768 × 512, 180 frames at 60 fps. Classic T1–T4 use lossless sRGB16 FFV1/MKV. Natural T1–T5 and Classic T5 use lossless sRGB8 RGB H.264/MP4. Model decoder and processor receipts specify their own input dtype, spatial geometry, frame count and time grid.

- **T1.** Eight spatial frequencies, 24 phases and balanced horizontal directions at each modulation depth. The 0.5 extension contributes 384 trials per domain. Natural uses multiplicative modulation in linear RGB.
- **T2.** Local brightening during the 30-frame event. Natural uses a frozen 20-increment × 24-onset × two-side panel. Each of the 480 left/right pairs uses the same background and object. The four photo/generated background × object groups each contribute 240 trials, with 12 per increment and 10 per onset. All 200 backgrounds and 100 objects appear in each luminance-increment half. Use the [frozen combinations and renderer](https://huggingface.co/datasets/DrQY/vlm-psychophysics/blob/13e9cf67bb0b63be562ed7b7a48668fa7b7bca22/reproduction/README.md#natural-t2) and the [matched Classic pairs](https://huggingface.co/datasets/DrQY/vlm-psychophysics/blob/13e9cf67bb0b63be562ed7b7a48668fa7b7bca22/natural/t2/classic-natural-pairs.csv).
- **T3.** The objects appear for 30 frames. The current Natural grid pairs with the 2,736 Classic configurations. Left/right pairs keep the background, object, onset, angle and physical tilt direction fixed.
- **T4.** Classic crosses N = 8 or 16, eight nonzero coherence levels, 24 layouts and both directions. Each pair shares start positions. All trajectories stay contained and separated over the full motion interval. Natural crosses those same trajectories with same or distinct object appearances. Each appearance contributes 768 trials. The two appearances share a background and trajectory for every matched trial.
- **T5.** An orientation reference image precedes the video. Public inputs cover 192 unmasked trials. The original Cosmos evaluation includes an additional masked condition with access-controlled sources.

| Task | Ordered static input |
|---|---|
| T1 | Frames 0, 1, 2 and 3 |
| T2, T3 | One frame at onset + 15 |
| T4 | Frames 0 and 179 with the [endpoint prompt](reproduction/t4-static-protocol.json) |
| T5 | Reference image, then event midpoints for stimulus inspection |

The static evaluation uses RGB8 PNG inputs and the modality-specific prompts from the static manifests. Preserve the stated order and exact prompt, including whitespace. Report unique model calls and scientific aliases separately when identical ordered pixels, prompt and choices share a model input. Static T2 judges apparent brightness in a frame. T1 and T4 use ordered displacement cues.

## Execution routes and runtime contracts

The four-model evaluation uses the official mixed NVFP4 checkpoints for Cosmos, Qwen, Nemotron and Gemma, retaining each checkpoint's native BF16 components. The [T1–T3 video configuration](classic_precision/current_x_protocols.json) pins the model and tokenizer revisions, request adapters, engine parameters and generation settings. Video uses each model's accepted native 180-frame route. Qwen's QSA extension must apply inside the worker. Gemma's video budget is 280 soft tokens, with 176 effective tokens per frame and 16 bottom padding rows in the qualified T1–T3 input. Its static image budget is 140 and has separate processor evidence.

```bash
python -m classic_precision.reproduce show --model gemma --protocol current-x --task T1
```

| Scope | Entry point or worker |
|---|---|
| Classic T1–T3 video | [`official_x_runner_v2`](classic_precision/official_x_runner_v2.py), with [`official_x_runner_qsa`](classic_precision/official_x_runner_qsa.py) for Qwen |
| Natural T1–T3 video | [`natural_x_runner`](classic_precision/natural_x_runner.py), using the accepted per-model native routes |
| T1 m = 0.5 video, Classic T4 video | [`t1_m05_t4_classic`](classic_precision/t1_m05_t4_classic.py) |
| Classic/Natural T1–T3 static | [`static_x_plan`](classic_precision/static_x_plan.py), [`static_x_queue`](classic_precision/static_x_queue.py) and [`static_x_admission`](classic_precision/static_x_admission.py) |
| T1 m = 0.5 static | [`static_t1_m05`](classic_precision/static_t1_m05.py) |
| Classic T4 static | [`static_t4_strict_operations`](classic_precision/static_t4_strict_operations.py) and [`static_t4_strict_queue`](classic_precision/static_t4_strict_queue.py) |
| Natural T4 video and static | [`natural_t4_x`](classic_precision/natural_t4_x.py) |
| T5 reference image and video | [`cosmos_t45`](classic_precision/cosmos_t45.py) |

The worker entry points consume sealed local profiles, source and runtime receipts, and physical GPU leases. Set `PSYCHOPHYSICS_WORKSPACE` to the execution workspace and keep its checkout at `code/vlm-psychophysics`. Set `PSYCHOPHYSICS_QUEUE_OWNER` and `PSYCHOPHYSICS_PARENT_QUEUE_OWNER` to the IDs recorded in your own current and parent queue receipts when using those queue stages. Use matching checkpoint, tokenizer, processor, backend, GPU qualification and source evidence when sealing a new run. The configuration export and [profile extracts](reproduction/current-x-profiles.json) inspect settings without loading a model. The [extension source identities](reproduction/current-x-code.json) record the original and checkout SHA-256 values of workspace-bound modules.

Natural T4 video and static each completed 6,144 calls across four models and 512 condition cells. Both passed terminal and returned-file SHA-256 verification. The first static launch stopped at admission with zero calls. The [recovery module](classic_precision/natural_t4_static_recovery.py) binds the original video closure and the successful static re-seal. For a new static run, the default route scopes the endpoint validator to the verified Natural T4 static protocol and delegates prior qualification records to their original checker. It preserves the endpoint inputs, prompt and generation settings. The [profile extracts](reproduction/current-x-profiles.json) include both terminal digests.

## Reproduce the released stimuli

Use the [stimulus reproduction guide](https://huggingface.co/datasets/DrQY/vlm-psychophysics/blob/13e9cf67bb0b63be562ed7b7a48668fa7b7bca22/reproduction/README.md) for download instructions, frozen inputs and replay commands.

## Model API

Start with **T1 video** on the [pinned public release](https://huggingface.co/datasets/DrQY/vlm-psychophysics/tree/13e9cf67bb0b63be562ed7b7a48668fa7b7bca22). Use `--task t1-classic` for FFV1/MKV or `--task t1-natural` for RGB H.264/MP4. Each has 4,224 videos at 60 fps, including contrast 0.5. To resume a run sealed against an earlier dataset revision, use its original checkout and output directory.

### All-frame run

Use a deployment that accepts the original video and can report frame indices after preprocessing. Candidates for managed vLLM are [Qwen3.5-397B-A17B](https://huggingface.co/Qwen/Qwen3.5-397B-A17B#video-input) and [Qwen3-VL-235B-A22B-Instruct](https://github.com/QwenLM/Qwen3-VL#online-serving). The server must permit the [vLLM multimodal request arguments](https://docs.vllm.ai/en/latest/features/multimodal_inputs/) and have room for the full visual input. Install the CLI with `python -m pip install .`. Set `VLLM_BASE_URL` and `VLLM_API_KEY`, then send one Classic pilot.

```bash
model-api evaluate --provider vllm \
  --model Qwen/Qwen3.5-397B-A17B --task t1-classic \
  --out qwen397b-t1-classic --max-requests 1 --execute
```

For the NVIDIA internal gateway use `--provider nvidia-internal`, its exact account model ID and `NVIDIA_API_KEY`. The request uses the screenshot's Chat Completions endpoint. Confirm that the gateway forwards the full-frame controls to a compatible serving backend.

The runner asks the loader to keep original frames and disables processor sampling. To continue, rerun with `--frame-report server-report.json --max-requests 4224`. The JSON is a list of reports from the server logs. Each includes `endpoint`, the returned `model`, a pinned `deployment` ID, the pilot `request_sha256` from `attempts.jsonl`, and the arrays `decoded_indices` and `processor_indices`. Both arrays must contain every index from 0 to 179 in order. Use indices after the last sampling step and before the vision encoder. Each task and domain needs its own pilot. For Classic, record the effective bit depth and resize policy alongside the frame evidence. The script blocks the batch until its report is recorded.

Return `return.zip` and `return.zip.sha256`. Verify with `model-api verify --archive qwen397b-t1-classic/return.zip`. Requests send verified video bytes and the frozen prompt. The original filename, trial ID and ground truth stay local.

### Separate API comparisons

| Model | Prepared route | Sampling or input limit |
|---|---|---|
| Gemini 3.1 Pro | `model-api evaluate --provider gemini --model gemini-3.1-pro-preview --task t1-natural --sampled-video` | Requests 24 fps. Set `GEMINI_API_KEY`. |
| Qwen3.8-Max | `model-api evaluate --provider qwen --model qwen3.8-max --task t1-natural --sampled-video` | Requests 10 fps, the API limit and sampling boundary for the 5 Hz stimulus. Set `DASHSCOPE_API_KEY`. |
| Kimi K3.1 Pro | [Moonshot pilot](#additional-video-pilots) | MP4 via file ID. Service selects keyframes. |
| GLM-5V-Turbo | [BigModel pilot](#additional-video-pilots) | HTTPS video. Post-processor frame coverage unverified. |
| Astra | [Frame-image control](classic_precision/astra_direct_eval.py) | Public [OpenAI API](https://developers.openai.com/api/docs/models/gpt-6-astra) accepts text and images. |

### Additional video pilots

Fetch one pinned Natural T1 MP4 and its manifest.

```bash
python -m pip install '.[download]' openai requests
python -m classic_precision.current_release --scope natural/t1 --input video --limit 1 --output t1-data
python -m classic_precision.provider_api --provider moonshot --model kimi-k3.1-pro \
  --inputs t1-data/natural/t1/metadata.jsonl --root t1-data --out kimi-t1-pilot
```

Set `MOONSHOT_API_KEY` and add `--execute` for the live Kimi pilot. For GLM, use `--provider bigmodel --model glm-5v-turbo` and `ZHIPU_API_KEY`. Supply `--video-url` for an HTTPS copy named by its SHA-256. The helper checks the remote bytes before inference. Return the pilot output directory with its raw response and receipt.

[Gemini video API](https://ai.google.dev/gemini-api/docs/video-understanding) · [Qwen video API](https://help.aliyun.com/en/model-studio/vision) · [Kimi video API](https://platform.kimi.ai/docs/guide/use-kimi-vision-model) · [GLM-5V-Turbo API](https://docs.bigmodel.cn/cn/guide/models/vlm/glm-5v-turbo)

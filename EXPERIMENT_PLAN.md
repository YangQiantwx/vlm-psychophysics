# Stimulus and model protocol

The runner pins the [dataset revision and model recipes](configs/current-x.json). The [dataset index](https://huggingface.co/datasets/DrQY/vlm-psychophysics/blob/main/dataset-index.json) points to the paired video and static manifests. The [published answer tables](https://huggingface.co/datasets/DrQY/vlm-psychophysics/tree/main/results) bind completed runs to their inputs and settings.

## Stimuli

Every released video contains 180 frames at 60 fps and 768 × 512 pixels in lossless RGB8 H.264/MP4. Classic T1–T4 evaluation files store the exact RGB8 frames read from the original RGB16 FFV1/MKV masters by the Protocol X loader. The [master archive](https://huggingface.co/datasets/DrQY/vlm-psychophysics-rgb16-masters/tree/c01a232fa55fac3e1ccf3e20ba833baf81a0f0b8) preserves the original sample IDs, file hashes and execution records. The [equivalence report](https://huggingface.co/datasets/DrQY/vlm-psychophysics/blob/main/media-equivalence.json) binds the evaluation files to those masters.

| Task | Conditions | Ordered static input |
|---|---|---|
| T1 | Eight spatial frequencies, 24 phases and both directions. The m=0.5 extension contributes 384 trials per domain. Natural modulation is multiplicative in linear RGB. | Frames 0–3 |
| T2 | Thirty-frame brightening. Classic crosses four background luminances with 20 increments, 24 onsets and two sides. Natural uses the same increment, onset and side grid in each of four photo/generated source groups. | One frame at onset + 15 |
| T3 | Thirty-frame orientation comparison. The 2,736 Natural trials pair with the Classic grid. Left/right pairs hold the background, object, onset, angle and physical tilt direction fixed. | One frame at onset + 15 |
| T4 | Classic crosses 8 or 16 elements, eight nonzero coherence levels, 24 layouts and both directions. Classic includes circle and mixed-shape appearances. Natural uses the same trajectories with same or distinct object appearance. Trajectories stay contained and separated. | Frames 0 and 179 |
| T5 | Each domain contains 192 single-event and 192 dual-event videos. Each event lasts 30 frames. Dual trials add a 90° nonmatch before or after the fixed probe at SOAs of 30, 45 or 60 frames. The response is YES or NO. | Reference followed by each event midpoint in chronological order |

T5 video input is a centered 768 × 512 reference PNG followed by all 180 video frames. Static input is that reference followed by one or two event midpoint PNGs, also 768 × 512. The reference object has the same pixel size as the object in the scene.

Classic and Natural T2 each contain 3,840 trials: Classic has four background luminances, and Natural has four photo/generated background–object source groups, with 960 trials per group.

Each task's video manifest specifies the source conditions, prompt, truth and ordered static counterparts. The static manifest specifies the exact static prompt and image order. Use `sample_id` for scientific joins and the frozen static case identity for call accounting. Classic T2 has 3,840 static records across 160 distinct inputs. Classic T3 has 2,736 static records across 114 distinct inputs. Static PNGs use RGB8. Frame indices start at zero.

## Native execution

[Current X recipes](configs/current-x.json) pin official mixed NVFP4 checkpoints for Cosmos, Qwen, Nemotron and Gemma, including the higher-precision components shipped in each checkpoint. [Runtime lock](configs/runtime-lock.json) records the exact runtime files and package builds. Qwen uses the recorded X-QSA selector inside its worker. The model recipes bind native video preprocessing and ordered static inputs. Gemma's qualified current video and static recipes use a 280 soft-token budget with 176 effective tokens per frame or image.

Use the commands in the [repository README](README.md) for a bounded run. The runner records the selected dataset cases, configuration, input and runtime evidence, original answers, and counts. Each recipe checks the first actual request against its native CPU reference and GPU input, then checks prompt IDs and media ordering on subsequent requests. [Sealed historical profiles](https://github.com/YangQiantwx/vlm-psychophysics/blob/8dcff9eac165ca9c36ca19fed72470ce26cda96c/reproduction/current-x-profiles.json)

## Model API

Install the client with `python -m pip install -e '.[download]'`. Public manifests and media download on demand through the Hugging Face cache. Use `--data-root` for a dataset already downloaded with `run.py download`. The supported model IDs select their official API routes. Use `--provider` to select a specific serving endpoint or a legacy adapter explicitly. Keys come from the provider's environment variable. Hub authentication uses `hf auth login` or `HF_TOKEN`. [Hugging Face authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)

```bash
sample=t1_2379890eb33c9a254afd
python run.py download --task t1-classic --input video --sample-id "$sample" \
  --media-only --data-root benchmark-data --out download-t1

# Set OPENAI_API_KEY in the shell before running the Astra pilot.
python run.py api evaluate --model gpt-6-astra \
  --task t1-classic --input video --sample-id "$sample" --data-root benchmark-data \
  --out api-astra-ordered-t1 --max-requests 1 --reasoning-effort low --execute

# Set DASHSCOPE_API_KEY in the shell before running the Qwen pilot.
python run.py api evaluate --model qwen3.8-max-0902 \
  --task t1-classic --input video --sample-id "$sample" --data-root benchmark-data \
  --out api-qwen-t1 --max-requests 1 --temperature 0 --seed 0 --disable-thinking --execute

# Set ANTHROPIC_API_KEY in the shell before running the Claude pilot.
python run.py api evaluate --model claude-opus-5-5 \
  --task t1-classic --input video --sample-id "$sample" --data-root benchmark-data \
  --out api-claude-t1 --max-requests 1 --execute

# Set GEMINI_API_KEY in the shell before running the sampled Gemini comparison.
python run.py api evaluate --model gemini-4-argon --sampled-video \
  --task t1-classic --input video --sample-id "$sample" --data-root benchmark-data \
  --out api-gemini-t1-sampled --max-requests 1 --execute
```

To evaluate trial indices 20 through 39, preview the exact selection and then run a pilot. For a full task, use the same commands without the two index flags.

```bash
selection=(--task t1-classic --input video --from-index 20 --to-index 40 --out outputs/t1-range)
python run.py api plan "${selection[@]}"

# Set VLLM_BASE_URL and VLLM_API_KEY for the serving endpoint.
provider=(--provider vllm --model YOUR_SERVED_MODEL --temperature 0 --seed 0 --disable-thinking)
python run.py api evaluate "${selection[@]}" "${provider[@]}" --max-requests 1 --execute
python run.py api evaluate "${selection[@]}" "${provider[@]}" \
  --all-requests --concurrency 4 --frame-report server-report.json --execute
```

Indices use zero-based sorted `sample_id` order for one task and modality. The range is `[from-index, to-index)`. A missing start means zero and a missing stop means the task size. Static ranges select linked trials before deduplicating matching inputs. `--concurrency` accepts 1–16 and applies after admission. Pilots run serially. `--max-requests` caps new attempts including pilots, and `--all-requests` uses the selected unique-request count as its budget. The output directory pins the selection and records completed attempts for resume.

The client verifies the media SHA and preserves the published task prompt. The frame-list adapters decode the 180 RGB frames in order, check the 60 fps source timestamps and submit lossless PNGs. When the manifest supplies the Protocol X reader's RGB hash, the client checks it before submitting. T5 places its reference image first. Select `--input static` on a frame-list adapter for the published ordered stills and download that modality first.

- `astra` uses ordered Responses `input_image` items with `detail: original`. OpenAI documents up to 1,500 images and 512 MB per request. Use `--reasoning-effort low|medium|high|xhigh|max` and optionally `--reasoning-mode standard|pro`. [Model and modalities](https://developers.openai.com/api/docs/models/gpt-6-astra) · [Image inputs](https://developers.openai.com/api/docs/guides/images-vision?api-mode=responses)
- `claude` sends ordered PNG image blocks to Messages API version `2026-10-21` for `claude-opus-5-5` and `claude-fable-5-1`. The request uses `transformations.oversized_image: error` on each image. Claude pads to its visual patch boundary. The two models support up to 600 images, and the client checks the exact serialized body against the 128 MB Messages limit. The Classic T1 pilot body is 5,957,184 bytes. The Natural T4 example `natural-t4-strict-v1-n16-distinct-l00-c01-left` is 138,635,966 bytes and fails the inline size check. [Models](https://platform.claude.com/docs/en/models/overview) · [Vision and limits](https://platform.claude.com/docs/en/build-with-claude/vision) · [Resize policy](https://platform.claude.com/docs/en/build-with-claude/vision-coordinates) · [Request limit](https://platform.claude.com/docs/en/api/overview)
- `gemini` uses the original MP4 and requests 24 fps in the `generateContent` video metadata. The published field range is (0, 24] fps. This adapter handles one public MP4 per request and uses the published question directly. Choose `--sampled-video` and keep these results separate from all-frame runs. Gemini 4 Argon is a public model on this endpoint. [Gemini 4 Argon](https://aistudio.google.com/docs/latest-model?generate-content=true) · [Video metadata](https://aistudio.google.com/api-reference/generate-content#videometadata)
- `qwen-frames` uses one DashScope `type: video` part with 180 ordered PNG data URLs. Qwen3.8 accepts frame lists of 4–8,000 images. The Qwen3.8-Max example omits `fps` because Model Studio does not document frame-list timestamp support for that model. The client still checks the original 60 fps source timestamps. It rejects JSON bodies above 16 MiB on legacy DashScope domains or 64 MB on workspace domains before sending. Some Natural clips exceed both limits as lossless PNGs. Use Astra for those clips. [Frame-list input and limits](https://www.alibabacloud.com/help/en/model-studio/vision) · [Request parameters](https://www.alibabacloud.com/help/en/model-studio/qwen-api-via-openai-chat-completions)
- `vllm` submits the original MP4 with server controls for full decoding and disabled processor sampling. Set `VLLM_BASE_URL`, `VLLM_API_KEY` and the exact served `--model`. `nvidia-internal` uses `NVIDIA_API_KEY` and requires the gateway to forward the video controls. The gateway sets its generation defaults. `--temperature`, `--seed` and `--disable-thinking` are available for Qwen and compatible vLLM endpoints. [vLLM serving](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)

A previous OpenAI Responses request for the Classic T1 example received a completed `gpt-6-astra` response with 180 original-detail PNGs. The PNGs' ordered RGB hash matches the published MP4's reader hash for `t1_2379890eb33c9a254afd`. That request used an earlier frame-order note. Current Astra and Claude video requests prepend “The video is provided as 180 consecutive frames in chronological order at 60 fps.” The run stores the prefix in `config.json` and the full request text SHA-256 in `attempts.jsonl`. The source plan records the source prompt and its SHA-256. [Public frame-equivalence index](https://huggingface.co/datasets/DrQY/vlm-psychophysics/blob/1a1f9921d6428b9d4b2a95c77f7ba4255ed13ac3/media-equivalence.json)

An all-frame pilot stays `PARTIAL` with its raw answer recorded until evidence arrives from the provider or deployment operator. To qualify the pilot and start a batch, repeat the command with `--frame-report server-report.json` and a new request budget. The report is a JSON list of objects with `endpoint`, `model` (returned ID), `deployment` (pinned serving identity), `request_sha256` (from `attempts.jsonl`), and `decoded_indices` and `processor_indices` with integers 0 through 179 in order. Use processor indices after final sampling. If the server reports them, include `resize_policy` and `crop_policy` as strings and `bit_depth` as an integer. The return package keeps these fields. The gate is per task/domain/input profile. For Astra and Claude, the timing cue is text accompanying an image sequence. Qwen has a separate frame-list path. Protocol X uses native video timing.

Meta Muse Wonder documents a native video input with a sampling-rate field. Its published schema leaves the supported fps range and resize behavior unspecified, so it remains a candidate for a separate pilot. ByteDance's video path caps its requested rate at 5 fps and compresses frames for processing. [Meta video schema](https://dev.meta.ai/docs/api-reference/responses/schemas#input-video-content) · [ByteDance fps range](https://docs.volcengine.com/docs/82379/1494384?lang=en) · [BytePlus frame preprocessing](https://docs.byteplus.com/en/docs/modelark/video-understanding)

The output directory contains `results.csv`, `results.jsonl`, `config.json`, `attempts.jsonl` and `summary.json`. Return `return.zip` and `return.zip.sha256`. `python run.py api verify --archive PATH/return.zip` verifies the package and recomputes the labels and scores from the saved raw responses. Select the legacy DashScope sampled route explicitly with `--provider qwen --sampled-video`. [Historical Model API protocol](https://github.com/YangQiantwx/vlm-psychophysics/blob/8dcff9eac165ca9c36ca19fed72470ce26cda96c/EXPERIMENT_PLAN.md#model-api)

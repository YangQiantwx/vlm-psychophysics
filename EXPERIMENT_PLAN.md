# Stimulus and model protocol

The runner pins the [dataset revision and model recipes](configs/current-x.json). The [dataset index](https://huggingface.co/datasets/DrQY/vlm-psychophysics/blob/main/dataset-index.json) points to the paired video and static manifests. The [published answer tables](https://huggingface.co/datasets/DrQY/vlm-psychophysics/tree/main/results) bind completed runs to their inputs and settings.

## Stimuli

Every released video contains 180 frames at 60 fps and 768 × 512 pixels in lossless RGB8 H.264/MP4. Classic T1–T4 evaluation files store the exact RGB8 frames read from the original RGB16 FFV1/MKV masters by the Protocol X loader. The [master archive](https://huggingface.co/datasets/DrQY/vlm-psychophysics-rgb16-masters/tree/acd03f1772e4ee61fd3dcd180ab6cbfb580c9ce7) preserves the original sample IDs and file hashes. The [equivalence report](https://huggingface.co/datasets/DrQY/vlm-psychophysics/blob/main/media-equivalence.json) binds the evaluation files to those masters.

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

Install the client with `python -m pip install -e ".[download]"`. Download one published video from the pinned release and reuse the same `--data-root` for the API pilot. Omit `--sample-id` from both commands to select the whole task. `--max-requests` caps new attempts in each invocation, including pilots.

```bash
sample=t1_db543a401163261ed5b4
python run.py download --task t1-classic --input video --sample-id "$sample" \
  --media-only --data-root benchmark-data --out download-t1

# Set DASHSCOPE_API_KEY in the shell before running the Qwen pilot.
python run.py api evaluate --provider qwen-frames --model qwen3.8-max-0902 \
  --task t1-classic --input video --sample-id "$sample" --data-root benchmark-data \
  --out api-qwen-t1 --max-requests 1 --temperature 0 --seed 0 --disable-thinking --execute

# Set OPENAI_API_KEY in the shell before running the Astra pilot.
python run.py api evaluate --provider astra --model gpt-6-astra \
  --task t1-classic --input video --sample-id "$sample" --data-root benchmark-data \
  --out api-astra-t1 --max-requests 1 --reasoning-effort medium --execute
```

The client verifies the media SHA and uses the exact published prompt. Both frame-list adapters decode the 180 RGB frames in order, check the 60 fps source timestamps and submit lossless PNGs. When the manifest supplies the Protocol X reader's RGB hash, the client checks it before submitting. T5 places its reference image first. Select `--input static` for the published ordered stills and download that modality first.

- `qwen-frames` uses one DashScope `video` frame-list part. It requests 10 fps, the documented maximum. Source time remains 60 fps. The client rejects requests above the 16 MiB legacy domain or 64 MB workspace limit. A representative Natural T5 clip produced a 186 MB payload at lossless PNG compression. Use `DASHSCOPE_BASE_URL` for your official regional workspace endpoint when needed. [Frame and payload limits](https://www.alibabacloud.com/help/en/model-studio/vision) · [Request parameters](https://www.alibabacloud.com/help/en/model-studio/qwen-api-via-openai-chat-completions)
- `astra` uses ordered Responses `input_image` items with `detail: original`. The API documents up to 1,500 images and a 512 MB payload. Astra uses `--reasoning-effort low|medium|high|xhigh|max` and optionally `--reasoning-mode standard|pro`. [Image inputs](https://developers.openai.com/api/docs/guides/images-vision?api-mode=responses) · [Reasoning](https://developers.openai.com/api/docs/guides/reasoning)
- `vllm` submits the original MP4 with server controls for full decoding and disabled processor sampling. Set `VLLM_BASE_URL`, `VLLM_API_KEY` and the exact served `--model`. `nvidia-internal` uses `NVIDIA_API_KEY` and requires the gateway to forward those controls. `--temperature`, `--seed` and `--disable-thinking` are optional for Qwen and compatible vLLM templates. [vLLM serving](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)

An all-frame pilot stays `PARTIAL` with its raw answer recorded until server evidence arrives. To qualify the pilot and start a batch, repeat the command with `--frame-report server-report.json` and a new request budget. The report is a JSON list of objects with `endpoint`, `model` (returned ID), `deployment` (pinned serving identity), `request_sha256` (from `attempts.jsonl`), and `decoded_indices` and `processor_indices` with integers 0 through 179 in order. Use processor indices after its final sampling step and record the effective resize and bit-depth policy. The gate is per task/domain/input profile. Hosted pilots can establish what was sent and what the API returned. They require the same server evidence for all-frame scoring and batch execution. Qwen's frame-list timing and Astra's ordered-image input are distinct from Protocol X's native timeline.

The output directory contains `results.csv`, `results.jsonl`, `config.json`, `attempts.jsonl` and `summary.json`. Return `return.zip` and `return.zip.sha256`. `python run.py api verify --archive PATH/return.zip` verifies the package and recomputes the labels and scores from the saved raw responses. The `qwen` and `gemini` adapters remain available with `--sampled-video` for separate native-video comparisons. [Historical Model API protocol](https://github.com/YangQiantwx/vlm-psychophysics/blob/8dcff9eac165ca9c36ca19fed72470ce26cda96c/EXPERIMENT_PLAN.md#model-api)

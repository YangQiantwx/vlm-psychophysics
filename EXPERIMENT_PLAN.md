# Stimulus and model protocol

The runner pins the [dataset revision and model recipes](configs/current-x.json). The [dataset index](https://huggingface.co/datasets/DrQY/vlm-psychophysics/blob/main/dataset-index.json) points to the paired video and static manifests. The [published answer tables](https://huggingface.co/datasets/DrQY/vlm-psychophysics/tree/main/results/current-x) bind earlier completed runs to their original inputs and settings.

## Stimuli

Every video contains 180 frames at 60 fps and 768 × 512 pixels. Classic T1–T4 use lossless sRGB RGB16 FFV1/MKV. Natural T1–T5 and Classic T5 use lossless sRGB RGB8 H.264/MP4.

| Task | Conditions | Ordered static input |
|---|---|---|
| T1 | Eight spatial frequencies, 24 phases and both directions. The m=0.5 extension contributes 384 trials per domain. Natural modulation is multiplicative in linear RGB. | Frames 0–3 |
| T2 | Thirty-frame brightening. Classic crosses four background luminances with 20 increments, 24 onsets and two sides. Natural uses the same increment, onset and side grid in each of four photo/generated source groups. | One frame at onset + 15 |
| T3 | Thirty-frame orientation comparison. The 2,736 Natural trials pair with the Classic grid. Left/right pairs hold the background, object, onset, angle and physical tilt direction fixed. | One frame at onset + 15 |
| T4 | Classic crosses 8 or 16 elements, eight nonzero coherence levels, 24 layouts and both directions. Natural uses the same trajectories with same or distinct object appearance. Trajectories stay contained and separated. | Frames 0 and 179 |
| T5 | Paired-v1 contains 192 single-event and 192 dual-event videos per domain. Each event lasts 30 frames. Dual trials add a 90° nonmatch before or after the fixed probe at SOAs of 30, 45 or 60 frames. The response is YES or NO. | Reference followed by each event midpoint for inspection |

Paired T5 model inputs use the original reference PNG, then all 180 video frames. The earlier 96 unmasked videos per domain retain their original manifests and reproduction entry points.

The full Natural T2 panel contains 3,840 trials. The [frozen 960-trial selection](https://huggingface.co/datasets/DrQY/vlm-psychophysics/blob/main/natural/t2/selection.json) includes 240 trials per source group and retains its own Classic pairing table.

Each task's video manifest specifies the source conditions, prompt, truth and ordered static counterparts. The static manifest specifies the exact static prompt and image order. Use `sample_id` for scientific joins and the frozen static case identity for call accounting. Classic T2 has 3,840 static records across 160 distinct inputs. Classic T3 has 2,736 static records across 114 distinct inputs. Static PNGs use RGB8. Frame indices start at zero.

## Native execution

[Current X recipes](configs/current-x.json) pin official mixed NVFP4 checkpoints for Cosmos, Qwen, Nemotron and Gemma, including the higher-precision components shipped in each checkpoint. [Runtime lock](configs/runtime-lock.json) records the exact runtime files and package builds. Qwen uses the recorded X-QSA selector inside its worker. The model recipes bind native video preprocessing and ordered static inputs. Gemma's qualified current video and static recipes use a 280 soft-token budget with 176 effective tokens per frame or image.

Use the commands in the [repository README](README.md) for a bounded run. The runner records the selected dataset cases, configuration, input and runtime evidence, original answers, and counts. Each recipe checks the first actual request against its native CPU reference and GPU input, then checks prompt IDs and media ordering on subsequent requests. [Sealed historical profiles](https://github.com/YangQiantwx/vlm-psychophysics/blob/8dcff9eac165ca9c36ca19fed72470ce26cda96c/reproduction/current-x-profiles.json)

## Model API

Use an endpoint that accepts the original video and can report post-processor frame indices. Set `VLLM_BASE_URL` and `VLLM_API_KEY`, then send one pilot.

```bash
model-api evaluate --provider vllm \
  --model Qwen/Qwen3.5-397B-A17B --task t1-classic \
  --out qwen397b-t1-classic --max-requests 1 --execute
```

The NVIDIA internal gateway uses `--provider nvidia-internal`, `NVIDIA_API_KEY` and the exact model ID for the account. The gateway must forward the full-frame controls to the serving backend.

For an all-frame batch, pass `--frame-report server-report.json` and the authorized `--max-requests`. Each report identifies the endpoint, returned model, pinned deployment and pilot `request_sha256` from `attempts.jsonl`, and gives `decoded_indices` and `processor_indices` containing every index from 0 to 179 in order. Use indices after the last sampling step. Record effective bit depth and resize policy for Classic. The runner checks the task-specific pilot before proceeding.

Return `return.zip` and `return.zip.sha256`. Run `model-api verify --archive PATH/return.zip` to verify the package. The [previous Model API protocol](https://github.com/YangQiantwx/vlm-psychophysics/blob/8dcff9eac165ca9c36ca19fed72470ce26cda96c/EXPERIMENT_PLAN.md#model-api) records the separate sampled-video and frame-image comparisons.

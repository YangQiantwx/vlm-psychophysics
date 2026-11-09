# Visual Psychophysics

Five controlled visual judgments in Classic and Natural scenes. The dataset contains 25,440 lossless MP4 videos and 25,440 linked static records. [Dataset](https://huggingface.co/datasets/DrQY/vlm-psychophysics) · [Stimulus viewer](https://vision-tasks.pages.dev/library/) · [Model results](https://vision-tasks.pages.dev/experiments/)

## Download the dataset

Install the Hub client with `python -m pip install huggingface_hub`. Download the complete dataset, including metadata, results and source assets.

```bash
hf download DrQY/vlm-psychophysics --repo-type dataset --local-dir ./dataset
```

For one task and domain, download its folder and the dataset index.

```bash
hf download DrQY/vlm-psychophysics --repo-type dataset \
  --include "dataset-index.json" "natural/t5/*" --local-dir ./dataset
```

Add `--revision` with the dataset commit in [configs/models.json](configs/models.json) to pin either download. For one trial by its video filename, run these commands from this repository's root.

```bash
python -m pip install -e '.[download]'
python run.py download --media-only --task t5-classic --input video \
  --sample T5_single_yes_onset006_left.mp4 --data-root ./dataset
```

This downloads the MP4 and its T5 reference PNG and verifies their SHA-256 hashes against the pinned release. Use `--input static` to download the reference and event images. T5 static folders share their video's filename stem and contain `reference.png` and one or two `frameNNN.png` images.

`--sample` works on download, API and local runs. It accepts an exact video filename, a dataset-relative video path or a stable trial ID. Include `--task` when the same filename appears in both domains. Omit `--sample` to select the whole task.

## Model API quickstart

API calls use the published prompt and media. The client fetches and verifies the selected inputs on demand. Set `--data-root ./dataset` to reuse local files and cache missing inputs there. Each run writes `results.csv`, `results.jsonl`, `summary.json`, `attempts.jsonl` and `return.zip` to its output directory.

### OpenAI GPT-6 Astra

Set `OPENAI_API_KEY` in your shell. Evaluate one T5 trial with its ordered static images.

```bash
python run.py api evaluate --model gpt-6-astra --task t5-classic --input static \
  --sample T5_single_yes_onset006_left.mp4 --out outputs/t5-astra-one \
  --max-requests 1 --reasoning-effort low --execute
```

Run a complete static task by omitting the selector and using `--all-requests`.

```bash
python run.py api evaluate --model gpt-6-astra --task t5-natural --input static \
  --out outputs/t5-natural-astra --all-requests --reasoning-effort low --execute
```

With `--input video`, Astra receives the T5 reference followed by 180 lossless PNGs in chronological order at `detail: original`. The request includes a 60 fps timing note and the published prompt. An all-frame video pilot saves the raw answer, then reports `PARTIAL` (exit code 2) while awaiting a server frame report for scoring and batch execution. [OpenAI SDK example](EXPERIMENT_PLAN.md#openai-ordered-frames) · [Frame report format](EXPERIMENT_PLAN.md#model-api)

### NVIDIA Inference Gateway

Set `NVIDIA_API_KEY` and `NVIDIA_VIDEO_MODEL_ID` for a video-capable model on your gateway account.

```bash
python run.py api evaluate --preset nvidia-repeat --task t1-classic \
  --sample T1_left_sf8_contrast1_trial01.mp4 --out outputs/t1-nvidia-one \
  --max-requests 1 --execute
```

The preset sends the original MP4 and prompt with `temperature=0`, `top_p=1`, `seed=0`, and one answer per trial. [NVIDIA SDK example](EXPERIMENT_PLAN.md#nvidia-native-video)

### Other hosted models

The API client also supports DashScope Qwen, Claude, Gemini and compatible vLLM endpoints. The [API guide](EXPERIMENT_PLAN.md#model-api) gives their keys, model IDs, input formats and size limits. Use `api plan` to preview a selection before calling a model. For a range of trials, use `--from-index 20 --to-index 40` with one task and input mode. The indices are zero-based in sample ID order, and the stop is exclusive. Repeating the same run in the same output directory resumes the selection.

```bash
python run.py api verify --archive outputs/t5-astra-one/return.zip
```

## Tasks

| Task | Judgment | Classic | Natural |
|---|---|---:|---:|
| T1 | Horizontal stripe motion | 4,224 | 4,224 |
| T2 | Side that briefly brightens | 3,840 | 3,840 |
| T3 | Side with the vertical object | 2,736 | 2,736 |
| T4 | Horizontal motion signal | 1,536 | 1,536 |
| T5 | Orientation match to a reference | 384 | 384 |

The counts are trials. Static evaluation shares a call when ordered media, prompt and choices match. T5 has 192 single-event and 192 dual-event trials per domain. Its centered 768 × 512 references preserve the object's pixel size. The 768 videos correspond to 425 independent static inputs, 44 Classic and 381 Natural.

## Run the local models

Use the pinned Linux x86-64 environment with Python 3.12, RTX PRO 6000 Blackwell GPUs and CUDA Toolkit 13.0.3. Recipes that compile CUDA kernels also require Ninja. Install into your existing Python environment from the repository root.

```bash
python -m pip install -e '.[download,inference]'
```

Run all video and static trials in one task on one GPU.

```bash
CUDA_VISIBLE_DEVICES=0 python run.py --task t5-classic --model qwen --out outputs/t5-classic-qwen
```

Use `--task t5-natural` and a fresh output directory for Natural. Use `--sample T5_single_yes_onset006_left.mp4` for a single trial, `--input video` or `--input static` for one modality, or `--model all` to run all four checkpoint recipes. Omit `CUDA_VISIBLE_DEVICES` to select among compatible idle GPUs. Each GPU runs one recipe at a time.

The runner verifies the installed runtime and downloads the pinned inputs and checkpoint. It reuses `./dataset` when it contains the matching index on SSD, or selects a writable SSD cache. Use `--data-root`, `--model-root` and `--out` to choose locations. Without `--out`, it uses `outputs/` on SSD or puts the output folder under the data cache. Set `VLM_PSYCHOPHYSICS_CUDA_HOME` for a nonstandard CUDA install. Authenticate with Hugging Face for gated weights and set `NGC_API_KEY` for the NVIDIA checkpoint.

Use `run.py plan` to check the selected cases and `run.py doctor` to check the runtime. Repeat the same run to resume its untouched calls. Outputs include `results.csv`, `results.jsonl`, `summary.json`, `curves.png` and `return.zip`. The `evidence/` folder records input, source and runtime hashes, prompts, ground truth, settings and execution records.

```bash
python run.py report --out outputs/t5-classic-qwen
python run.py verify outputs/t5-classic-qwen/return.zip
```

T5 video input is the reference followed by the 180 video frames. Static input is the reference followed by one or two event midpoint images in chronological order. The four published models used 4,772 independent calls across both T5 domains and both modalities. [Input definitions for all five tasks](EXPERIMENT_PLAN.md#stimuli)

## RGB16 masters and MP4 equivalence

Each Classic T1–T4 trial keeps its `sample_id` and filename stem. Task metadata and the [per-task result folders](https://huggingface.co/datasets/DrQY/vlm-psychophysics/tree/901ab8c3065f3d21acb870c74a4cce76f0908b20/results) provide the released MP4 path and SHA-256. The [RGB16 FFV1 archive](https://huggingface.co/datasets/DrQY/vlm-psychophysics-rgb16-masters/tree/c01a232fa55fac3e1ccf3e20ba833baf81a0f0b8) preserves the masters and original execution records.

Published results use `results/t1/` through `results/t5/`. Each folder contains `results.csv.gz` and `counts.csv`, with the same columns for domain, model, modality, sample ID, ground truth and prediction. T5 also includes `pairs.csv.gz` for matched single/dual comparisons.

The [verification index](https://huggingface.co/datasets/DrQY/vlm-psychophysics/blob/901ab8c3065f3d21acb870c74a4cce76f0908b20/media-equivalence.json) binds the two encoded files for each of 12,336 Classic trials. It records exact equality of all 2,220,480 frames at the evaluation RGB8 reader boundary, including frame order and reader metadata. The archive retains the original 16-bit values.

```bash
video=classic/t1/videos/T1_left_sf8_contrast1_trial01
hf download DrQY/vlm-psychophysics-rgb16-masters "$video.mkv" --repo-type dataset \
  --revision c01a232fa55fac3e1ccf3e20ba833baf81a0f0b8 --local-dir ./masters
python run.py download --task t1-classic --input video --sample T1_left_sf8_contrast1_trial01.mp4 \
  --media-only --data-root ./dataset
hf download DrQY/vlm-psychophysics media-equivalence.jsonl --repo-type dataset \
  --revision 901ab8c3065f3d21acb870c74a4cce76f0908b20 --local-dir ./dataset
python run.py verify-media --master-root ./masters --data-root ./dataset \
  --sample-id t1_2379890eb33c9a254afd
```

Run this in the local inference environment. The verifier checks file hashes, every decoded RGB8 frame and its timestamp. To check the full archive, download both sets of videos and omit `--sample-id`.

## Regenerate a stimulus

The videos are rendered at 768 × 512. Natural backgrounds are fitted to this canvas before the objects are placed. The evaluation loader reads the full canvas at all 180 time points. Gemma adds 16 rows at the bottom for its processor.

```bash
python run.py generate --task t2-classic --sample-id t2_000201cb3983b82df70a --out replay
```

Use `python run.py generate --task all` to regenerate all released T1–T5 videos. Repeat `--sample-id` to select several trials. Classic T1–T4 rendering first reconstructs the RGB16 master, then produces the verified lossless RGB8 MP4. The renderer checks decoded pixels and timestamps and reports encoded container equality separately. Set `--data-root` to reuse local sources. [Source files and attribution](https://huggingface.co/datasets/DrQY/vlm-psychophysics/tree/901ab8c3065f3d21acb870c74a4cce76f0908b20/sources)

The released input manifests contain the exact prompt, media order and hashes. Send only the prompt and media to a model. Filenames, labels and provenance stay in the evaluation record. [Per-trial answers, prompts and counts](https://huggingface.co/datasets/DrQY/vlm-psychophysics/tree/901ab8c3065f3d21acb870c74a4cce76f0908b20/results)

## Code layout

`run.py` is the entry for evaluation, generation, hosted APIs and result verification. `python -m vlm_psychophysics` uses the same dispatcher. `generate.py` and the installed `model-api` command remain small compatibility entries.

| Location | Responsibility |
|---|---|
| `configs/` | Pinned model settings, runtime versions and download hashes |
| `vlm_psychophysics/release.py`, `catalog.py`, `data.py` | Dataset revision, manifest validation, IDs and call deduplication |
| `vlm_psychophysics/generators/`, `stimuli.py`, `media.py` | Source rendering, pixel checks and video reconstruction |
| `vlm_psychophysics/inputs.py`, `models.py`, `observers.py`, `runtime/` | Ordered media, model loading and actual-input checks |
| `vlm_psychophysics/assets.py`, `runner.py` | SSD selection, downloads, GPU assignment and durable resume |
| `vlm_psychophysics/results.py`, `api/` | Exact-label scoring, plots, return packages and hosted endpoints |
| `tests/` | Input, resume, API, scoring and numerical regressions |

```bash
python run.py generate --task t1-classic --sample-id SAMPLE_ID
python run.py api evaluate --provider vllm --model MODEL --task t1-classic --out api-run --max-requests 1 --execute
python -m pytest -q
```

Scientific render functions are retained from the execution source; the portable runner supplies storage, scheduling and return packaging. Each run records its own runner source hashes in `evidence/source-code.json`. The source profile hashes in [configs/models.json](configs/models.json) identify the published execution profiles. The code and evidence from earlier runs are available at the [historical source revision](https://github.com/YangQiantwx/vlm-psychophysics/tree/8dcff9eac165ca9c36ca19fed72470ce26cda96c). Website and annotation code are maintained separately in [Vision Tasks](https://github.com/YangQiantwx/vision-tasks).

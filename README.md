# Visual Psychophysics

Five controlled visual judgments in Classic and Natural scenes. The release contains 25,440 MP4 videos and 25,440 corresponding static evaluation records. [Dataset](https://huggingface.co/datasets/DrQY/vlm-psychophysics) · [Stimulus viewer](https://vision-tasks.pages.dev/library/) · [Model results](https://vision-tasks.pages.dev/experiments/)

| Task | Judgment | Classic | Natural |
|---|---|---:|---:|
| T1 | Horizontal stripe motion | 4,224 | 4,224 |
| T2 | Side that briefly brightens | 3,840 | 3,840 |
| T3 | Side with the vertical object | 2,736 | 2,736 |
| T4 | Horizontal motion signal | 1,536 | 1,536 |
| T5 | Orientation match to a reference | 384 | 384 |

The counts are trials. Static evaluation shares a call when ordered media, prompt and choices match. T5 has 192 single-event and 192 dual-event trials per domain, using centered 768 × 512 reference images that preserve the object's pixel size. Its 768 videos correspond to 425 unique static inputs.

## Run

Clone this repository and run the commands from its root. Use Linux, Python 3.12, RTX PRO 6000 Blackwell GPUs and CUDA Toolkit 13.0.3. Install into your existing environment:

```bash
python -m pip install -e ".[download,inference]"
python run.py --task t1-classic --model qwen
```

The runner selects writable SSD storage, downloads the pinned dataset and model, checks the runtime, and assigns work by available GPU memory. Each GPU loads and finishes one model recipe at a time. Set `VLM_PSYCHOPHYSICS_CUDA_HOME` if the CUDA toolkit is outside the standard location. Authenticate with Hugging Face for gated models and set `NGC_API_KEY` for NVIDIA downloads.

```bash
# All four models, T1–T5, both domains, video and static
python run.py

# One published video; a local video path also works
python run.py --task t1-classic --model qwen --video T1_left_sf0.5_contrast0.05_trial01.mp4

# Same-size T5 reference, both domains, all four models, video and static
python run.py --task t5-classic --task t5-natural

# Rebuild plots and verify a returned result
python run.py report --out outputs/RUN
python run.py verify outputs/RUN/return.zip
```

Each output directory contains `results.csv`, `results.jsonl`, `summary.json`, `curves.png` and `return.zip`. The `evidence/` folder holds exact prompts, ground truth, resolved model settings, source and input hashes, and execution records. Return `return.zip` for verification and rescoring. Repeating a command with the same output resumes untouched inputs and preserves recorded attempts.

Static inputs use one shared ordered-image path: T1 has frames 0–3; T2/T3 use the event midpoint (`onset + 15`); T4 uses frames 0 and 179. T5 places the reference first, followed by one or two event midpoint images in chronological order. All T5 images are 768 × 512. Video input is the reference followed by all 180 frames. The frozen video and static prompts are preserved byte for byte. Four models evaluate 4,772 independent inputs across both T5 domains; single and dual cohorts retain their respective image limits of 2 and 3.

Use `--input video` or `--input static` to select a modality, `--sample-id` for a scientific ID, or repeat `--task` for several scopes. `--config configs/current-x.json` selects the complete model recipe. Optional `--data-root`, `--model-root`, `--out` and `--gpus` select storage and GPU UUIDs. `run.py plan` shows the selection, `run.py download` fetches it, and `run.py doctor` checks the environment. Add `--media-only` to download data without weights, or `--offline` to use existing local files.

## Download and use a Model API

Install the download tools in an existing Python environment. A hosted evaluation does not need local model weights.

```bash
python -m pip install -e ".[download]"
python run.py download --media-only --data-root ./dataset
```

To start with a single published video, add `--task t1-classic --input video --sample-id t1_db543a401163261ed5b4` to the download command. Use the same selection when you prepare the API run.

```bash
python run.py api plan --task t1-classic --input video \
  --sample-id t1_db543a401163261ed5b4 --data-root ./dataset --out api-run
```

[Hosted commands and return packages](EXPERIMENT_PLAN.md#model-api) cover model selection, the Qwen ordered-frame video route, Astra, vLLM and the service-side frame check. Results, attempts and resolved parameters are saved in the run directory.

## RGB16 masters and MP4 equivalence

Each Classic T1–T4 trial keeps its `sample_id` and filename stem. The task metadata points to its current MP4 and contains a separate `master_video` descriptor. Download the [RGB16 FFV1 masters](https://huggingface.co/datasets/DrQY/vlm-psychophysics-rgb16-masters/tree/acd03f1772e4ee61fd3dcd180ab6cbfb580c9ce7) for source precision analysis. Original answer records retain their execution-time media SHA-256.

The [verification index](https://huggingface.co/datasets/DrQY/vlm-psychophysics/blob/main/media-equivalence.json) binds the two encoded files for each of 12,336 Classic trials. It records exact equality of all 2,220,480 frames at the Protocol X RGB8 reader boundary, including frame order and reader metadata. The archive retains the original 16-bit values.

```bash
hf download DrQY/vlm-psychophysics-rgb16-masters --repo-type dataset \
  --revision acd03f1772e4ee61fd3dcd180ab6cbfb580c9ce7 --local-dir ./masters
hf download DrQY/vlm-psychophysics media-equivalence.jsonl --repo-type dataset \
  --revision d5ca80fade6bcc12386c10b187daca24ee6a41dd --local-dir ./dataset
python run.py verify-media --master-root ./masters --data-root ./dataset --workers 8
```

The verifier uses the pinned Protocol X vLLM/OpenCV reader from the inference installation and checks file hashes, arrays, frame order and timestamps. Add `--sample-id` to verify a selected trial.

## Regenerate a stimulus

```bash
python run.py generate --task t2-classic --sample-id t2_000201cb3983b82df70a --out replay
```

Use `python run.py generate --task all` to regenerate all released T1–T5 videos. Repeat `--sample-id` to select several trials. Classic T1–T4 rendering first reconstructs the RGB16 master, then produces the verified lossless RGB8 MP4. The renderer checks decoded pixels and timestamps and reports encoded container equality separately. Set `--data-root` to reuse local sources. [Source files and attribution](https://huggingface.co/datasets/DrQY/vlm-psychophysics/tree/main/sources)

The released input manifests contain the exact prompt, media order and hashes. Send only the prompt and media to a model. Filenames, labels and provenance stay in the evaluation record. [Per-trial answers, prompts and counts](https://huggingface.co/datasets/DrQY/vlm-psychophysics/tree/main/results)

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

Scientific render functions are retained from the execution source; the portable runner supplies storage, scheduling and return packaging. Each run records the source it actually used. Original workstation scripts and sealed execution identities remain at the [historical source revision](https://github.com/YangQiantwx/vlm-psychophysics/tree/8dcff9eac165ca9c36ca19fed72470ce26cda96c). Website and annotation code are maintained separately in [Vision Tasks](https://github.com/YangQiantwx/vision-tasks).

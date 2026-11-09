# Visual Psychophysics

Five controlled visual judgments in Classic and Natural scenes. The release contains 25,440 MP4 videos and 25,440 corresponding static evaluation records. [Dataset](https://huggingface.co/datasets/DrQY/vlm-psychophysics) · [Stimulus viewer](https://vision-tasks.pages.dev/library/) · [Model results](https://vision-tasks.pages.dev/experiments/)

## Model API quickstart

Clone this repository and run the commands from its root. Start with one Classic T1 video. For the models below, `--model` selects the provider, media format and prompt handling. The client downloads the pinned public media as needed and checks its SHA-256.

```bash
python -m pip install -e '.[download]'
```

### OpenAI GPT-6 Astra

Set `OPENAI_API_KEY` in your shell or secret store before running the pilot.

```bash
python run.py api evaluate --model gpt-6-astra \
  --task t1-classic --input video --sample-id t1_2379890eb33c9a254afd \
  --out outputs/t1-astra-ordered-one --max-requests 1 --reasoning-effort low --execute
```

The Responses API receives 180 ordered lossless PNG images at `detail: original`. The request text starts with “The video is provided as 180 consecutive frames in chronological order at 60 fps.” The published task prompt follows verbatim. [GPT-6 Astra](https://developers.openai.com/api/docs/models/gpt-6-astra) · [Responses image inputs](https://developers.openai.com/api/docs/guides/images-vision?api-mode=responses)

### NVIDIA Inference Gateway

Set `NVIDIA_API_KEY` and `NVIDIA_VIDEO_MODEL_ID` for a video-capable model on your gateway account.

```bash
python run.py api evaluate --preset nvidia-repeat \
  --task t1-classic --sample-id t1_2379890eb33c9a254afd \
  --out outputs/t1-nvidia-one --max-requests 1 --execute
```

The preset sends the published MP4 and prompt with `temperature=0`, `top_p=1`, `seed=0`, and one answer per trial. Send back `return.zip` and `return.zip.sha256`. [Repeat a trial, select a range, or request all frames](EXPERIMENT_PLAN.md#model-api)

### Qwen3.8-Max

Set `DASHSCOPE_API_KEY` and, for a workspace-specific endpoint, `DASHSCOPE_BASE_URL`. The Qwen pilot uses the same Classic trial.

```bash
python run.py api evaluate --model qwen3.8-max-0902 \
  --task t1-classic --input video --sample-id t1_2379890eb33c9a254afd \
  --out outputs/t1-qwen-one --max-requests 1 --temperature 0 --seed 0 \
  --disable-thinking --execute
```

DashScope receives the 180 PNGs as one ordered `type: video` frame list. Use Astra for clips whose lossless request exceeds DashScope's size limit. [DashScope video inputs](https://www.alibabacloud.com/help/en/model-studio/vision) · [Regional endpoints](https://www.alibabacloud.com/help/en/model-studio/regions/)

### Other hosted models

Use the same task, sample ID and request budget. Choose a fresh `--out` and supply the selected model's API key. `--reasoning-effort` is specific to the Astra example.

| Model ID | Key environment variable | Video delivery |
|---|---|---|
| `claude-opus-5-5`, `claude-fable-5-1` | `ANTHROPIC_API_KEY` | 180 ordered lossless PNGs with the same chronology note. Claude rejects images it would downscale. The inline request limit is 128 MB. |
| `gemini-4-argon`, `gemini-3.8-flash` | `GEMINI_API_KEY` | Original MP4 with server sampling requested at 24 fps. Add `--sampled-video` for this separate T1–T4 comparison. |

[Claude image inputs and limits](https://platform.claude.com/docs/en/build-with-claude/vision) · [Claude resize setting](https://platform.claude.com/docs/en/build-with-claude/vision-coordinates#turn-resizing-into-an-error-with-transformations) · [Gemini video sampling](https://aistudio.google.com/docs/video-understanding?generate-content=true)

| Scope | Arguments |
|---|---|
| One trial | `--sample-id ID --max-requests 1` |
| Index range | `--from-index 20 --to-index 40 --all-requests` |
| Entire task | `--task t1-classic --all-requests` |
| Parallel calls | `--concurrency 4` |

Indices are zero-based in sorted `sample_id` order within one task and modality. `--to-index` is exclusive. `api plan` previews the selection and, when supplied, the model route without calling the model. Static ranges select video associations first, then share identical image requests. Use a separate `--out` per model and selection. The same output directory resumes the same selection.

If your Anthropic API key is organization-wide, set `ANTHROPIC_WORKSPACE_ID` to the workspace that should handle the request.

Public dataset access works anonymously. Use `hf auth login` for interactive Hub authentication or inject `HF_TOKEN` as a CI or Space secret.

A successful full-frame video pilot records the raw answer and client-side delivery in `results.csv` and `attempts.jsonl`. It reports `PARTIAL` (exit code 2) while it awaits a frame report from the serving deployment. All-frame scoring and batch concurrency require that evidence. Verify the saved results with `python run.py api verify --archive outputs/t1-astra-ordered-one/return.zip`. [Range and batch examples](EXPERIMENT_PLAN.md#model-api) · [Hugging Face authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)

To cache a whole task before evaluating, run:

```bash
python run.py download --task t1-classic --input video --media-only --data-root ./dataset
```

Pass the same `--data-root ./dataset` to the API client to use that cache. Omit `--task` and `--input` from the download command for the complete dataset.

## Tasks

| Task | Judgment | Classic | Natural |
|---|---|---:|---:|
| T1 | Horizontal stripe motion | 4,224 | 4,224 |
| T2 | Side that briefly brightens | 3,840 | 3,840 |
| T3 | Side with the vertical object | 2,736 | 2,736 |
| T4 | Horizontal motion signal | 1,536 | 1,536 |
| T5 | Orientation match to a reference | 384 | 384 |

The counts are trials. Static evaluation shares a call when ordered media, prompt and choices match. T5 has 192 single-event and 192 dual-event trials per domain, using centered 768 × 512 reference images that preserve the object's pixel size. Its 768 videos correspond to 425 unique static inputs.

## Run the local Protocol X models

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

## RGB16 masters and MP4 equivalence

Each Classic T1–T4 trial keeps its `sample_id` and filename stem. Task metadata and the [per-trial result tables](https://huggingface.co/datasets/DrQY/vlm-psychophysics/tree/main/results) provide the current MP4 path and SHA-256. The [RGB16 FFV1 archive](https://huggingface.co/datasets/DrQY/vlm-psychophysics-rgb16-masters/tree/c01a232fa55fac3e1ccf3e20ba833baf81a0f0b8) preserves the masters and original execution records.

The [verification index](https://huggingface.co/datasets/DrQY/vlm-psychophysics/blob/main/media-equivalence.json) binds the two encoded files for each of 12,336 Classic trials. It records exact equality of all 2,220,480 frames at the Protocol X RGB8 reader boundary, including frame order and reader metadata. The archive retains the original 16-bit values.

```bash
video=classic/t1/videos/T1_left_sf8_contrast1_trial01
hf download DrQY/vlm-psychophysics-rgb16-masters "$video.mkv" --repo-type dataset \
  --revision c01a232fa55fac3e1ccf3e20ba833baf81a0f0b8 --local-dir ./masters
python run.py download --task t1-classic --input video --sample-id t1_2379890eb33c9a254afd \
  --media-only --data-root ./dataset
hf download DrQY/vlm-psychophysics media-equivalence.jsonl --repo-type dataset \
  --revision 1a1f9921d6428b9d4b2a95c77f7ba4255ed13ac3 --local-dir ./dataset
python run.py verify-media --master-root ./masters --data-root ./dataset \
  --sample-id t1_2379890eb33c9a254afd
```

Run this in the local inference environment. The verifier checks file hashes, every decoded RGB8 frame and its timestamp. To check the full archive, download both sets of videos and omit `--sample-id`.

## Regenerate a stimulus

The videos are rendered at 768 × 512. Natural backgrounds are fitted to this canvas before the objects are placed. Protocol X reads the full canvas at all 180 time points. Gemma adds 16 rows at the bottom for its processor.

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

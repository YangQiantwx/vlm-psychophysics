# Visual Psychophysics

Five controlled visual judgments in Classic and Natural scenes. The release contains 24,864 videos, 23,904 static evaluation records and 960 T5 inspection records. [Dataset](https://huggingface.co/datasets/DrQY/vlm-psychophysics) · [Stimulus viewer](https://vision-tasks.pages.dev/library/) · [Model results](https://vision-tasks.pages.dev/experiments/)

| Task | Judgment | Classic | Natural |
|---|---|---:|---:|
| T1 | Horizontal stripe motion | 4,224 | 4,224 |
| T2 | Side that briefly brightens | 3,840 | 3,840 |
| T3 | Side with the vertical object | 2,736 | 2,736 |
| T4 | Horizontal motion signal | 768 | 1,536 |
| T5 | Orientation match to a reference | 384 paired + 96 earlier | 384 paired + 96 earlier |

The counts are trials. Static evaluation shares a call when ordered media, prompt and choices match. T5 equal-canvas evaluation uses the original paired videos with centered 768 × 512 reference images, preserving the object's pixel size. Its 768 videos correspond to 425 unique static inputs. The earlier T5 inspection records retain their original reference format.

Evaluation of the additional 768 Classic T4 videos is complete; publication of the extension will bring Classic T4 to 1,536 videos.

## Run

Use Linux, Python 3.12, RTX PRO 6000 Blackwell GPUs and CUDA Toolkit 13.0.3. Install into your existing environment:

```bash
python -m pip install -e ".[download,inference]"
python run.py --task t1-classic --model qwen
```

The runner selects writable SSD storage, downloads the pinned dataset and model, checks the runtime, and assigns work by available GPU memory. Each GPU loads and finishes one model recipe at a time. Set `VLM_PSYCHOPHYSICS_CUDA_HOME` if the CUDA toolkit is outside the standard location. Authenticate with Hugging Face for gated models and set `NGC_API_KEY` for NVIDIA downloads.

```bash
# All four models, T1–T4, both domains, video and static
python run.py

# One published video; a local video path also works
python run.py --task t1-classic --model qwen --video T1_left_sf0.5_contrast0.05_trial01.mkv

# Same-size T5 reference, both domains, all four models, video and static
python run.py --task t5-classic-equal-canvas --task t5-natural-equal-canvas

# Replay the original 160 × 160 reference-image T5 video version
python run.py --task t5-classic-paired --model qwen

# Rebuild plots and verify a returned result
python run.py report --out outputs/RUN
python run.py verify outputs/RUN/return.zip
```

Each output directory contains `results.csv`, `results.jsonl`, `summary.json`, `curves.png` and `return.zip`. The `evidence/` folder holds exact prompts, ground truth, resolved model settings, source and input hashes, and execution records. Return `return.zip` for verification and rescoring. Repeating a command with the same output resumes untouched inputs and preserves recorded attempts.

Static inputs use one shared ordered-image path: T1 has frames 0–3; T2/T3 use the event midpoint (`onset + 15`); T4 uses frames 0 and 179. T5 places the reference first, followed by one or two event midpoint images in chronological order. All T5 equal-canvas images are 768 × 512. Video input is the reference followed by all 180 frames. The frozen video and static prompts are preserved byte for byte. Four models evaluate 4,772 independent inputs across both T5 domains; single and dual cohorts retain their respective image limits of 2 and 3.

To rescore a paired T5 answer bundle, run `python run.py report --paired-results BUNDLE --out REPORT`. This writes the planned trial scores, paired single-to-dual changes and cohort counts.

Use `--input video` or `--input static` to select a modality, `--sample-id` for a scientific ID, or repeat `--task` for several scopes. `--config configs/current-x.json` selects the complete model recipe. Optional `--data-root`, `--model-root`, `--out` and `--gpus` select storage and GPU UUIDs. `run.py plan` shows the selection, `run.py download` fetches it, and `run.py doctor` checks the environment. Add `--media-only` to download data without weights, or `--offline` to use existing local files.

The repository is currently private; collaborators need GitHub access before cloning. [Protocol and hosted Model API](EXPERIMENT_PLAN.md)

## Regenerate a stimulus

```bash
python generate.py --task t2-classic --sample-id t2_000201cb3983b82df70a --out replay
```

Use `python generate.py --task all` to regenerate all released T1–T5 videos. Repeat `--sample-id` to select several trials. Select `t5-classic-paired` or `t5-natural-paired` for paired T5, and `t5-classic` or `t5-natural` for the earlier unmasked stimuli. The renderer uses the frozen source parameters and assets from the pinned dataset release, checks decoded pixels and timestamps, and reports encoded container equality separately. Set `--data-root` to reuse local sources. [Source files and attribution](https://huggingface.co/datasets/DrQY/vlm-psychophysics/tree/main/sources)

The released input manifests contain the exact prompt, media order and hashes. Send only the prompt and media to a model. Filenames, labels and provenance stay in the evaluation record. [Per-trial answers, prompts and counts](https://huggingface.co/datasets/DrQY/vlm-psychophysics/tree/main/results/current-x)

# Visual Psychophysics

Five controlled visual judgments in Classic and Natural scenes. The [Hugging Face dataset](https://huggingface.co/datasets/DrQY/vlm-psychophysics) contains **20,256 videos and 20,256 static records**. Explore the [stimulus library](https://vision-tasks.pages.dev/library/) and [model results](https://vision-tasks.pages.dev/experiments/).

[Model API](EXPERIMENT_PLAN.md#model-api)

| Task | Judgment | Classic | Natural |
|---|---|---:|---:|
| T1 | Horizontal stripe motion | 4,224 | 4,224 |
| T2 | Side that briefly brightens | 2,880 | 960 |
| T3 | Side with the vertical object | 2,736 | 2,736 |
| T4 strict | Horizontal motion signal | 768 | 1,536 |
| T5 unmasked | Orientation match to a reference | 96 | 96 |

The table counts scientific trials. Identical PNGs can be shared. T5 static records contain the reference and event midpoint images for stimulus inspection.

## Download and inspect

Use Python 3.11 or later in your existing environment. The example downloads the first Natural T2 video and static input, verifies each media SHA-256, and reports the fixed dataset commit.

```bash
python -m pip install '.[download]'
python -m classic_precision.current_release --scope natural/t2 --limit 1 --output data
```

Omit `--limit` to fetch the full task. Repeat `--scope` to select more tasks, or omit it for all ten scopes. Use `--input video` or `--input static` for one input type. The T5 video option also fetches the required reference image. The dataset index specifies manifest paths, including the `strict-v1` video folders for T4.

Each row gives its ID, question, ground truth, conditions, relative paths and hashes. Send only media in manifest order and the exact question to the model. Filenames and metadata can reveal the answer.

## Reproduce

- [Stimulus and inference protocol](EXPERIMENT_PLAN.md) describes the accepted X routes, configuration inspection, and evidence scope.
- [T1 m = 0.5 and strict Classic T4 source bundle](https://huggingface.co/datasets/DrQY/vlm-psychophysics/blob/20baa977b183c8ecf56b5e4f7da569c5a93e9464/reproduction/t1-m05-t4-strict-v1/README.md) contains the frozen generators and geometry checks.
- The [Natural T2 renderer](reproduction/natural-t2-960/render.py) and [frozen allocation](reproduction/natural-t2-960/combinations.json) reproduce the 960-video panel with the [pinned source bundle and dependencies](https://huggingface.co/datasets/DrQY/vlm-psychophysics/blob/20baa977b183c8ecf56b5e4f7da569c5a93e9464/reproduction/natural-t2-960/README.md).
- [Strict Natural T4 and unmasked T5 source bundle](https://huggingface.co/datasets/DrQY/vlm-psychophysics/blob/20baa977b183c8ecf56b5e4f7da569c5a93e9464/reproduction/current-t4-t5.md) reproduces the published videos from frozen combinations and prepared inputs.

The [Label preview](https://vision-tasks.pages.dev/label/) is in test mode. The [Label implementation](web_label/README.md) has separate operator instructions.

## Execution routes and runtime contracts

The [X execution table](EXPERIMENT_PLAN.md#execution-routes-and-runtime-contracts) links the video and static workers for Cosmos, Qwen, Nemotron and Gemma. The [profile extracts](reproduction/current-x-profiles.json) record settings from the executed extensions and identify their original sealed profiles by SHA-256.

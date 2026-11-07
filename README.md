# Visual Psychophysics

Five controlled visual judgments in Classic and Natural scenes. The [Hugging Face dataset](https://huggingface.co/datasets/DrQY/vlm-psychophysics) contains **20,256 videos and 20,256 static records**. Explore the [stimulus library](https://vision-tasks.pages.dev/library/) and [model results](https://vision-tasks.pages.dev/experiments/).

[Model API](EXPERIMENT_PLAN.md#model-api)

| Task | Judgment | Classic | Natural |
|---|---|---:|---:|
| T1 | Horizontal stripe motion | 4,224 | 4,224 |
| T2 | Side that briefly brightens | 2,880 | 960 |
| T3 | Side with the vertical object | 2,736 | 2,736 |
| T4 | Horizontal motion signal | 768 | 1,536 |
| T5 | Orientation match to a reference | 96 | 96 |

The table counts scientific trials. Identical PNGs can be shared. T5 static records contain the reference and event midpoint images for stimulus inspection.

## Download and inspect

Use Python 3.11 or later in your existing environment. The example downloads the first Natural T2 video and static input, verifies each media SHA-256, and reports the fixed dataset commit.

```bash
python -m pip install '.[download]'
python -m classic_precision.current_release --scope natural/t2 --limit 1 --output data
```

Omit `--limit` to fetch the full task. Repeat `--scope` to select more tasks, or omit it for all ten scopes. Use `--input video` or `--input static` for one input type. The T5 video option also fetches the required reference image. Each task directory contains `metadata.jsonl`, `videos/` and `static/`.

Each row gives its ID, question, ground truth, conditions, relative paths and hashes. Send only media in manifest order and the exact question to the model. Filenames and metadata can reveal the answer.

## Reproduce

The [stimulus reproduction guide](https://huggingface.co/datasets/DrQY/vlm-psychophysics/blob/13e9cf67bb0b63be562ed7b7a48668fa7b7bca22/reproduction/README.md) links the generators, frozen inputs and source attributions. The [experiment protocol](EXPERIMENT_PLAN.md) describes the stimulus design and model inputs.

The [Label preview](https://vision-tasks.pages.dev/label/) is in test mode. The [Label implementation](web_label/README.md) has separate operator instructions.

## Execution routes and runtime contracts

The [execution table](EXPERIMENT_PLAN.md#execution-routes-and-runtime-contracts) links the video and static workers for Cosmos, Qwen, Nemotron and Gemma. The [profile extracts](reproduction/current-x-profiles.json) record settings from the executed extensions and identify their original sealed profiles by SHA-256.

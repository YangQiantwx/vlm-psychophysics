# Visual Psychophysics

Controlled motion, brightness and orientation tasks with classic and natural-content source videos. The current source release covers **22,944 T1–T3 videos**. Each source is 768×512, 180 frames at 60 fps (3 seconds).

| Source | T1: motion | T2: brightness | T3: orientation | Encoding |
|---|---:|---:|---:|---|
| Classic | 3,840 | 5,760 | 2,928 | Lossless RGB16 FFV1/MKV |
| Natural | 3,840 | 3,840 | 2,736 | Lossless RGB8 H.264/MP4 |

The classic 9,456-video primary selection retains all T1, 30-frame T2 exposures and T3 angles excluding 1.75°/2.25°. Auxiliary source conditions remain available. [The protocol](EXPERIMENT_PLAN.md) defines stimuli and scientific scope; [the research report](report.md) preserves earlier findings and negative results.

Download sources from Hugging Face:

- [Classic T1–T3](https://huggingface.co/datasets/DrQY/vlm-psychophysics-classic-t1-t3)
- [Natural T1](https://huggingface.co/datasets/DrQY/vlm-psychophysics-natural-t1), [T2](https://huggingface.co/datasets/DrQY/vlm-psychophysics-natural-t2), [T3](https://huggingface.co/datasets/DrQY/vlm-psychophysics-natural-t3)

Each source repository includes its own downloader and reading example. From this checkout, install the optional download dependency in your existing Python environment, then select a complete folder at a fixed dataset commit:

```bash
python -m pip install '.[download]'
python -B -m classic_precision.download_dataset --domain classic --task T3 \
  --revision COMMIT_SHA --output data --workers 8
python -B -m classic_precision.read_dataset_example --root data/classic
```

Use `--domain natural --task T1` for a natural source repository. Omit `--task` to download all tasks in a domain. `--folder T3` selects an entire classic folder; natural folder names start with the task, for example `--domain natural --folder T3`. Each download resolves the requested revision to an immutable commit, records a receipt and verifies the downloaded video SHA256 values.

`catalog.jsonl` and `metadata.jsonl` provide source identities, paths, GT, parameters and condition keys. `metadata-demo.csv` contains a small source-only example. `aliases.csv` and [`IdentityRegistry`](classic_precision/dataset_identity.py) resolve canonical IDs, legacy IDs and exact old/new paths. Classic T1 now uses a versioned SHA-bound `t1_<20hex>` ID; the other IDs retain their existing identity. Filenames, directories and hosts do not enter the identity hash. Parent source IDs describe lineage, rather than aliases for replaced videos.

Readable filenames contain labels. Models and participants receive only the video and task question. T3 `verticalLeft_tiltRight` means that the left object is vertical and the right object's upper end leans right; 90° is `horizontal`. Natural source numbers remain fixed and retain their attribution and license evidence.

[`dataset_release.py`](classic_precision/dataset_release.py) builds a source-only staging tree without uploading. Publishing requires complete SHA-bound, decoded-content validation for every video. Source QA, encoded pixels and actual model preprocessing are separate measurements.

Current official **X** recipes for all four models are in [`current_x_protocols.json`](classic_precision/current_x_protocols.json), with exact source hashes, model/tokenizer revisions, actual runtime receipts, input adapters and generation arguments. They use vLLM 0.33 and official mixed NVFP4 checkpoints.

| Model | FlashInfer autotune | KV cache | Context |
|---|---|---|---|
| Cosmos | ON | 17 GiB | 65,536 |
| Qwen T1 | OFF | 8 GiB | 98,304 |
| Qwen T2/T3 | OFF | 4 GiB | 65,536 |
| Nemotron | OFF | 2 GiB | 65,536 |
| Gemma | ON | 48 GiB | 65,536 |

Inspect a recipe without loading a model:

```bash
python -B -m classic_precision.reproduce show --model gemma --protocol current-x --task T1
```

All use the native 180-frame path. Gemma requires its pinned input adapter: its video budget is 280 soft tokens, yielding 176 effective tokens per frame with 16 bottom padding rows; the profile's image budget of 140 is a separate setting. Qwen requires its pinned QSA worker extension. The export preserves these distinctions and does not start inference. Execute through a finite sealed queue with the source catalog, exact task prompt, native input checks and physical GPU lease.

The older `native-default`, `native-original` and `controlled-rgb16` exports remain historical protocols in `reproduction.json`; their framework/support notes do not describe current X. Running profiles and original scientific IDs remain frozen. A source download does not change a running experiment.

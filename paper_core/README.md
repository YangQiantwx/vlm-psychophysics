# Paper reproduction

The paper uses the 60 fps release with 25,440 native 180-frame clips, 18,795 independent diagnostic-image inputs, and a matched panel of 1,480 clips and 1,346 distinct diagnostic-image inputs. The four original local models’ exact output text, prompts, results and stimulus hashes are in the base release. The panel publishes the association between a clip and its static case so each model's independent static calls can be counted once.

## Retrieve the release

The [paper data index](https://huggingface.co/datasets/DrQY/vlm-psychophysics/tree/2e95900ba7d6c9c9b1220d834e34bf57d2681485/paper-core) records the source revisions, SHA-256 hashes and file sizes. Obtain just the metadata and results to recompute the numbers. The figures also require `paper-core/figures/`, the one original Fig. 4 example MP4 and its marked companion. The example ID and two paths are in `t3-axis-inputs.json`.

```sh
hf download DrQY/vlm-psychophysics --repo-type dataset \
  --revision 29d6d2a70c5b1c0a5b3b33f027a2870add570c93 \
  --include 'dataset-index.json' '*/metadata.jsonl' 'results/t*/results.csv.gz' --local-dir full
hf download DrQY/vlm-psychophysics-subset --repo-type dataset \
  --revision f562d61b4703b744c2f00f66a7ebabeacc4e0a0a \
  --include 'dataset-index.json' '*/metadata.jsonl' 'results/t*/results.csv.gz' 'subset/*.csv' --local-dir panel
python -m paper_core.scores --base full --panel panel --study release/paper-core --out reproduced
```

Use `hf download DrQY/vlm-psychophysics --repo-type dataset --revision 2e95900ba7d6c9c9b1220d834e34bf57d2681485 --include 'paper-core/*' --local-dir release` to retrieve the frozen paper-only files. Install the project's regular dependencies plus `pandas` and `pymupdf` for `--figures`. The TeX Gyre fonts reproduce the typesetting used for the plots when available.

## Figures and tables

`paper_core.scores` parses the recorded labels and combines them with the frozen manifests. `tables.json` reports exact denominators and numerators. A prediction missing after reaching its output limit stays in the planned denominator and scores zero. The Nemotron T5 control records the original truncation and the one same-configuration retry used for the final number.

| Item | Reproduction inputs and entry point |
| --- | --- |
| Table 1 | Counts the original video manifests by task and domain. Their `parameters` fields preserve the stimulus factors. |
| Table 2 | Joins matched 1,480-clip and full predictions, then equally weights task and scene. |
| Table 3 | Recomputes five-model clip-pooled accuracy and the sustained 75% signal criterion from the full predictions. |
| Table 4 | Compares the same clips using per-trial model labels. For the four observers it uses aggregate counts on the 1,478 common clips. |
| Fig. 1 | `--figures` calls `paper_core.figure1` with the original 25 verified source frames and their receipts, generating PDF and editable draw.io. |
| Figs. 2, 3 and 5 | `--figures` recomputes the 60 fps orientation, brightening and coherence curves from full-set predictions. |
| Fig. 4 | `--figures` reads the paired 120-clip axis control, recomputes the object-block bootstrap and decodes the 4° display examples. |
| Fig. 6 | `--figures` plots the 134 aggregate signal-level cells and fixed-observer stimulus-block intervals. |
| Fig. 7 | `--figures` rescans the four Qwen video conditions and rebuilds the stripe, group-motion and matching plots. |

Human accuracy, uncertainty and timing are aggregate values. `human-figure6.json` contains no observer rows or per-trial human choices. The hosted API tables contain verified final label projections. Raw API response envelopes and request credentials are held in the original access-controlled records.

## Model and intervention inputs

The base `run.py` runs the four frozen direct-answer local recipes. `configs/runtime-lock.json` pins model artifacts and environment. The reproducible local run uses a Linux NVIDIA environment with the matching runtime, weights and one selected physical GPU. The [API runner](https://github.com/YangQiantwx/vlm-psychophysics-subset/tree/2b49d6efe2d79c8e8e24b6dd847d6a2c4f634fa1) pins the requested deployment, the input manifests and the immutable PNG-image index. `api-configs.json` records the transport and generation settings used for the paper. For the ordered conditions, one MP4 supplies 180 PNG images in chronological order and T5's reference comes first. The native Gemini path sends MP4. Static cases use the images in manifest order. The provider returns final responses, but its internal preprocessing can differ from the submitted inputs.

For the Qwen reasoning controls, the Nemotron/Cosmos T5 controls and Muse, `paper_core.replay` verifies one selected trial's source bytes and builds the model request. `--execute` runs it using the matching pinned weight directory. Choose the process's GPU with `CUDA_VISIBLE_DEVICES`. The local `model-configs.json` records the observed engine flags, model revisions, package versions and source profile SHA-256 values. The base results provide the pre-existing direct baselines. This command is a single-trial replay and writes exact prompt and output token IDs for comparison.

```sh
python -m paper_core.replay --study release/paper-core --data panel \
  --id n5p1_08094d1355d601e965e3ecfa --model qwen --mode png --arm low --out replay
```

The command above prepares the ordered PNG request without inference. Use `--mode video` for native MP4 and `--mode static` for the diagnostic images. For the marked and enlarged videos, pass `--variant axis` or `--variant enlarged` on the corresponding sample ID. The Muse label-only recipe lives in `vlm_psychophysics/glimmer.py`. The Molmo2 BF16 input bridge and original execution entry point are in `paper_core/molmo/`. Prepare its Hugging Face snapshot and runtime from `model-configs.json` and use its named pinned dataset directory. See the runner's `--help` for output, GPU lock and checkpoint receipt parameters.

`paper_core.media` regenerates exact intervention pixels in the paper. For T3, supply the native source video and original diagnostic midpoint. Extract the digest-named alpha arrays from `t3-axis-patches.zip` for the background-removal control. The recipe verifies every resulting view's decoded-pixel digest. For T5, supply the original video and reference from the full release and the source sprite from `sources/t5/objects`; see `t5-enlarged-inputs.json` for the sprite path and digest. Encoded container hashes of the regenerated media can vary with the toolchain. The decoded RGB and frame times are verified separately.

Source photos and assets retain the terms documented by the base release. Refer to [Natural T3 credits](https://huggingface.co/datasets/DrQY/vlm-psychophysics/blob/29d6d2a70c5b1c0a5b3b33f027a2870add570c93/natural/t3/attribution.json) and [Natural T4/T5 credits and license](https://huggingface.co/datasets/DrQY/vlm-psychophysics/tree/29d6d2a70c5b1c0a5b3b33f027a2870add570c93/sources/attribution). The axis intervention overlays guides on the parent images or videos. The T5 intervention enlarges the parent reference and candidate objects.

# Current stimulus and model protocol

The fixed input release is [DrQY/vlm-psychophysics at `20baa977`](https://huggingface.co/datasets/DrQY/vlm-psychophysics/tree/20baa977b183c8ecf56b5e4f7da569c5a93e9464). The [dataset index](https://huggingface.co/datasets/DrQY/vlm-psychophysics/blob/20baa977b183c8ecf56b5e4f7da569c5a93e9464/dataset-index.json) binds the video and static manifests. Source lineage, released media hashes, exact prompts, ground truth and ordered input paths are recorded per trial. Historical configurations and results remain in the [prior protocol](https://github.com/YangQiantwx/vlm-psychophysics/blob/e9be5fd206cd7d0307528b94f5192375e687a1f3/EXPERIMENT_PLAN.md) and [research report](report.md).

## Stimuli

All videos are 768 × 512, 180 frames at 60 fps. Classic T1–T4 use lossless sRGB16 FFV1/MKV. Natural T1–T5 and Classic T5 use lossless sRGB8 RGB H.264/MP4. Model decoder and processor receipts specify their own input dtype, spatial geometry, frame count and time grid.

- **T1.** Eight spatial frequencies, 24 phases and balanced horizontal directions at each modulation depth. The 0.5 extension contributes 384 trials per domain. Natural uses multiplicative modulation in linear RGB.
- **T2.** Local brightening during the 30-frame event. Natural uses a frozen 20-increment × 24-onset × two-side panel. Each of the 480 left/right pairs uses the same background and object. The four photo/generated background × object groups each contribute 240 trials, with 12 per increment and 10 per onset. All 200 backgrounds and 100 objects appear in each luminance-increment half. Use the [frozen combinations and renderer](https://huggingface.co/datasets/DrQY/vlm-psychophysics/tree/20baa977b183c8ecf56b5e4f7da569c5a93e9464/reproduction/natural-t2-960) and the [matched Classic pairs](https://huggingface.co/datasets/DrQY/vlm-psychophysics/blob/20baa977b183c8ecf56b5e4f7da569c5a93e9464/natural/t2/classic-natural-pairs.csv).
- **T3.** The objects appear for 30 frames. The current Natural grid pairs with the 2,736 Classic configurations. Left/right pairs keep the background, object, onset, angle and physical tilt direction fixed.
- **T4 strict v1.** Classic crosses N = 8 or 16, eight nonzero coherence levels, 24 layouts and both directions. Each pair shares start positions. All trajectories stay contained and separated over the full motion interval. Natural crosses those same trajectories with same or distinct object appearances. Each appearance contributes 768 trials. The two appearances share a background and trajectory for every matched trial.
- **T5.** An orientation reference image precedes the video. Public inputs cover 192 unmasked trials. The original Cosmos evaluation includes an additional masked condition with access-controlled sources.

| Task | Ordered static input |
|---|---|
| T1 | Frames 0, 1, 2 and 3 |
| T2, T3 | One frame at onset + 15 |
| T4 | Frames 0 and 179 with the [endpoint prompt](reproduction/t4-static-protocol.json) |
| T5 | Reference image, then event midpoints for stimulus inspection |

The static X protocol uses RGB8 PNG inputs and the modality-specific prompts from the static manifests. Preserve the stated order and exact prompt, including whitespace. Report unique model calls and scientific aliases separately when identical ordered pixels, prompt and choices share a model input. Static T2 judges apparent brightness in a frame. T1 and T4 use ordered displacement cues.

## Execution routes and runtime contracts

Current **X** uses the official mixed NVFP4 checkpoints for Cosmos, Qwen, Nemotron and Gemma, retaining each checkpoint's native BF16 components. The [T1–T3 video configuration](classic_precision/current_x_protocols.json) pins the model and tokenizer revisions, request adapters, engine parameters and generation settings. Video uses each model's accepted native 180-frame route. Qwen's QSA extension must apply inside the worker. Gemma's video budget is 280 soft tokens, with 176 effective tokens per frame and 16 bottom padding rows in the qualified T1–T3 input. Its static image budget is 140 and has separate processor evidence.

```bash
python -m classic_precision.reproduce show --model gemma --protocol current-x --task T1
```

| Scope | Entry point or worker |
|---|---|
| Classic T1–T3 video | [`official_x_runner_v2`](classic_precision/official_x_runner_v2.py), with [`official_x_runner_qsa`](classic_precision/official_x_runner_qsa.py) for Qwen |
| Natural T1–T3 video | [`natural_x_runner`](classic_precision/natural_x_runner.py), using the accepted per-model native routes |
| T1 m = 0.5 video, strict Classic T4 video | [`t1_m05_t4_classic`](classic_precision/t1_m05_t4_classic.py) |
| Classic/Natural T1–T3 static | [`static_x_plan`](classic_precision/static_x_plan.py), [`static_x_queue`](classic_precision/static_x_queue.py) and [`static_x_admission`](classic_precision/static_x_admission.py) |
| T1 m = 0.5 static | [`static_t1_m05`](classic_precision/static_t1_m05.py) |
| Strict Classic T4 static | [`static_t4_strict_operations`](classic_precision/static_t4_strict_operations.py) and [`static_t4_strict_queue`](classic_precision/static_t4_strict_queue.py) |
| Strict Natural T4 video and static | [`natural_t4_x`](classic_precision/natural_t4_x.py) |
| T5 reference image and video | [`cosmos_t45`](classic_precision/cosmos_t45.py) |

The X worker entry points consume sealed local profiles, source and runtime receipts, and physical GPU leases. Set `PSYCHOPHYSICS_WORKSPACE` to the execution workspace and keep its checkout at `code/vlm-psychophysics`. Set `PSYCHOPHYSICS_QUEUE_OWNER` and `PSYCHOPHYSICS_PARENT_QUEUE_OWNER` to the IDs recorded in your own current and parent queue receipts when using those queue stages. Use matching checkpoint, tokenizer, processor, backend, GPU qualification and source evidence when sealing a new run. The configuration export and [profile extracts](reproduction/current-x-profiles.json) inspect settings without loading a model. The [extension source identities](reproduction/current-x-code.json) record the original and checkout SHA-256 values of workspace-bound modules.

Natural T4 video and static each completed 6,144 calls across four models and 512 condition cells. Both passed terminal and returned-file SHA-256 verification. The first static launch stopped at admission with zero calls. The [recovery module](classic_precision/natural_t4_static_recovery.py) binds the original video closure and the successful static re-seal. For a new static run, the default route scopes the endpoint validator to the verified Natural T4 static protocol and delegates prior qualification records to their original checker. It preserves the endpoint inputs, prompt and generation settings. The [profile extracts](reproduction/current-x-profiles.json) include both terminal digests.

## Replay the released Natural T4 and unmasked T5 stimuli

Download the required `reproduction/` files and video manifests from the pinned Hub snapshot. Then run the [checkout renderer](reproduction/render_new_stimuli.py) with that snapshot and the pinned dependencies. Its exact container check uses FFmpeg `libavcodec 62.28.101` and `libavformat 62.12.101`.

```bash
python reproduction/render_new_stimuli.py natural-t4 --snapshot HUB_SNAPSHOT \
  --output T4_OUTPUT --sample-id natural-t4-strict-v1-n8-same-l00-c01-left
python reproduction/render_new_stimuli.py t5-unmasked --snapshot HUB_SNAPSHOT \
  --output T5_OUTPUT --sample-id c45_04398e2a95e9107049dbf1c5
```

## Run the Jinwei evaluation

Use your own NVIDIA internal API account. Set `NVIDIA_API_KEY` in your shell and set `JINWEI_MODEL` to the exact model ID listed by your account. The inventory command reads the account's model list without inference. The planning command verifies the public release manifests and reports the number of requests.

```bash
python -m classic_precision.provider_api --provider nvidia-internal \
  --inventory --out jinwei-models
python -m classic_precision.jinwei_eval plan --out jinwei-output
```

The full public plan contains 20,256 video requests and 14,682 distinct static X requests covering 20,064 scientific static records. The 192 T5 static inspection records stay outside the API evaluation. The executor first admits each input route through a pilot. Run with the same output path to continue pending requests. `--max-requests` caps new attempts per invocation, including pilots.

```bash
python -m classic_precision.jinwei_eval evaluate --out jinwei-output \
  --model "$JINWEI_MODEL" --execute --max-requests 34938 --workers 2
python -m classic_precision.jinwei_eval verify --archive jinwei-output/return.zip
```

Return `jinwei-output/return.zip` with `return.zip.sha256`. The archive includes the plan, input evidence, original responses, failures and coverage. A partial run returns status 2 and still writes both files. Use `--media-root` for a local copy of the pinned media. An authorized private masked T5 supplement is a separate opt-in through the generation-pinned private index and `--allow-private-media-transfer`.

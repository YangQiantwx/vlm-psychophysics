# Classic protocol and reproducibility

This document defines stimuli, denominators and validated scope. It is not a scheduler or permission to resume a queue. Exact reproduction requires the versioned manifest, model/processor/code identities and complete raw references. Current local decisions are maintained in the benchmark registry.

## Source panel

| Task | Conditions | Videos per model |
|---|---|---:|
| T1 | 8 spatial frequencies × 10 modulation depths × 24 phases × 2 directions | 3,840 |
| T2 | 13 increments × 3 backgrounds × 24 onsets × 2 sides × 2 durations | 3,744 |
| T3 | 19 angles × 24 onsets × sign/side; equivalent signs at 90° deduplicated | 1,776 |
| Total | 180 frames per video, 60 Hz | 9,360 |

Videos are 768×512, FFV1 level 3 in MKV, gbrp16le, 16 bits per channel. Continuous linear-light stimuli use FP64 pixel-area integration followed by one sRGB16 quantization. Decode into `uint16 [180,512,768,3]` in RGB order; the FP32 entry mapping is `codes * 255/65535`. No uint8 intermediate, frame subsampling or hidden resize. Gemma adds 16 background rows at the bottom without changing the original active region.

T2/T3 onsets are `18 + 5k`, k=0…23; event windows are half-open. These are timing instances of the same content, not 24 independent scenes. T1 instead varies initial phase and moves throughout the clip. Each manifest row retains frame indices, PTS, event support, truth, prompt and parameters.

## Stimuli

**T1 — motion direction.** SF is .5/1/2/4/8/16/32/64 cycles per 448 px aperture; modulation is .05/.1/.2/.3/.4/.6/.7/.8/.9/1. Phase is `7.5 + 15k` degrees; temporal frequency is 5 Hz. The aperture is centered at (384,256), flat for r≤192 px and cosine tapered over 192–224 px. Mean linear Y is .31. The continuous field is `Y = μ[1 + mW(r)cos(2πSF(x−384)/448 + φ − s2π5t)]`, with s=+1 for RIGHT. Each SF×m cell has 48 videos.

**T2 — transient brightening.** Background Y is .05/.20/.40. Both circles initially have Y=.50, centers (224,302)/(544,302), diameter 80.73040721938105 px and area 5118.7529296875 px². Only the target increases by ΔY=.05/.06/.07/.08/.09/.10/.11/.12/.13/.14/.15/.20/.40 for 10 or 30 frames, then returns to baseline. The reference and background stay fixed; there is no clipping. Each background×increment×duration cell has 48 videos. The current set has 1,872 matched duration pairs: the original 2,304-video panel plus 1,440 short-duration extensions. The old BG+.10 object baseline is a separate version.

**T3 — image-plane orientation.** Background/foreground Y is .10/.82. Rectangles are continuously defined as 20×190 px at (224,304)/(544,304). Both appear for 30 frames; one is vertical, the other already tilted by ±θ. There is no rotation animation. The answer is the more vertical side. Angles are .5/.6/.7/.8/.9/1/1.5/1.75/2/2.25/2.5/3/3.5/4/8/16/32/45/90°. Ordinary cells contain 96 videos; 90° contains 48. Coverage comes from continuous polygon/pixel intersections, not repeated rotation of a small bitmap. Axes are periodic over 180°; this is not a retinal visual angle or 3D pose measurement.

## Validation and model input

The full source audit passed 9,360 videos and 1,684,800 frames: decoded RGB codes equal their sources, no failed IDs, maximum PTS rounding error .333 ms. All eight T1 basis images were rebuilt; 648 sampled pixels were independently integrated. Maximum T1 blind-readout errors were 4.730e−6 in modulation, 3.267e−6 cycles/aperture in SF and 1.445e−6 Hz in temporal frequency. T2 core ΔY error was at most 2.2641e−5, with no change outside the target. T3 maximum errors were 5.3904e−5° by PCA and 3.4647e−6° by an independent boundary fit.

Repeated onsets share a state measurement only when bound by full decoded-frame equality. Do not count them as independent geometries. Source acceptance is T2 increment error <1e−4; for T3, each estimator has MAE <.0002° and each geometry has error <.001°. RGB16→FP32 roundtrip preserves all 65,536 codes. These checks do not certify every BF16 signal or a physical display. Brightness is digital linear luminance, not cd/m².

Retain official normalization, patch construction and position handling. Some processors return FP32 before the model casts to BF16. New geometry, processor, modality or engine gets representative actual CPU/GPU qualification. Ordinary requests then retain identity, frame/shape/event/prompt/runtime checks and full outputs, without repeated pixel inversion, processor passes or permanent tensor dumps. Do not compensate source values according to model scores.

## Native deployment sensitivity is a separate protocol

The `native-default` and `native-original` routes both use the official file uint8 reader and all indices0…179. Disable processor frame sampling as well as requesting180 frames in the reader; requesting180 at only one boundary was insufficient for Qwen/Cosmos. Preserve the exact prompt and generation controls below. Observers record dtype, source shape, actual processor frame/grid output and timing; they do not cast pixels. The first two observed requests are excluded from ordinary-request timing.

Original-size Qwen/Cosmos use `do_resize=False` at global and request scope. Nemo uses native vision-config fields `video_target_num_patches=1536` and `video_maintain_aspect_ratio=True`; it does not accept Qwen's do_resize switch. Native default Nemo resizes768×512 to512×512 rather than letterboxing. Native Qwen/Cosmos default output is448×288. Neither operation is geometrically identical to the source; image-plane angle and pixel support can change.

Pinned Gemma native video forces70 soft tokens/frame. Disabling resize on512×768 yields1536 patches, incompatible with its original pooling/token-count path; no valid original-size native GPU timing is claimed. Its existing all180 default-resize T2 median is7.81s, versus about21.38s on a matched historical Fourier subset; host, geometry, soft-token budget and input route change together. This is not a bit-depth-only speedup.

Processor outputs in these pinned native paths are FP32 for Gemma/Qwen/Cosmos and BF16 for Nemo, followed by native vLLM's model-dtype conversion. Gemma's internal `2*(x-.5)` then operates on the received BF16 in native vLLM. Direct HF FP32 input can have a different order. Our Gemma bridge's explicit earlier cast must be disclosed, not described as leaving all postprocessor work untouched. The two static151 configurations remain valid separate observations (102 versus100 correct;18 changed answers).

The old-to-native comparisons change more than bit depth. They do not establish that RGB16 preservation universally improves or harms VLM accuracy. Same-uint8 native A/B comparisons isolate the supported spatial setting more closely. Keep selected cells and full curves separate, and do not select the highest configuration per model/cell for the main result.

## Model paths and generation

| Validated path | Precision and framework |
|---|---|
| Qwen3.8 Flash Next, fc694b54… | Official mixed NVFP4/FP8/BF16; vLLM 0.31; QSA-only deterministic selection; BF16 KV 4 GiB; native 24 GiB expert offload |
| Nemotron3.5 Super VL, 82d93699… | Official mixed NVFP4/FP8/BF16; vLLM 0.31; BF16 KV 2 GiB |
| Cosmos3 Super, f543c562… | Official BF16 Reasoner and Transformers 5.17/SDPA; complete classic reference |
| Cosmos official NGC NVFP4, manifest 7bbfe34d… | ModelOpt W4A4 and vLLM/FlashInfer; BF16 vision/lm_head/KV 17 GiB; all9,360 complete |
| Gemma4-31B official NVFP4, 4135a98… | MLP W4A4; BF16 attention/vision/lm_head/KV 32 GiB; vLLM0.31; no CPU offload; original3,145 preserved;explicit same-protocol resume1 of6,215 active |

Pin checkpoints, processor/tokenizer/chat template, code, SDK, kernels, Torch policy and frame/prompt identities. Fix thinking OFF, temperature=0, top_p=1, seed=0, greedy decoding, context=65,536, output cap=16,384 and repetition penalty=1. HF uses do_sample=False and num_beams=1. Native vLLM LEFT/RIGHT choice constraints or the equivalent HF prefix/EOS constraint are part of the evaluated system. Contexts are independent. Source, processor, visual activation, weight and KV precision are different quantities; NVFP4 does not mean every layer is four-bit.

A fixed decode/processor pipeline is not model-generated Python or agentic tool use. Optional API/agent evaluations need separate enforceable tool permissions. A sentence in a prompt does not prove service behavior; completed native results do not need rerunning merely to reword that statement.

## Completed reproducibility and precision evidence

Gemma original3,145 formal outputs remain preserved; the human authorized same-protocol three-card resume1 of6,215 never-attempted IDs. Each new worker's first2 actual source/CPU/GPU/fullprompt checks passed. This does not change input geometry/dtype policy or authorize old-worker restart.

Latest native QSA follow-up: T2 default46/48 and original-size48/48, with48/48 full prompt/output/answer/finish/stop MATCH per arm across Fourier cards. T3 .7° default61/96 independently replayed96/96 full-output MATCH. Existing unpatched T2 and controlled-input references remain separate.


- Corrected Qwen: 49 T2/T3 inputs across four observations and four T1 inputs across four observations matched in full tokens; 48 cross-host T1 controls also matched. Old QSA0 and 2,240 legacy answers remain separate.
- Nemo: the same 48 inputs matched across two cold loads and two observations per load on Fourier. Three differences against 16 old Radon inputs are cross-runtime historical disagreement, not internal instability of the four Fourier observations.
- Cosmos BF16: 49 inputs across four independent cold observations matched corresponding production. Official NVFP4 completed 432 fixed IDs: T1 70→74/144, T2 120→121/144, T3 144→144/144; total 334→339 with 23 changed answers. Precision and backend/timing scopes change together. Subsequent all-T3 work independently repeated the original144 pilot IDs across three cards with full-token/answer/finish MATCH. Old actual-prompt hashes exist for 144/432 IDs and GPU-content hashes for 0/432; retain these gaps.
- Gemma NVFP4: the same 48 videos in two independent cold engines on Radon had zero input/full-token/answer/finish differences. This does not establish all-input or arbitrary-hardware equivalence. One of the initial 20 answers changed against BF16 C3; offload changed from 8 to 0 GiB and the publisher did not disclose the exact BF16 parent revision.

Qwen, Nemo, Cosmos NVFP4 and Gemma NVFP4 each completed9,360 controlled-input videos. Their single-image456 and ordered-image960 controls are also complete. Earlier BF16 runs, speed-search and six-cold plans remain historical, not an automatic schedule.

## Reporting and finite execution

A normal record needs one durable complete raw response, tokens/finish, exact sample/attempt identity and a bound technical pass. Persist generate intent before the call and raw before derived technical results. Preserve actual exit codes and separate load, decode, prepare, generation and recording costs. Missing durable output requires diagnosis. Current refinement plans permit no automatic retry; completed or uncertain IDs are not redispatched. Do not resample complete wrong answers or redispatch an uncertain commit. Use the existing physical GPU lease and protect other processes.

Report correct/N, side/sign/onset bias, strict format and paired-duration wins/losses. Plot measured points without forced monotonic fitting. Phase/onset-block uncertainty describes the fixed stimulus set, not arbitrary natural content. Answer changes and full-token changes are distinct. Speed comparisons name their timing scope, host, backend and loading cost; do not subtract clocks from different hosts.

Post-run selection uses seed `cvpr2027-postrun-cells-20261023-v1`: rank complete cells by exact correct/N, order ties by seed, split into rank thirds, select one random cell per third and run every ID once. Seal all candidates, reference hashes and IDs before calling a model. Changed inputs/runtime are INVALID; different full tokens under matched controls are DIFFERENT and stop the affected group for diagnosis. Keep coverage gaps explicit.

## T4/T5 and natural transfer

Current T4 results are ordered endpoint images. Seven new RDM source videos passed 1,260-frame QA; actual processor/model pilots and human review remain pending. The proposed three-interleaved-set, Bernoulli-signal, random-position-noise stimulus is a versioned Kiani-style variant. Record visible membership, motion, edge handling and 0% response bias. Endpoint validity does not establish full-video validity.

T5 mask/bias results remain in report.md. The proposed next pilot separates long-duration visibility, no/short mask and single/double events, with balanced YES/NO. Report hits and false alarms. This is a combined-task proposal, not a demonstrated human attentional-blink mechanism. Review actual samples before new formal GPU work.

Natural backgrounds and object appearance retain their baselines. Deterministic target manipulations use FP64 linear RGB, premultiplied alpha and one final RGB16 quantization. T1 grating-on-background transfer is separate from moving texture. T2 uses `F1=F0+k(1−F0)`, `k=Δ/(1−μ0)`: Δ is an alpha-weighted mean increment, not the same increment at every pixel. T3 uses one affine transform from the original cutout, fixed axis/scale/pivot and independent geometric readout. T4 reuses trajectories when changing appearance; large sprites require matched size/alpha controls. T5 uses finite masks and matched event timing. The material pool is not an exhaustive BG×OBJ cross product. Full natural-video/processor acceptance and formal inference await review.

## Code and publication

[README](README.md) documents usable entrypoints and the legacy 3,504-video CLI. [report.md](report.md) preserves historical findings. The current 9,360-video index, QA and finite-run references are separate release inputs. A local wheel/help check is not a published reproduction. Complete portable T1–T4 integration, frozen dependencies/reference bundles, clean-checkout curve/sample reconstruction, selected same-version GPU audits and verification of the actual published commit. The T1–T3 source dataset/read-only reader v0.1 is now public at https://huggingface.co/datasets/DrQY/vlm-psychophysics-classic-t1-t3 (v0.3 commit a28b0add9ee7621191b453c03d9a69d60e30f83e); all9,360 remote source hashes match and anonymous clean downloads from all four source families passed. This does not complete the full GPU/code reproduction release; reuse licensing remains unspecified.

Published naming uses `dataset_names` and three task folders (`T1/`, `T2/`, `T3/`). Stable IDs/parameters/source bytes are unchanged; trial labels follow the original phase/onset bank. Active remote formal catalogs and source paths remain frozen.

## Bounded T2/T3 extension (prepared, not executed)

Use `classic_precision.refinement` against v0.3. T2 adds ΔY .16/.17/.18/.19/.25/.30/.35 across three backgrounds,10/30-frame exposure,24 onsets and two sides:2,016 videos. T3 adds1.2/1.4/1.6/1.8/2.1/2.2° across24 onsets, both signs and both reference sides:576 videos. Total2,592 new videos and10,368 answers for Qwen/Nemo/Cosmos/Gemma under their existing qualified mixed NVFP4 controlled-input protocol. The earlier .225 proposal was removed before execution. Existing1.75/2.25° and all9,360 original IDs remain unchanged. The common grid is fixed before new answers; do not expand it automatically. New-source QA and runtime admission remain pending. No new static grid, T4/T5, natural content or additional models are included.

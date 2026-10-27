# Selected findings · 16–27 October 2026

<!-- reader-route-smoke-current -->
## 读取路线 smoke：原始账齐，Nemo原协议验收HOLD

2026-10-27 04:18:29 HST 完整逐ID技术账已回传：1728唯一raw/PASS与1728时长，15个成功逻辑worker真实wait0/owned退出/原同inode lease free齐全；两条原0raw CPU失败及原claim保留。新repair1队列终态COMPLETE，旧HELD未改。**这是原始技术账齐，不是四模型原协议最终验收。** Nemo432条实际使用与历史不同的FlashInfer后端，以下Nemo*行仅保留变体，不当原协议数据。正式补点没有恢复。

A为原RGB16→FP32读取，B只换官方OpenCV uint8→FP32，C为官方读取/原生处理；Gemma C保留共用空间/输入适配。全部180帧0..179、有效768×512，Gemma另底pad16到768×528，176有效视觉token/帧。原题目、greedy/seed/nativeEOS与QSA约束沿用。

|模型|任务/分母|旧历史|A|B|C|A→B 改答(对→错/错→对)|B→C|A→C|普通E2E均值秒 A/B/C|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
|qwen|T2/48|38|38|37|37|17 (9/8)|0 (0/0)|17 (9/8)|14.75/13.10/12.67|
|qwen|T3/96|54|54|60|60|22 (8/14)|0 (0/0)|22 (8/14)|15.51/13.77/13.50|
|cosmos|T2/48|37|37|42|42|9 (2/7)|0 (0/0)|9 (2/7)|14.31/12.50/12.18|
|cosmos|T3/96|81|81|73|73|12 (10/2)|0 (0/0)|12 (10/2)|20.16/18.71/18.36|
|gemma|T2/48|24|24|24|24|0 (0/0)|0 (0/0)|0 (0/0)|21.14/20.32/19.74|
|gemma|T3/96|49|49|51|51|6 (2/4)|0 (0/0)|6 (2/4)|21.24/20.41/19.75|
|nemo*|T2/48|28|26|30|28|10 (3/7)|8 (5/3)|10 (4/6)|10.63/9.48/9.21|
|nemo*|T3/96|48|48|48|48|0 (0/0)|0 (0/0)|0 (0/0)|10.84/9.54/9.26|

**Qwen/Cosmos/Gemma三个模型：** 在这两个固定点的432个model/sample配对中，历史→A为0改答；A→B为66/432改答，B→C为0/432改答。该结果说明实际RGB16与官方uint8解码差异会改变答案，而这三个模型本次剩余输入路线没有继续改答；A/B包含实际解码及颜色转换差异，不是理想量化单因素。不能据此推广为所有点或任意路线可互换，也不能只看净正确数。其余三模型1296条调用的逐ID、三组两两转移、LR/sign/onset及阶段mean/median/p90/吞吐在partial CSV/summary。

**Nemo*后端更正与修复状态：** 原正式/成功复现实用CutlassNvFp4LinearKernel、CutlassFP8ScaledMMLinearKernel、VLLM_CUTLASS MoE；本轮A/B/C及836条正式补点却auto改成FlashInfer对应实现。权重/processor/源码不变仍不等于实际算子不变。现用官方KernelConfig(linear_backend=cutlass,moe_backend=cutlass)恢复原实现，8项相关CPU检查通过；3个固定ID×旧/新入口×Radon两次冷载入/Fourier一次，共18次诊断的完整历史token/finish/stop全部匹配，入口间FP32/prompt/GPU视觉输入与输出hash相同。只测试其中2个原改答，未声称14条全部复验。诊断真实闭合证据在runs/reproducibility/nemo-reproduction-diagnosis-20261027/diagnosis.json。

Nemo432 smoke与836正式的原答/claim保留，不混原协议或覆盖旧9360。准确1268个技术修复tuple/ID及旧profile/raw/claim SHA已封plans/reader-route-smoke-20261026/nemo-legacy-restoration-intent.json；追加预算已问，尚未执行。完整reconcile与正式continuation守卫继续HOLD，未签发COMPLETE/REPORTED决定。

**计时口径：** 普通请求与首task冷观察、load分开；同worker monotonic从读取到durable raw/PASS。B与Gemma C的native_reader计数是外部read_decode中的同一次读取，不再相加；其他C的native读/processor位于engine内部。engine-inclusive并非GPU-only，bridge/IPC/H2D/视觉/prefill/decode未独立拆开。相同source OS页缓存预读、fresh engine、prefix/MM/decoded cache关闭，旧历史耗时不用于同输入加速结论。Gemma C准确是官方uint8＋原生HF video processor＋processor后BF16＋共用空间/输入适配，不能称未经改动的vLLM file70路径。

完整逐ID技术闭合独立审计：runs/reproducibility/reader-route-smoke-20261026/partial/terminal-ledger-audit.json。科学原答和负结果保持原版本；本表属于带Nemo原协议HOLD的部分分析，不覆盖正式曲线。
<!-- /reader-route-smoke-current -->

This is the consolidated experiment log. Counts describe completed historical inputs, not files still on disk. These exploratory results explain the current protocol; they do not authorize a queue or become answers for newly rendered videos. The historical sections below retain their original scope. The current controlled-input NVFP4 panel is complete for Qwen/Nemo/Cosmos/Gemma:9,360 videos and1,416 static responses per model. The next common T2/T3 extension is prepared but not executed; see README and EXPERIMENT_PLAN.

## Prompt length and native thinking

The matched five-task pilot contained 36 inputs per model per arm (four arms, 288 observations across two models). Qwen Next scored long+ON 18/36, short+ON 25/36, long+OFF 30/36 and short+OFF 26/36. Nemotron scored 26/36, 27/36, 26/36 and 27/36. Neither prompt length nor thinking mode was a consistent winner.

Qwen's normal-stop, closed-thinking final responses violated the single-label format in 15/36 long+ON and 6/36 short+ON cases. These are format failures, not proof of incorrect visual judgments. Nemotron followed the format in this pilot. The early T4 6/6 result used a single small layout panel, not a stable population result.

**Decision:** fixed long prompts, native thinking OFF, uniform greedy/seed 0, and actual LEFT/RIGHT constrained decoding. This improves experimental control; it is not a claim that longer prompts or OFF universally improve ability. The historical unconstrained and sampled answers remain separate.

## T1: perfect pairs did not survive phase expansion

On the same 3,840 videos (8 SFs × 10 contrasts × 24 phases × LEFT/RIGHT), Qwen Next scored **1,951/3,840 (50.8%)**, and Nemotron **1,936/3,840 (50.4%)**. Their RIGHT answer counts were 2,781 and 3,542. Near-chance accuracy therefore does not mean unbiased guessing.

The independent earlier Next contrast-1 screen used 13 phases × two directions per SF. SF2/4/8/16/32/64 scored 18/26, 12/26, 13/26, 14/26, 16/26 and 13/26; SF0.5 scored 16/26 in another panel. The original phase-zero 2/2 answers are already inside these denominators. No candidate passed the preregistered 24/26 criterion with at least 12/13 in each direction. Changing wording gave 14/26 and 13/26 versus the original 18/26 at the same SF2 point. Phase alignment and temporal-frequency probes produced local differences, without a stable high-accuracy region.

**Lesson:** use multiple predetermined instances per configuration and report direction bias. Do not select phases after scoring, call a single correct pair a threshold, or force a monotonic curve. No new T1 full run or 16-bit bridge is authorized by the current codebase.

## T2: a reproducible transition, with exposure dependence

The old complete curve contained 6,624 answers over 23 increments; an independent three-increment supplement added 864, for 7,488 distinct historical inputs. The original baseline pairs were (background Y, initial dot Y) = (.05,.15), (.20,.30), (.40,.50), fixing dot−background at .10. Both 167 ms and 500 ms used 24 onsets and two target sides.

Each pooled point below contains 144 videos (3 baseline pairs × 24 onsets × two sides).

| Absolute increment ΔY | 167 ms correct/144 | 500 ms correct/144 |
| --- | ---: | ---: |
| .07 | 72 | 75 |
| .08 | 77 | 87 |
| .085 | 87 | 107 |
| .09 | 107 | 127 |
| .095 | 127 | 136 |
| .10 | 131 | 140 |
| .105 | 140 | 142 |
| .11 | 142 | 144 |
| .12 | 144 | 144 |

The supplement scored 739/864; its paired 500 ms results won 36 and lost 5 relative to 167 ms. Weak-signal answers were LEFT-biased. The brightening was a local, temporary event; the whole video was never globally brightened.

**Decision:** the new design fixes initial dot Y=.50 across all three backgrounds, separates nominal and encoded increments, uses 13 levels .05–.15 in .01 steps plus .20/.40, and fixes 500 ms for the main curve. A small matched 167 ms control at .08/.09/.10 tests the exposure effect. This is a new baseline policy, so old answers cannot fill its new curve. The 167 ms setting was an engineering inheritance, not a claimed universal human protocol.

## T3: angle sensitivity is real, but the renderer must be certified

The historical padded renderer was evaluated at 21 angles, 1,968 videos: angles ≤1° scored 528/1,056; 1.5° scored 43/96; 2° scored 78/96; tested 4°–90° scored 720/720. An older unpadded renderer scored 89/96 at 1° and 96/96 at 2° under matched choice decoding. These are different rasterizations, not repeated answers for identical videos. Very small old angles could collapse to identical pixels.

Linux precision validation compared analytic area integration, independent polygon clipping, CUDA supersampling and actual Psychtoolbox Screen readback (660 captures), retaining failed and negative cases. The published Manson FilterBox rasterizer was also run: on 52 shared geometries, its coverage differed from the analytic implementation by at most 2.5531e−13; their final RGB16 pixels were identical. This supports the drawing algorithm, not a universal bound on model preprocessing.

The current source-grid audit uses 52 unique geometries, including both centers and the upright reference once per geometry. Source RGB16 angle errors were: covariance/PCA MAE 0.000017345° and maximum 0.000053903°; independent blind half-plane fit MAE 0.000000990° and maximum 0.000003465°. The fit is not supplied the target angle. Analytic strip coverage and GEOS polygon clipping agreed within 2.697e−13 in this grid. These are source-image checks; the new full model curve has not run.

**Decision:** exact area rasterization, one RGB16 quantization, and independent geometric plus image-based checks. Use a fixed 13-angle grid .5/1/1.5/2/2.5/3/3.5/4/8/16/32/45/90°, balanced side/sign/onset, with 90° symmetry deduplicated.

## T4: layout and answer bias remain unresolved

With strict zero overlap over 180 frames, a regular-layout endpoint panel used 120 videos. N15 classic/photo/GPT scored 15/20, 14/20, 15/20; N30 scored 10/20, 10/20, 17/20. A 60-video irregular-layout follow-up scored 10/20, 10/20 and 15/20; changing layout did not restore a reliable high endpoint. Radius and speed diagnostics also did not recover one. An endpoint-wrap hypothesis explained a physical possibility, not the main error pattern.

The latest ordered-first/last-image experiment used N15/20/25/30 × 8 coherence levels × 24 constrained-random layouts × LEFT/RIGHT = 1,536 cases. Nemotron scored **801/1,536**, with **1,503 RIGHT** answers. At 100% coherence, the four N settings scored 30/48, 25/48, 27/48 and 28/48. These are two-image results, not full-video results. Signal-horizontal counts and noise angles must be specified; rejection for zero overlap creates conditioned randomness, not unconstrained independent positions.

**Lesson:** preserve T4 originals, distinguish correspondence/visibility from answer bias, and do not generalize an early 6/6 layout. T4 is outside the new T2/T3 production run.

## T5: mask retained, mode changes often shift bias

On the same 84 masked inputs (42 present, 42 absent), Nemotron medium scored 55/84: hit 41/42, false alarm 28/42. High/full scored 50/84: hit 41/42, false alarm 31/42 and two length failures. OFF scored 43/84 under explicit-final scoring: hit 1/42, false alarm 0/42, YES 1/NO 83. Its original strict score was 42/84 because one response had extra format text. High used a different physical placement, so this is not a fixed-host causal isolation.

The dual-event protocol kept the PsyToolkit mask, 500 ms ISI, and equal event durations; changing duration also changed SOA and remaining mask time. Separate unmasked, single-event duration diagnostics scored 7/9, 9/9, 9/9 and 9/9 at 100/250/500/1000 ms. They cannot replace the masked task or isolate a mask effect by themselves. Short-mask and prompt/mechanism probes did not establish a reliable masked threshold.

**Lesson:** report hits and false alarms, not accuracy alone. More YES responses are not evidence of restored detection. T5 replacement ideas are still proposals, not a completed result or a new production queue.

## Static controls and engineering

The first image-control batch scored T2 41/60, T3 38/38 and T4 515/956. At 100% T4 coherence, N15 image/video was 33/48 versus 34/48; N30 was 39/48 versus 25/48. This small matched control does not establish a general static/video ranking. VideoJev 4B scored 6/12; 35B scored 126/252 with every answer LEFT.

Independent video requests in batching windows 1 and 2 took 42.468 and 41.978 seconds for four inputs, about 1.17% difference. That did not justify default batching. Requests must retain independent contexts and output identities.

## Source precision and the four-model input path

Four models completed two actual GPU input-qualification calls each on the same RGB16 samples. This was eight technical checks, not eight correct answers or a complete curve. RGB48 decode and FP32 entry mapping preserved all original uint16 codes up to processor entry; official processing and later BF16 conversion have their own rounding; frame count remained 180, with no external resize. Gemma needed only 16 background rows at the bottom. Other model weights and original failures were retained.

| Source point | Nemo BF16 error | Qwen BF16 error | Cosmos BF16 error | Gemma BF16 error |
| --- | ---: | ---: | ---: | ---: |
| ΔY=.09, absolute increment error | .000924 | .001281 | .001281 | .003468 |
| ΔY=.10, absolute increment error | .001331 | .001448 | .001448 | .000034 |
| .5°, maximum image-axis error | .000279° | .000487° | .000487° | .000731° |
| 1°, maximum image-axis error | .000290° | .000250° | .000250° | .000506° |

These are measured points, not global bounds. Native file loaders decoded these 16-bit masters into uint8; at nominal ΔY=.09, decode alone produced about .094315. Our explicit RGB48 decoder prevents that avoidable early quantization. BF16 rounding remains part of the deployed model; it is value-dependent and must not be corrected by changing labels or tuning input values to model accuracy.

**Reason to retain the official processor:** normalization, patchification and token assembly define the pretrained input contract. The adapter preserves the source before that contract; it does not redesign the vision encoder. Float stimuli rather than uint8 quantization also have a low-level vision precedent in [Cai et al., CVPR 2025](https://arxiv.org/html/2502.20256v2#S3.SS1), which studied foundation image encoders and does not certify VLM BF16 precision.

## Native-input follow-up · 25 October 2026 HST

These later observations qualify the earlier rationale for preserving source precision. Same IDs and all180 source frames were used; native routes read uint8. Results are **old controlled / native default resize / native original size**:

| Selected condition | Qwen | Cosmos | Nemo |
|---|---:|---:|---:|
| T1 SF4,m=.6;48 | 32/22/21 | 23/22/23 | 24/24/25 |
| T1 SF64,m=.6;48 | 27/22/24 | 25/31/25 | 23/24/24 |
| T3 .7°;96 | 54/61/57 | 81/48/73 | 48/48/48 |
| T2 BG.05,Δ.15,10frames;48;native QSA not installed | 38/45/47 | 48/48/48 | 46/47/48 |

The T2 native Qwen rows above are explicitly historical **without custom QSA**, while T1/T3 installed the deterministic selector. The new T2 QSA version is **46/48 default resize and48/48 original size**; each arm independently repeated all48 prompt/output token sequences, answers and finish/stop states identically on the second Fourier card. The default arm changed5 answers versus the unpatched route and gained1 net correct; the original-size arm changed1 and gained1. Old unpatched answers remain a separate version. Timing on Fourier0 was3.225/7.144s and Fourier1 was3.180/7.078s, excluding each arm's first2 observed requests; original unpatchedRadon3.699/8.127s is not a same-host QSA-only timing comparison. The .7° Qwen default-resize96 was replayed in a fresh engine on another Fourier card: all96 complete prompt/output tokens, answers and finish/stop fields matched, again61/96. This supports that exact finite deployment scope, not arbitrary hardware or every input. Its61 versus57 is four more correct answers at this point, not proof that resize universally helps.

At the 0.7° T3 condition, Cosmos native original size scored 73/96 versus 48/96 with native default resize: 25 fewer correct answers, or 26.04 percentage points. The controlled route scored 81/96; its 8-answer gap against native original size is an input-route difference, not an isolated bit-depth estimate. Across tasks, Qwen T1 was 59/96 controlled versus 45/96 native original size, so the observed controlled/native-original differences cannot be ranked as universally largest for Cosmos.

Preserving high-precision source stimuli remains useful for a measured, reusable dataset. These trials do **not** show a universal VLM accuracy benefit from the complete custom input route, and do not isolate uint8 quantization from every other change. In T1 Qwen scored higher under the old route; T2 favored the native route; Cosmos T3 suffered a large default-resize drop. Timing also changes with visual-token count, host and scope. Keep source accuracy, framework deployment and model capability claims separate.

The official default-frame-sampling smoke was a different lower T2 cell (BG.4,Δ.05,30frames): Qwen34/48 with6 frames, Nemo34/48 with32, Cosmos24/48 with6, Gemma24/48 with32. It is not a same-cell6/32-versus180 experiment. At the higher full180 cell, native Gemma was24/48, with only14 historical paired IDs (9→7 correct); never infer a historical48 denominator. Gemma original-size native video is unsupported in pinned vLLM0.31 and has no new model outputs.

## Retention and reproduction

Current production source and settings: [README](README.md) and [protocol](EXPERIMENT_PLAN.md). The current and historical implementation is retained in `classic_precision/`; current full and paused formal coverage is described in README and the benchmark registry. Historical formal trials are compacted with complete fields and raw/token IDs, rather than aggregate-only replacement. The private workspace’s `archive_index.json` records retained gzip bundles and verified reconstruction. `archive/history.py` verifies and restores them into an empty destination. T4 originals and the latest precision/input evidence remain separate.

One-off smoke media, proxy frames and loose input receipts are retired when rebuildability or full-field conversion is verified. This log retains scientific findings and lessons; it is not a mirror of every failed scratch directory. Private meetings, source photographs/cutouts, model weights, environments and active website assets are protected.

Completed full paired smoke tables, summaries and pinned settings are collected in the benchmark `runs/reproducibility/smoke-results/` index and synchronized to both workstations. Original raw answers/negative records remain in their run roots. Public datasetv0.2 changes filenames and paths only; formal inference IDs and protocols are unchanged.

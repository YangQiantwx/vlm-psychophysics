<!-- gemma-x-full-current -->
**2026-10-31 06:30 HST：Fourier两卡Gemma X正式任务已实际推理；Radon正在新RAM部署资格启动。** 最新人类授权明确允许Fourier先正式跑，R资格失败时安全暂停；这取代旧“等六worker全过才启动F”的门，不冒称三卡已经复现。F四个cold共384条的完整output/prompt IDs逐项全等，均74/96，原A为67/96；证明`plans/gemma-x-provisional-20261031/fourier-gate.json`，状态PROVISIONAL_FOURIER_PASS_RADON_PENDING。原96输入、A预选依据与旧启动失败均保留。

F当前唯一正式queue为`plans/gemma-x-provisional-20261031/fourier0-full-queue.json`及fourier1对应文件，各3600；R2256保留，合计9456唯一正式ID。两queue已activate及LAUNCHED_ONCE，勿重复；run根`runs/reproducibility/gemma-x-provisional-20261031`下真实outer launch/owner/wait。06:03 HST有界快照F0/F1各61/3600、owner LIVE、各1在途、无failure；最近20普通19.109/19.383秒。两卡T1/T2/T3六个首实际GPU输入均PASS，`startup-fourier.json`，无额外调用，不重复首图。原CPU链/全帧180/有效768×512、补底16到528、budget280/实际176tokens、official auto/autotuneON/ctx65536/KV48GiB BF16/batch49152/eager/oneSeq/cache0科学设置不改。

R当前唯一queue=`plans/gemma-x-ram-qualification-20261031/execution-repair2/radon-qualification-queue.json`，SHA796175287a6912913c336a49fb1af9d60cfb9d244b80145034bfd017c383d0ea；外层prefix=`radon0-ram-qualification-exec2-queue`，已唯一启动，当前未声称首GPU或复现PASS。两worker为radon0-ram-qualification-cold1-exec2及cold2-exec2，顺序96+96。six-worker-plan SHAd03e812508b7b62d802ecd207bb16696d65906cc8ac97f3d9176242ad33c34d0保留原F四worker，只替换R两部署。官方UVA filter仅BF16 self_attn.qkv_proj.weight；cpu_offload_gb=12为上限，实际预期60层11,010,048,000B=10.25390625GiB，NVFP4/vision/KV/ctx/batch/auto不改；gpu_memory_utilization=.8只调整显式48GiB KV下的官方startup预算。GPU>=77GiB、RAM>=40GiB；本次启动实见78.96875GiB/116.900GiB。加载后及首预算内请求必须核真实host指针/参数/字节与>=1GiB可用余量；不能只认配置。逐条先落原raw，再比较完整output/prompt IDs；不一致写reproduction-failure.json并停止，随后F在请求边界pause。R资格未完成前不派2256正式。

R原4GiB候选因官方.9×94.97GiB startup门在权重加载前拒绝：0claim/0raw/0INTENT，实际worker和outer wait1、owned退出/leasefree。原记录/源码SHA9861ad19保留于implementation-history，原controls已generation+1 fence，不能恢复。官方49,152-token warmup临时张量说明4GiB余量不足；8GiB仅CPU讨论，未运行。新RAM源码SHA5f3325f985333b0062ea07bc0cdcb040a990498ce2929301e98cc880b18badd7；F provisional SHAa1811e56（完整以profile绑定为准）冻结，原四Gemma模块不改。R权重/DS/run均NVMe device66306；新run canonical指向/home/q/cvpr2027_runs_ssd/gemma-x-ram-qualification-20261031，旧大资产不重传。

另保留exec1零调用失败：实际已offload10.25GiB并加载，但旧FLASHINFER_NVCC路径缺失使官方JIT退出；worker/outer真实wait1，0claim/raw/INTENT，controls已fence。恢复该路径到本机已有CUDA13.0.88，nvcc与Fourier逐字节SHA一致（b8e2347d…），未安装/升级SDK。官方已有build.ninja经16 CPU并行81.24秒编译成功，真实returncode0，证据nvcc-path-repair.json与official-kernel-cpu-build.json。当前exec2于06:14唯一启动；06:28有界快照owner LIVE、0claim/raw、无failure，实际model_runner加载35.79秒/20.93GiB，仍初始化，首GPU验收PENDING。新gemma_x_ram_full.py（0468da76…）已CPU验证接续R2256及最终审计/比较入口；未真实六workerPASS前拒绝准备R正式draft。

CPU只读status按实际queue.profiles调用cosmos_x_full_status.inspect_profile，并以outer boot/PID/startticks核owner；Gemma资格queue无state_path，不使用通用queue_status。启动/资格每10分钟、长段每小时，平稳不通知。执行queue/supervisor必须用sealed GPU解释器及显式PYTHONPATH，不能用CPU解释器启动worker。R两cold原记录齐后用gemma_x_ram_qualification summarize核完整576/96/真实闭合/输入链与RAM residency，只有PASS_GEMMA_X_THREE_CARDS_TWO_COLD才接R正式。启动配置问题可修部署并保留失败；已claim/UNCERTAIN不得自动重派。全部9456及169cell验收后出A/X和四模型X同ID比较、网站/HF/GitHub交付，完成后停唯一cvpr-4；不新增natural/T4/T5/其它模型。
<!-- /gemma-x-full-current -->

<!-- qwen-x-remaining-t13-current -->
**2026-10-31 05:56 HST：Qwen X经典T1–T3全部9456条已完成、严格验收并发布。** T1 1918/3840，T2 2616/2880，T3 2564/2736；169个cell均有完整分母。人类明确授权只补跑重启中断的`t3_2e29ac8a4e90ee6c0f7f`，独立attempt答LEFT且正确；实际worker/outer wait0、owned退出及原lease释放、原科学配置/输入/GT/prompt/tokens审计通过。0新资格、仅1个授权新调用；原claim、228重启前缀、87恢复段、全部失败与原9455 PARTIAL保持原样，不再运行任何Qwen任务或health poll。

完整选择入口`runs/reproducibility/qwen-x-remaining-t13-20261030/classic-analysis-complete-retry1/summary.json`（SHA 5f78f51e0072030b128b65e264ad44392a23cee328b7e27a533990461a816a48）；独立单条审计`one-attempt/summary.json` SHA738c4feea0576488910986e3b33d997712b131e8c4f6ec9541fb8fe40b58192b。完整T13为原6575加这1条，`classic-analysis-complete-retry1/t13/summary.json`绑定两部分；旧11段manifest与原PARTIAL审计不改。A/X共同ID配对：T1 1924→1918但1468答案改变；T2 1872条1528→1610；T3 1584条1419→1420。Cosmos/Nemo/Qwen X共同9456视频对照和逐ID/逐cell/速度/PNG/PDF均已生成，图片实际检查。

网站https://vision-tasks.pages.dev/experiments/已发布deployment0b7d1341，线上四资源SHA及实际A/X、选线UI验收通过。HF `DrQY/vlm-psychophysics-classic-t1-t3/results/qwen-X-20261031-complete` commit10900c9c00df32a00da6ea6310cbd116d0365d62：13文件公开读回SHA/bytes一致；T3 1.3°为93/96。旧partial prefix与manifest保留。回执`classic-analysis-complete-publication.json`和`hf-results-complete-publication.json`；代码classic-precision已推103e2d407821985e931c131bfd8ed2407d7acb09。临时HF上传包已清理，正式原记录和复现代码保留。
<!-- /qwen-x-remaining-t13-current -->

<!-- five-task-research-cards-20261030 -->
## What the five tasks measure

These tasks measure specific visual judgments under controlled stimulus changes. Human psychophysics supplies testable behavioral signatures; it does not turn a VLM score into a measurement of a particular cortical area. A direct human–model comparison will use the same stimuli and response rule. Published human thresholds use their own display calibration, visual angles, viewing durations and participants.

|Task|Human-vision question and established finding|Current model evidence|Practical inference to test|
|---|---|---|---|
|T1 · Drifting gratings|How does contrast sensitivity depend on spatial frequency? [Campbell & Robson (1968)](https://doi.org/10.1113/jphysiol.1968.sp008574) measured a frequency-dependent grating-detection sensitivity: intermediate frequencies are more visible than the lowest and highest frequencies under their conditions. Our response is motion direction, which additionally requires temporal correspondence; it is not a replication of static contrast detection.|Official-input X: Cosmos 1,735/3,840; Nemotron 1,924/3,840. Their RIGHT counts are 3,427 and 3,708, respectively. This is a direction-judgment weakness with substantial response bias. Corrected natural T1 has 3,840 validated sources but no model result yet.|Fine texture being visible in an image does not establish that a video evaluator can recover its motion. Measure the direction operating range before using such judgments for generated-video criticism. Drift speed is coupled to spatial frequency rather than independently manipulated; no optimal robot speed is established.|
|T2 · Transient brightening|How does detecting a luminance increment depend on background and signal strength? [Blackwell (1946)](https://doi.org/10.1364/JOSA.36.000624) varied target size and adapting luminance and measured detection thresholds. Our digital linear-Y increments are not calibrated cd/m² or a direct test of physiological adaptation.|X classic: Cosmos 2,627/2,880; Nemotron 2,520/2,880; Qwen 2,616/2,880. Qwen natural, including the source revision, is 3,756/3,840. The natural and classic totals cover different condition counts; compare their condition curves. Natural appearance and baseline luminance also differ.|A critic or monitor can be checked for the smallest reproducible transient it detects under each scene condition. Natural-scene success supports that task range, not general illumination invariance or a human-superiority claim.|
|T3 · Orientation comparison|How precisely can orientation differences be discriminated? [Girshick, Landy & Simoncelli (2011)](https://pmc.ncbi.nlm.nih.gov/articles/PMC3125404/) found orientation-dependent discrimination and perceptual biases toward cardinal axes. The present upright-reference task measures geometric orientation sensitivity; it does not sweep reference orientation to establish the oblique effect.|X: Cosmos 2,625/2,736; Nemotron 1,998/2,736. Cosmos is 78/96 at 0.7°; a fine-angle result must retain side/sign/onset balance and actual rendered geometry. Revised natural T3 has 2,736 validated sources and no model inference yet.|Fine planar orientation is a candidate strength for geometric comparison or visual criticism. Test actual downstream geometry and tolerance before claiming 3D calibration, pose estimation, or physical-plausibility judgment.|
|T4 · Direction discrimination|How are local motion signals pooled in noise? Higher coherence generally makes the global direction easier for people to judge. Exact performance also depends on the random-dot algorithm, contrast, duration and displacement: [Pilly & Seitz (2009)](https://doi.org/10.1016/j.visres.2009.03.019) showed that these implementation choices can materially change human performance.|Final Cosmos X: classic 190/384 and natural 201/384 direction answers correct. RIGHT occurs 406/408 and 397/408 times, respectively, including zero controls. A separate metadata-only final-centroid rule gets 360/384 classic labels correct; it is not a VLM score. These biased curves do not identify a threshold.|The assay reports direction discrimination. Final spatial configuration already predicts most labels, so accuracy alone does not establish cross-frame integration. Natural cutouts share a 56 px support, not equal visible area or contrast.|
|T5 · Sequential target detection|Is the reference target detected among two successive visual events, and does a mask alter detection? [PsyToolkit's AB implementation](https://www.psytoolkit.org/experiment-library/ab.html) follows a human sequential-target paradigm. Human masking depends on target/mask timing and structure, as demonstrated by [Agaoglu et al. (2015)](https://doi.org/10.1016/j.visres.2015.07.003). Fixed 500 ms exposures and a fixed 500 ms ISI do not measure an attentional-blink time course.|Final Cosmos X: classic 100/192 and natural 99/192 correct, with NO on 178/192 and 189/192 trials. Present-target hits are 9/96 and 3/96; false alarms are 5/96 and 0/96. Even the no-mask baseline is strongly NO-biased; an independent mask cost has not been identified.|Report hits, false alarms and response bias separately. Masked hit reductions coincide with false-alarm reductions and near-floor baselines; they do not isolate perceptual sensitivity or attention. First/second contrasts additionally mix spatial axis and recency, and do not identify pure order, dwell-time or temporal-integration effects.|

### Compact experiment cards

All current source videos use 768×512 pixels, 60 fps and 180 frames (3 s). These are source specifications; each API or model input route must record its actual sampling and processing. Scientific conditions and responses are retained in analysis metadata and withheld from the human response interface.

|Task|Variable shown on the card|Fixed or balanced structure|Current scope|
|---|---|---|---|
|T1|Spatial frequency × modulation contrast|8 frequencies × 10 contrasts; 24 phases × LEFT/RIGHT per cell|3,840 classic; 3,840 corrected natural|
|T2|Absolute ΔY, grouped by background/source|500 ms event; 24 onset times × LEFT/RIGHT = 48 per cell|20 increments × 3 classic backgrounds = 2,880; 20 × 4 natural source groups = 3,840|
|T3|Angular difference|500 ms event; side, sign and onset balanced; 90° symmetry deduplicated|28 non-90° levels × 96 + 48 at 90° = 2,736 per stimulus family|
|T4|Coherence; 0% shown separately|16 dots; diameter 56 px; 8 nonzero levels × 24 seeds × LEFT/RIGHT; paired natural assets|384 direction trials + 24 zero-coherence controls = 408 per family|
|T5|Mask ON/OFF × target first/second/absent|Always two 500 ms events; offset-to-next-onset ISI 500 ms; matched timing/position and natural assets|24 first + 24 second + 48 absent, each × 2 mask states = 192 per family|

The strongest supported story is a dissociation between task-specific fine spatial/photometric judgments and motion-direction judgments in the tested deployments. It is a behavioral profile of the evaluated VLM systems. The benchmark has not yet established human-equivalent mechanisms, world-model simulation fidelity, or robot-action success.
<!-- /five-task-research-cards-20261030 -->

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

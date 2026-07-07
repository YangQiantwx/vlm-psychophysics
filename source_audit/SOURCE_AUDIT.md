# Source Audit

This file keeps the web demo honest. The React page is a preview tool for the proposal, not a completed benchmark runner. When a usable web package exists, the package is installed in this app and the visible controls map to the package's trial or stimulus field names.

## Moving Stripes

- Reader link: https://jspsychophysics.hes.kyushu-u.ac.jp/demo_explanation/
- Local source: `source_audit/jspsych-psychophysics/drifting-gabor.html`
- Installed package: `@kurokida/jspsych-psychophysics` 5.1.0, MIT license, npm tarball `https://registry.npmjs.org/@kurokida/jspsych-psychophysics/-/jspsych-psychophysics-5.1.0.tgz`
- What the source supports: a drifting Gabor stimulus with `tilt`, `sf`, `phase`, `contrast`, and `drift`.
- What the demo does: renders a Gabor preview from those official fields. Contrast, stripe count, and speed use draggable bars; direction uses a left/right button choice.
- Boundary: the website preview is not a full jsPsych runner. The benchmark generator should call the plugin or the downloaded demo code directly.

## Brief Flash

- Reader link: https://www.ncbi.nlm.nih.gov/books/NBK11525/
- Local source: `source_audit/jspsych-psychophysics/keyboard_event.html`
- What the source supports: Webvision describes increment-threshold light-adaptation experiments; the jspsych-psychophysics demo shows a luminance-controlled circle in a browser canvas with `radius`, `fill_color`, and updateable color.
- What the demo does: renders a brief brighter spot on a controllable gray background using those circle field meanings. This is a screen-value preview, not a calibrated luminance experiment.
- Boundary: Webvision is the task definition source, not runnable code. The browser preview uses the jspsych-psychophysics luminance-circle example as the closest runnable web source.

## Tilted Lines

- Reader link: https://psychopy.org/coder/tutorial2.html
- Local source: `source_audit/psychopy/tutorial2_orientation_jnd.py`
- What the source supports: a PsychoPy orientation JND task using two `GratingStim` patches, a staircase over angular increment, 500 ms presentation, and left/right response.
- What the demo does: renders two Gabor patches using `tilt`, `sf`, `contrast`, and `width` field meanings. Different side uses a left/right button; base angle, angle gap, contrast, and show time use draggable bars.
- Boundary: the page is a preview of the visual stimulus. A real human or VLM batch run should preserve the source task's staircase or fixed-grid condition logic.

## Moving Dots

- Reader link: https://www.jspsych.org/6.3/plugins/jspsych-rdk/
- Local source: `source_audit/jspsych-rdk/index.ts`
- Installed package: `@jspsych-contrib/plugin-rdk` 2.0.0, MIT license, npm tarball `https://registry.npmjs.org/@jspsych-contrib/plugin-rdk/-/plugin-rdk-2.0.0.tgz`
- What the source supports: a random-dot kinematogram with parameters including `number_of_dots`, `coherent_direction`, `coherence`, `dot_radius`, and `move_distance`.
- What the demo does: renders a circular random-dot preview from those official fields. Shared motion, dot step, dot count, and dot size use draggable bars; direction uses button choices. Direction follows the plugin convention: 0 right, 90 up, 180 left, 270 down.
- Boundary: the website preview uses the same parameter meanings. The benchmark generator should call the plugin for official RDK trials.

## Attentional Blink

- Reader link: https://www.psytoolkit.org/experiment-library/ab.html
- Local source: `../../official_source/ab_unzipped/ab.psy`, `../../official_source/ab_unzipped/ab-table.r`, and `../public/official-ab/*.png`
- What the source supports: official PsyToolkit AB task assets, 50 ms stimulus presentations, masks, and a condition table generated from `times=c(0,100,196,280,430,590,900)`.
- What the demo does: uses the official PNG assets. The 7 official gaps use an ordered draggable bar; target timing, target presence, and locations use button choices.
- Boundary: this is the strongest direct-source panel. The preview uses the same official stimulus PNGs and timing values visible in the source task file.

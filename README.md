# VLM psychophysics probe demo

Interactive proposal page for showing what each psychophysics video probe would look like to a VLM.

The page uses source material where available. The visible controls map to stimulus or trial fields from jsPsych psychophysics, jsPsych RDK, PsychoPy, Webvision, and PsyToolkit sources. The Attentional Blink panel uses the official PsyToolkit PNG assets included in `public/official-ab/`.

This is a stimulus preview page, not a finished benchmark runner. Model results, private notes, emails, and local screenshots are not part of the deployed site.

## Run

```bash
npm install
npm run dev -- --port 5177
```

Open `http://127.0.0.1:5177/`.

## Tasks shown

- Moving stripes: ask whether stripes are visible and which way they move.
- Brief flash: ask whether a small bright spot appeared on a light-adaptation background.
- Tilted lines: ask which side has the different angle between two Gabor patches.
- Moving dots: ask which way the random-dot motion field moves overall.
- Attentional Blink: ask whether the letter L appeared. This panel uses the official PsyToolkit AB source assets and the 7 gap values from the official condition table.

Each task shows one source link, the human response format, the VLM response format, draggable parameter bars, and a canvas video with playback speed control.

## Release check

```bash
npm run check:release
```

The release check builds the static site, verifies deployment headers are present, checks for obvious secret patterns, and blocks public source maps.

## Deploy

Cloudflare Pages can serve the generated static site:

```bash
npm run check:release
npx wrangler pages project create vlm-psychophysics
npx wrangler pages deploy dist --project-name vlm-psychophysics --branch main
```

The public Pages URL is expected to be `https://vlm-psychophysics.pages.dev`.

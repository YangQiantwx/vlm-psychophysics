import React, { useEffect, useRef, useState } from "react";
import {
  buildLightCircleSpec,
  buildMovingGaborSpec,
  buildOrientationGaborSpec,
  buildRdkSpec,
} from "./officialStimulusSpecs";
import {
  AB_IMAGE_PATHS,
  CANVAS_HEIGHT,
  CANVAS_WIDTH,
  initParams,
  playbackOptions,
  taskMap,
  tasks,
} from "./tasks";

const FRAME_INTERVAL_MS = 1000 / 30;

function App() {
  const [activeTaskId, setActiveTaskId] = useState(tasks[0].id);
  const [paramsByTask, setParamsByTask] = useState(initParams);
  const [isPlaying, setIsPlaying] = useState(true);
  const [playbackRate, setPlaybackRate] = useState(0.5);
  const [restartSignal, setRestartSignal] = useState(0);
  const [frameInfo, setFrameInfo] = useState({ time: 0, frame: 0, duration: 3 });

  const activeTask = taskMap.get(activeTaskId);
  const params = paramsByTask[activeTaskId];

  const updateParam = (id, value) => {
    setParamsByTask((current) => ({
      ...current,
      [activeTaskId]: {
        ...current[activeTaskId],
        [id]: value,
      },
    }));
  };

  const resetTask = () => {
    setParamsByTask((current) => ({
      ...current,
      [activeTaskId]: activeTask.defaults,
    }));
    setRestartSignal((value) => value + 1);
  };

  return (
    <main className="app-shell">
      <header className="topbar">
        <a className="brand-mark" href="#top" aria-label="VLM video probe demo">
          <span className="brand-dot" />
          <span>VLM Video Probe Demo</span>
        </a>
        <div className="topbar-meta">
          <span>CVPR 2027 proposal demo</span>
          <a href={activeTask.sourceUrl} target="_blank" rel="noreferrer">
            Source
          </a>
        </div>
      </header>

      <section className="workspace" id="top">
        <nav className="task-rail" aria-label="Video tasks">
          {tasks.map((task) => (
            <button
              key={task.id}
              className={task.id === activeTaskId ? "task-tab active" : "task-tab"}
              type="button"
              onClick={() => {
                setActiveTaskId(task.id);
                setRestartSignal((value) => value + 1);
              }}
            >
              <span>{task.index}</span>
              <strong>{task.title}</strong>
              <em>{task.label}</em>
            </button>
          ))}
        </nav>

        <section className="stimulus-stage" aria-label={`${activeTask.title} video preview`}>
          <div className="stage-heading">
            <div>
              <span className="eyebrow">What the VLM sees</span>
              <h1>{activeTask.title}</h1>
            </div>
            <div className="frame-readout">
              <span>{frameInfo.frame.toString().padStart(3, "0")}</span>
              <b>{frameInfo.time.toFixed(2)}s</b>
            </div>
          </div>

          <CanvasStimulus
            task={activeTask}
            params={params}
            isPlaying={isPlaying}
            playbackRate={playbackRate}
            restartSignal={restartSignal}
            onFrame={setFrameInfo}
          />

          <div className="transport">
            <button type="button" onClick={() => setIsPlaying((value) => !value)}>
              {isPlaying ? "Pause" : "Play"}
            </button>
            <button type="button" onClick={() => setRestartSignal((value) => value + 1)}>
              Restart
            </button>
            <div className="speed-row" aria-label="Playback speed">
              {playbackOptions.map(({ rate, label }) => (
                <button
                  key={rate}
                  className={playbackRate === rate ? "active" : ""}
                  type="button"
                  onClick={() => setPlaybackRate(rate)}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>
        </section>

        <aside className="inspector" aria-label={`${activeTask.title} controls`}>
          <div className="task-copy">
            <span className="eyebrow">{activeTask.sourceLabel}</span>
            <h2>{activeTask.label}</h2>
            <p>{activeTask.task}</p>
            <a className="source-link" href={activeTask.sourceUrl} target="_blank" rel="noreferrer">
              Open source
            </a>
          </div>

          <div className="control-list">
            {activeTask.controls.map((control) => (
              <ParameterControl
                key={control.id}
                control={control}
                value={params[control.id]}
                onChange={(value) => updateParam(control.id, value)}
              />
            ))}
          </div>

          <button className="reset-button" type="button" onClick={resetTask}>
            Reset task defaults
          </button>

          <div className="readout-block">
            <span>Response</span>
            <p>{activeTask.humanAsk}</p>
            <p>{activeTask.vlmAsk}</p>
            <small>{activeTask.measure}</small>
          </div>

          <div className="answer-block">
            <span>Task source</span>
            {activeTask.safeNotes.map((note) => (
              <p key={note}>{note}</p>
            ))}
          </div>
        </aside>
      </section>
    </main>
  );
}

function ParameterControl({ control, value, onChange }) {
  if (control.type === "buttonGroup") {
    const selected = control.options.find((option) => option.value === value);
    return (
      <div className="control-block">
        <div className="control-label">
          <span>{control.label}</span>
          <b>{selected?.label}</b>
        </div>
        <div className="choice-buttons" role="group" aria-label={control.label}>
          {control.options.map((option) => (
            <button
              key={option.label}
              className={option.value === value ? "active" : ""}
              type="button"
              onClick={() => onChange(option.value)}
            >
              {option.label}
            </button>
          ))}
        </div>
        {control.hint ? <small className="control-hint">{control.hint}</small> : null}
      </div>
    );
  }

  const isIndexed = control.type === "indexedRange";
  const options = control.options || [];
  const selectedIndex = Math.max(0, options.findIndex((option) => option.value === value));
  const sliderValue = isIndexed ? selectedIndex : value;
  const min = isIndexed ? 0 : control.min;
  const max = isIndexed ? options.length - 1 : control.max;
  const step = isIndexed ? 1 : control.step;
  const display = isIndexed ? options[selectedIndex]?.label : formatControlValue(control, value);
  const minLabel = isIndexed
    ? options[0]?.label
    : formatEndpointValue(control, control.min, control.minLabel);
  const maxLabel = isIndexed
    ? options[options.length - 1]?.label
    : formatEndpointValue(control, control.max, control.maxLabel);
  const fillRatio = max === min ? 0 : (sliderValue - min) / (max - min);
  const fillPercent = Math.min(100, Math.max(0, fillRatio * 100));
  const gradientId = `range-gradient-${control.id}`;

  return (
    <label className="control-block range-block">
      <div className="control-label">
        <span>{control.label}</span>
        <b>{display}</b>
      </div>
      <div className="range-slider">
        <svg
          className="range-track-visual"
          viewBox="0 0 100 7"
          preserveAspectRatio="none"
          aria-hidden="true"
          focusable="false"
        >
          <defs>
            <linearGradient id={gradientId} x1="0" x2="1" y1="0" y2="0">
              <stop offset="0%" stopColor="rgba(86, 163, 255, 0.95)" />
              <stop offset="100%" stopColor="rgba(183, 239, 109, 0.9)" />
            </linearGradient>
          </defs>
          <rect className="range-track-bg" x="0" y="0" width="100" height="7" rx="3.5" />
          {fillPercent > 0 ? (
            <rect
              className="range-track-fill"
              x="0"
              y="0"
              width={fillPercent}
              height="7"
              rx="3.5"
              fill={`url(#${gradientId})`}
            />
          ) : null}
          <rect className="range-track-border" x="0.5" y="0.5" width="99" height="6" rx="3" />
        </svg>
        <input
          className="range-input"
          type="range"
          min={min}
          max={max}
          step={step}
          value={sliderValue}
          aria-label={control.label}
          onChange={(event) => {
            const next = Number(event.target.value);
            onChange(isIndexed ? options[next].value : next);
          }}
        />
      </div>
      <div className="range-scale" aria-hidden="true">
        <span>{minLabel}</span>
        <span>{maxLabel}</span>
      </div>
      {control.hint ? <small className="control-hint">{control.hint}</small> : null}
    </label>
  );
}

function formatControlValue(control, value) {
  if (control.format === "percent") return `${Math.round(value * 100)}%`;
  const decimals = control.decimals ?? (Number.isInteger(value) ? 0 : 2);
  const number = Number(value).toFixed(decimals);
  return `${number}${control.unit ? ` ${control.unit}` : ""}`;
}

function formatEndpointValue(control, value, label) {
  const formatted = formatControlValue(control, value);
  return label ? `${formatted} ${label}` : formatted;
}

function CanvasStimulus({ task, params, isPlaying, playbackRate, restartSignal, onFrame }) {
  const canvasRef = useRef(null);
  const timeRef = useRef(0);
  const lastFrameRef = useRef(0);
  const lastPaintRef = useRef(0);
  const rafRef = useRef(0);
  const reportRef = useRef(0);
  const abImages = useImageSet(AB_IMAGE_PATHS);

  useEffect(() => {
    timeRef.current = 0;
    lastFrameRef.current = 0;
    lastPaintRef.current = 0;
  }, [task.id, restartSignal]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return undefined;
    const ctx = canvas.getContext("2d", { alpha: false });
    const draw = (now) => {
      if (lastPaintRef.current && now - lastPaintRef.current < FRAME_INTERVAL_MS) {
        rafRef.current = requestAnimationFrame(draw);
        return;
      }

      if (!lastFrameRef.current) lastFrameRef.current = now;
      const delta = Math.min(0.08, (now - lastFrameRef.current) / 1000);
      lastFrameRef.current = now;
      lastPaintRef.current = now;

      const duration = getDuration(task.id, params);
      if (isPlaying) {
        timeRef.current = (timeRef.current + delta * playbackRate) % duration;
      }

      drawStimulus(ctx, task.id, params, timeRef.current, duration, abImages);

      if (now - reportRef.current > 80) {
        reportRef.current = now;
        onFrame({
          time: timeRef.current,
          frame: Math.floor(timeRef.current * 30),
          duration,
        });
      }

      rafRef.current = requestAnimationFrame(draw);
    };

    rafRef.current = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(rafRef.current);
  }, [task.id, params, isPlaying, playbackRate, onFrame, abImages]);

  return (
    <div className="canvas-wrap">
      <canvas
        ref={canvasRef}
        width={CANVAS_WIDTH}
        height={CANVAS_HEIGHT}
        aria-label={`${task.title} generated video`}
      />
    </div>
  );
}

function useImageSet(paths) {
  const [images, setImages] = useState({});

  useEffect(() => {
    let cancelled = false;
    const entries = Object.entries(paths);
    Promise.all(
      entries.map(
        ([key, src]) =>
          new Promise((resolve) => {
            const image = new Image();
            image.onload = () => resolve([key, image]);
            image.onerror = () => resolve([key, null]);
            image.src = src;
          })
      )
    ).then((loaded) => {
      if (cancelled) return;
      setImages(Object.fromEntries(loaded.filter(([, image]) => image)));
    });

    return () => {
      cancelled = true;
    };
  }, [paths]);

  return images;
}

function getDuration(taskId, params) {
  if (taskId === "ab") {
    return 1.05 + params.targetDurationMs / 1000 + params.lagMs / 1000 + params.targetDurationMs / 1000 + 0.72;
  }
  if (taskId === "tvi") return Math.max(2.6, params.flashOnset + params.flashDuration + 0.72);
  if (taskId === "orientation") return Math.max(2.1, params.duration + 0.95);
  return 3.2;
}

function drawStimulus(ctx, taskId, params, t, duration, abImages) {
  if (taskId === "stcsf") drawGrating(ctx, params, t);
  if (taskId === "tvi") drawLightAdaptation(ctx, params, t);
  if (taskId === "orientation") drawOrientation(ctx, params, t, duration);
  if (taskId === "rdk") drawRdk(ctx, params, t);
  if (taskId === "ab") drawAttentionalBlink(ctx, params, t, abImages);
}

function fill(ctx, value) {
  const v = clamp(Math.round(value * 255), 0, 255);
  ctx.fillStyle = `rgb(${v}, ${v}, ${v})`;
  ctx.fillRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
}

function drawFixation(ctx, color = "rgba(255,255,255,0.62)") {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(CANVAS_WIDTH / 2 - 12, CANVAS_HEIGHT / 2);
  ctx.lineTo(CANVAS_WIDTH / 2 + 12, CANVAS_HEIGHT / 2);
  ctx.moveTo(CANVAS_WIDTH / 2, CANVAS_HEIGHT / 2 - 12);
  ctx.lineTo(CANVAS_WIDTH / 2, CANVAS_HEIGHT / 2 + 12);
  ctx.stroke();
  ctx.restore();
}

function drawGrating(ctx, params, t) {
  const spec = buildMovingGaborSpec(params).object;
  const image = ctx.createImageData(CANVAS_WIDTH, CANVAS_HEIGHT);
  const data = image.data;
  const phaseRad = (spec.phase * Math.PI) / 180 + spec.drift * t * Math.PI * 2;
  const sigma = spec.sc;
  for (let y = 0; y < CANVAS_HEIGHT; y += 1) {
    for (let x = 0; x < CANVAS_WIDTH; x += 1) {
      const dx = x - CANVAS_WIDTH / 2;
      const dy = y - CANVAS_HEIGHT / 2;
      const stripe = Math.sin(2 * Math.PI * spec.sf * dx - phaseRad);
      const envelope = Math.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma));
      const value = clamp(0.5 + stripe * spec.contrast * 0.5 * envelope, 0, 1);
      const i = (y * CANVAS_WIDTH + x) * 4;
      const v = Math.round(value * 255);
      data[i] = v;
      data[i + 1] = v;
      data[i + 2] = v;
      data[i + 3] = 255;
    }
  }
  ctx.putImageData(image, 0, 0);
  drawFixation(ctx, "rgba(0,0,0,0.22)");
}

function drawLightAdaptation(ctx, params, t) {
  const spec = buildLightCircleSpec(params).object;
  fill(ctx, params.background);
  drawFixation(ctx, params.background > 0.45 ? "rgba(0,0,0,0.35)" : "rgba(255,255,255,0.4)");
  const inFlash = t * 1000 >= spec.show_start_time && t * 1000 <= spec.show_end_time;
  if (!inFlash) return;

  const value = clamp(params.background + params.increment, 0, 1);
  const radius = spec.radius;
  const gradient = ctx.createRadialGradient(
    CANVAS_WIDTH / 2,
    CANVAS_HEIGHT / 2,
    radius * 0.15,
    CANVAS_WIDTH / 2,
    CANVAS_HEIGHT / 2,
    radius
  );
  const v = Math.round(value * 255);
  const bg = Math.round(params.background * 255);
  gradient.addColorStop(0, `rgb(${v}, ${v}, ${v})`);
  gradient.addColorStop(0.72, `rgb(${v}, ${v}, ${v})`);
  gradient.addColorStop(1, `rgb(${bg}, ${bg}, ${bg})`);
  ctx.fillStyle = gradient;
  ctx.beginPath();
  ctx.arc(CANVAS_WIDTH / 2, CANVAS_HEIGHT / 2, radius, 0, Math.PI * 2);
  ctx.fill();
}

function drawOrientation(ctx, params, t, duration) {
  fill(ctx, 0.48);
  const spec = buildOrientationGaborSpec(params);
  const [leftGabor, rightGabor] = spec.objects;
  const visible = t <= params.duration;
  if (visible) {
    drawGabor(ctx, leftGabor);
    drawGabor(ctx, rightGabor);
  } else {
    drawNoiseMask(ctx, params.maskStrength, t);
  }
  drawFixation(ctx, "rgba(255,255,255,0.38)");
  if (t > duration - 0.45) {
    ctx.fillStyle = "rgba(122, 134, 150, 0.22)";
    ctx.fillRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
  }
}

function drawGabor(ctx, spec) {
  const size = spec.width;
  const sigma = spec.sc;
  const frequency = spec.sf;
  const angle = (spec.tilt * Math.PI) / 180;
  const phase = ((spec.phase || 0) * Math.PI) / 180;
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  const image = ctx.createImageData(size, size);
  const data = image.data;

  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      const dx = x - size / 2;
      const dy = y - size / 2;
      const rotated = dx * cos + dy * sin;
      const envelope = Math.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma));
      const grating = Math.sin(2 * Math.PI * frequency * rotated + phase);
      const value = clamp(0.5 + grating * spec.contrast * 0.5 * envelope, 0, 1);
      const alpha = envelope > 0.02 ? 255 : 0;
      const i = (y * size + x) * 4;
      const v = Math.round(value * 255);
      data[i] = v;
      data[i + 1] = v;
      data[i + 2] = v;
      data[i + 3] = alpha;
    }
  }
  ctx.putImageData(image, Math.round(spec.startX - size / 2), Math.round(spec.startY - size / 2));
}

function drawNoiseMask(ctx, strength, t) {
  const cell = 8;
  for (let y = 0; y < CANVAS_HEIGHT; y += cell) {
    for (let x = 0; x < CANVAS_WIDTH; x += cell) {
      const r = hash(Math.floor(x / cell) * 37 + Math.floor(y / cell) * 91 + Math.floor(t * 20) * 13);
      const v = Math.round((0.5 + (r - 0.5) * strength) * 255);
      ctx.fillStyle = `rgb(${v}, ${v}, ${v})`;
      ctx.fillRect(x, y, cell, cell);
    }
  }
}

function drawRdk(ctx, params, t) {
  const spec = buildRdkSpec(params).trial;
  fill(ctx, 0.13);
  const cx = CANVAS_WIDTH / 2;
  const cy = CANVAS_HEIGHT / 2;
  const radius = spec.aperture_width / 2;
  const angle = (spec.coherent_direction * Math.PI) / 180;
  const frame = Math.floor(t * 60);

  ctx.save();
  ctx.beginPath();
  ctx.arc(cx, cy, radius, 0, Math.PI * 2);
  ctx.clip();
  ctx.fillStyle = "#e9edf3";
  for (let i = 0; i < spec.number_of_dots; i += 1) {
    const coherent = hash(i * 101 + 19) < spec.coherence;
    const seedX = hash(i * 17 + 11);
    const seedY = hash(i * 31 + 7);
    const localAngle = coherent ? angle : hash(i * 47 + 13) * Math.PI * 2;
    const drift = spec.move_distance * frame;
    const baseX = (seedX - 0.5) * radius * 2;
    const baseY = (seedY - 0.5) * radius * 2;
    const x = wrapSigned(baseX + Math.cos(localAngle) * drift, radius);
    const y = wrapSigned(baseY - Math.sin(localAngle) * drift, radius);
    if (x * x + y * y > radius * radius) continue;
    ctx.beginPath();
    ctx.arc(cx + x, cy + y, spec.dot_radius, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();

  ctx.strokeStyle = "rgba(255,255,255,0.18)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(cx, cy, radius, 0, Math.PI * 2);
  ctx.stroke();
}

function drawAttentionalBlink(ctx, params, t, images) {
  fill(ctx, 0);
  drawOfficialAbBase(ctx, images);

  const fixationEnd = 1.05;
  const stimDur = params.targetDurationMs / 1000;
  const lag = params.lagMs / 1000;
  const s1Start = fixationEnd;
  const s1End = s1Start + stimDur;
  const s2Start = s1End + lag;
  const s2End = s2Start + stimDur;

  if (t < s1Start) return;

  const firstPos = officialAbPosition(1, params);
  const secondPos = officialAbPosition(2, params);
  const firstStim = getAbStimulusName(1, params);
  const secondStim = getAbStimulusName(2, params);

  if (t >= s1Start && t < s1End) {
    drawOfficialImage(ctx, images[firstStim], firstPos.x, firstPos.y, 1.2, () => drawAbStimulus(ctx, 1, params));
    return;
  }

  drawOfficialImage(ctx, images.mask, firstPos.x, firstPos.y, 1.2, () => drawAbMask(ctx, 1, params));

  if (t < s2Start) return;

  if (t >= s2Start && t < s2End) {
    drawOfficialImage(ctx, images[secondStim], secondPos.x, secondPos.y, 1.2, () => drawAbStimulus(ctx, 2, params));
    return;
  }

  drawOfficialImage(ctx, images.mask, secondPos.x, secondPos.y, 1.2, () => drawAbMask(ctx, 2, params));
}

function drawOfficialAbBase(ctx, images) {
  const positions = [
    officialAbPosition("left"),
    officialAbPosition("right"),
    officialAbPosition("top"),
    officialAbPosition("bottom"),
  ];
  positions.forEach(({ x, y }) => {
    drawOfficialImage(ctx, images.box, x, y, 1.2, () => drawAbBoxFallback(ctx, x, y));
  });
  drawOfficialImage(ctx, images.fix, CANVAS_WIDTH / 2, CANVAS_HEIGHT / 2, 1.2, () => drawFixation(ctx));
}

function officialAbPosition(slot, params = {}) {
  const cx = CANVAS_WIDTH / 2;
  const cy = CANVAS_HEIGHT / 2;
  const offsetX = 180;
  const offsetY = 150;
  if (slot === 1) return { x: cx + offsetX * (params.firstLocation ?? -1), y: cy };
  if (slot === 2) return { x: cx, y: cy + offsetY * (params.secondLocation ?? -1) };
  if (slot === "left") return { x: cx - offsetX, y: cy };
  if (slot === "right") return { x: cx + offsetX, y: cy };
  if (slot === "top") return { x: cx, y: cy - offsetY };
  return { x: cx, y: cy + offsetY };
}

function drawOfficialImage(ctx, image, x, y, scale, fallback) {
  if (!image) {
    fallback();
    return;
  }
  const width = image.width * scale;
  const height = image.height * scale;
  ctx.drawImage(image, x - width / 2, y - height / 2, width, height);
}

function drawAbBoxFallback(ctx, x, y) {
  ctx.save();
  ctx.strokeStyle = "rgba(255,255,255,0.45)";
  ctx.lineWidth = 3;
  ctx.strokeRect(x - 52, y - 52, 104, 104);
  ctx.restore();
}

function getAbStimulusName(slot, params) {
  if (params.targetPresent === 1 && params.targetSlot === slot) return "t1";
  if (slot === 1) return params.targetPresent === 1 ? "n2" : "n1";
  return params.targetPresent === 1 ? "n1" : "n3";
}

function drawAbStimulus(ctx, slot, params) {
  const visibleTarget = params.targetPresent === 1 && params.targetSlot === slot;
  const position = officialAbPosition(slot, params);
  if (visibleTarget) {
    drawLShape(ctx, position.x, position.y);
  } else {
    drawDistractor(ctx, position.x, position.y, slot);
  }
}

function drawLShape(ctx, x, y) {
  ctx.save();
  ctx.strokeStyle = "#111827";
  ctx.lineWidth = 16;
  ctx.lineCap = "square";
  ctx.beginPath();
  ctx.moveTo(x - 18, y - 34);
  ctx.lineTo(x - 18, y + 30);
  ctx.lineTo(x + 34, y + 30);
  ctx.stroke();
  ctx.restore();
}

function drawDistractor(ctx, x, y, slot) {
  ctx.save();
  ctx.strokeStyle = "#111827";
  ctx.lineWidth = 14;
  ctx.lineCap = "square";
  ctx.beginPath();
  if (slot === 1) {
    ctx.moveTo(x - 34, y - 30);
    ctx.lineTo(x + 34, y + 30);
    ctx.moveTo(x + 34, y - 30);
    ctx.lineTo(x - 34, y + 30);
  } else {
    ctx.moveTo(x - 34, y - 30);
    ctx.lineTo(x + 34, y - 30);
    ctx.moveTo(x, y - 30);
    ctx.lineTo(x, y + 34);
  }
  ctx.stroke();
  ctx.restore();
}

function drawAbMask(ctx, slot, params = {}) {
  const position = officialAbPosition(slot, params);
  ctx.save();
  ctx.translate(position.x, position.y);
  ctx.strokeStyle = "#111827";
  ctx.lineWidth = 7;
  for (let i = -44; i <= 44; i += 16) {
    ctx.beginPath();
    ctx.moveTo(-48, i);
    ctx.lineTo(48, -i);
    ctx.moveTo(i, -48);
    ctx.lineTo(-i, 48);
    ctx.stroke();
  }
  ctx.strokeRect(-50, -50, 100, 100);
  ctx.restore();
}

function hash(seed) {
  const x = Math.sin(seed * 12.9898) * 43758.5453;
  return x - Math.floor(x);
}

function wrapSigned(value, radius) {
  const span = radius * 2;
  let wrapped = ((value + radius) % span + span) % span;
  return wrapped - radius;
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

export default App;

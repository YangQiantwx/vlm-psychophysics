"""Regenerate the submitted figures from verified release inputs."""

from collections import defaultdict
import json
import subprocess
import sys

from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from .plot_style import (ACCURACY_COLORS, COLORS, INK, LABELS, MARKERS, MUTED,
                         WIDTH, axes_style, cvpr_font_families, percent_axis, plt)
from .scores import factors, read_json, require, sha256, sustained75 as stable75

MODELS = ("cosmos", "nemo", "qwen", "gemma")
SCENES = ("classic", "natural")
COLUMN = 3.35
CRITERION = "#687c8d"
ABOVE = "#f1f6f7"
TRACK = "#dce5eb"


def name(model):
    return LABELS["nemotron" if model == "nemo" else model]


def style(model):
    key = "nemotron" if model == "nemo" else model
    return COLORS[key], MARKERS[key]


def measurements(rows):
    points = sorted((float(level), correct/n) for level, (n, correct) in rows.items() if level != "all")
    levels, accuracy = zip(*points, strict=True)
    return np.asarray(levels, dtype=float), np.asarray(accuracy)


def curve(ax, model, x, accuracy, small=False):
    color, marker = style(model)
    ax.plot(x, accuracy, color=color, marker=marker, lw=.95 if small else 1.10,
            ms=1.85 if small else 2.35, mew=.45, mfc="white", zorder=3)


def onset(ax, model, x, y, size=3.8):
    color, marker = style(model)
    ax.plot(x, y, color=color, marker=marker, ls="none", ms=size, mfc=color,
            mew=.5, mec="white", zorder=5, clip_on=False)


def accuracy_scale(ax, lower):
    percent_axis(ax, (lower, 1.026), (.5, .75, 1.0))
    ax.axhline(.75, color=CRITERION, linestyle=(0, (3, 2.1)), linewidth=.8, zorder=2)


def tilt_position(angle):
    """Join linear 0.5–2.5° and logarithmic 2.5–90° scales continuously."""
    value = np.asarray(angle, dtype=float)
    return np.where(value <= 2.5, .57 * (value - .5) / 2,
                    .57 + .43 * np.log(value / 2.5) / np.log(90 / 2.5))


def plot_orientation(data):
    fig = plt.figure(figsize=(WIDTH, 2.22))
    layout = fig.add_gridspec(2, 2, left=.148, right=.981, bottom=.140, top=.845,
                              height_ratios=(2.53, 1.0), hspace=.29, wspace=.166)
    ticks = (.5, 1, 1.5, 2, 2.5, 4, 8, 16, 32, 90)
    for col, scene in enumerate(SCENES):
        accuracy, range_ax = (fig.add_subplot(layout[r, col]) for r in range(2))
        thresholds = {m: stable75(data[m, scene, "T3"]) for m in MODELS}
        thresholds_text = f"{float(min(thresholds.values())):g}° to {float(max(thresholds.values())):g}°"
        accuracy.set_title(scene.title(), loc="left", fontweight="bold", fontsize=9.1, pad=6)
        accuracy.set_title(thresholds_text, loc="right", color=MUTED, fontsize=7.5, pad=6)
        accuracy_scale(accuracy, .37)
        accuracy.axhspan(.75, 1.026, color=ABOVE, zorder=0)
        accuracy.set(xlim=(-.022, 1.030))
        accuracy.set_xticks([])
        accuracy.spines["bottom"].set_visible(False)
        accuracy.tick_params(axis="y", labelleft=col == 0, left=col == 0)
        if col == 0:
            accuracy.set_ylabel("Accuracy", labelpad=1.5)
        else:
            accuracy.spines["left"].set_visible(False)

        range_ax.spines[:].set_visible(False)
        range_ax.set(xlim=accuracy.get_xlim(), ylim=(-.48, 3.63))
        range_ax.set_xticks(tilt_position(ticks), [f"{t:g}" for t in ticks])
        range_ax.tick_params(axis="both", length=0, pad=3.3)
        if col == 0:
            range_ax.set_yticks(range(4), [name(m) for m in reversed(MODELS)])
        else:
            range_ax.set_yticks([])

        for y, model in enumerate(reversed(MODELS)):
            rows = data[model, scene, "T3"]
            signal, fraction = measurements(rows)
            curve(accuracy, model, tilt_position(signal), fraction)
            first = thresholds[model]
            start = tilt_position(float(first))
            n, correct = rows[first]
            onset(accuracy, model, start, correct / n, size=4.8)
            color, marker = style(model)
            range_ax.hlines(y, 0, 1, color=TRACK, lw=1.35)
            range_ax.hlines(y, start, 1, color=color, lw=2.25)
            tested = signal[(signal >= float(first))]
            range_ax.plot(tilt_position(tested), np.full(len(tested), y), ".", color=color, ms=2.1)
            range_ax.plot(start, y, color=color, marker=marker, ms=3.7, zorder=4, mec="white", mew=.45)
            range_ax.annotate(f"{float(first):g}°", (start, y), xytext=(5, 0), textcoords="offset points",
                              va="center", fontsize=6.6, fontweight="bold", color=color,
                              bbox={"facecolor": "white", "edgecolor": "none", "pad": .45}, zorder=5)
        range_ax.text(0, 1.10, "Tested levels sustaining ≥75%", transform=range_ax.transAxes,
                      fontsize=7.0, fontweight="bold", va="bottom")
    fig.supxlabel("Orientation gap (degrees)", x=.563, y=.025, fontsize=7.4)
    return fig


SIGNALS = {
    "T1": dict(ticks=(.05, .2, .5, 1), xlim=(.043, 1.14), scale="log", label="Modulation depth", lower=.315),
    "T2": dict(ticks=(.05, .10, .20, .40), xlim=(.045, .445), scale="log", label="Luminance increment ΔY", lower=.45),
    "T4": dict(ticks=(12.5, 50, 75, 100), xlim=(8, 104), scale="linear", label="Motion coherence (%)", lower=.43),
}


def plot_signal(data, task):
    spec = SIGNALS[task]
    fig, panels = plt.subplots(1, 2, figsize=(COLUMN, 1.52), sharey=True)
    fig.subplots_adjust(left=.150, right=.973, bottom=.235, top=.740, wspace=.175)
    multiplier = 100 if task == "T4" else 1
    for ax, scene in zip(panels, SCENES, strict=True):
        ax.set_title(scene.title(), loc="left", fontweight="bold", fontsize=8.0, pad=4.2)
        ax.set_xscale(spec["scale"])
        ax.set_xlim(*spec["xlim"])
        ax.set_xticks(spec["ticks"], [f"{t:g}" for t in spec["ticks"]])
        ax.minorticks_off()
        accuracy_scale(ax, spec["lower"])
        levels = {model: stable75(data[model, scene, task]) for model in MODELS}
        reached = [float(level) for level in levels.values() if level is not None]
        for model, first in levels.items():
            rows = data[model, scene, task]
            signal, value = measurements(rows)
            curve(ax, model, multiplier * signal, value, small=True)
            if first is not None:
                n, correct = rows[first]
                onset(ax, model, multiplier * float(first), correct / n)
        if len(reached) == len(MODELS) or (task == "T4" and reached):
            first, last = multiplier * max(reached), multiplier * max(signal)
            ax.fill_betweenx((.75, 1.026), first, last, color="#dcece6", linewidth=0, zorder=1)
        if task == "T4" and levels["qwen"] is not None:
            first_coherence = 100 * float(levels["qwen"])
            ax.annotate(f"{first_coherence:g}%", (first_coherence, .77),
                        xytext=(-2, 5), textcoords="offset points", ha="right", color=style("qwen")[0],
                        fontsize=6.7, fontweight="bold")
    panels[0].set_ylabel("Accuracy", labelpad=1.0)
    fig.supxlabel(spec["label"], x=.562, y=.028, fontsize=7.25)
    handles = [Line2D([], [], color=style(m)[0], marker=style(m)[1], lw=.95, ms=2.5, label=name(m))
               for m in MODELS]
    fig.legend(handles=handles, ncol=4, loc="upper center", bbox_to_anchor=(.52, 1.01),
               handlelength=1.25, columnspacing=.96, handletextpad=.41, fontsize=6.95)
    return fig


DOMAINS = SCENES
TASKS = ("T1", "T2", "T3", "T4", "T5")
SERIES = ("human", "astra", "qwen")
HUMAN_STYLES = (("Human mean", "#303840", "o", "-"),
          ("Astra · PNG Low", "#b25427", "s", "--"),
          ("Qwen · video Off", "#197762", "^", "-"))


def plot_human(cells, output):
    _, sans = cvpr_font_families()
    plt.rcParams.update({"font.family": sans, "font.size": 7.0,
                         "axes.labelsize": 7.0, "xtick.labelsize": 6.6,
                         "ytick.labelsize": 6.6, "legend.fontsize": 7.4})
    fig, axes = plt.subplots(2, 5, figsize=(WIDTH, 3.00),
                             gridspec_kw={"width_ratios": [1.30, 1, 1.25, 1, .75]})
    fig.subplots_adjust(left=.095, right=.992, bottom=.140, top=.840, wspace=.37, hspace=.57)
    titles = ("T1 · Stripe direction", "T2 · Brightening", "T3 · Orientation", "T4 · Group motion", "T5 · Matching")
    labels = ("Spatial frequency", r"Increment $\Delta Y$", "Angle gap (°)", "Coherence (%)", "Candidates")
    for ri, domain in enumerate(DOMAINS):
        for ci, task in enumerate(TASKS):
            ax = axes[ri, ci]
            group = [r for r in cells if (r["task"], r["domain"]) == (task, domain)]
            if ri == 0:
                ax.set_title(titles[ci], fontsize=7.3, weight="bold", pad=8)
            if task == "T1":
                data = np.array([[r[s]["accuracy"] for r in group] for s in SERIES]) * 100
                heatmap = ax.imshow(data, aspect="auto", cmap=ACCURACY_COLORS, vmin=0, vmax=100)
                ax.set_yticks(range(3), ["Human", "Astra", "Qwen"])
                ax.set_xticks(range(8), [f"{r['level']:g}" for r in group])
                ax.tick_params(length=0, pad=3)
                maxima = data.max(axis=0)
                for y in range(3):
                    # Label a contiguous run of identical column maxima once.
                    # The cells and color scale still display every observation.
                    winners = np.flatnonzero(np.isclose(data[y], maxima, rtol=0, atol=1e-10))
                    runs = []
                    for x in winners:
                        if (runs and x == runs[-1][-1] + 1 and
                                np.isclose(data[y, x], data[y, runs[-1][-1]], rtol=0, atol=1e-10)):
                            runs[-1].append(x)
                        else:
                            runs.append([x])
                    for run in runs:
                        value = data[y, run[0]]
                        ax.text(np.mean(run), y, f"{value:.0f}", ha="center", va="center",
                                fontsize=6.3, fontweight="bold",
                                color="white" if value >= 78 or value < 20 else "#152a3b",
                                bbox=dict(facecolor=heatmap.cmap(heatmap.norm(value)),
                                          edgecolor="none", pad=.2), zorder=4)
                ax.set_xticks(np.arange(-.5, 8, 1), minor=True)
                ax.set_yticks(np.arange(-.5, 3, 1), minor=True)
                ax.grid(which="minor", color="white", linewidth=.45)
                ax.tick_params(which="minor", length=0)
                for spine in ax.spines.values():
                    spine.set_visible(False)
            else:
                x = np.array([1, 2] if task == "T5" else [r["level"] for r in group])
                if task == "T4":
                    x = x * 100
                for name, (_, color, marker, ls) in zip(SERIES, HUMAN_STYLES):
                    y = np.array([r[name]["accuracy"] for r in group]) * 100
                    bounds = np.array([r[name]["interval95"] for r in group]) * 100
                    # Keep every estimate and interval. Dense point symbols obscure
                    # the curves at publication size, especially 29 T3 signal levels.
                    ax.fill_between(x, bounds[:, 0], bounds[:, 1], color=color,
                                    alpha=.065, linewidth=0, zorder=1)
                    ax.plot(x, y, color=color, marker=marker if task == "T5" else None,
                            ls=ls, lw=1.25 if name == "human" else 1.0,
                            markersize=2.7, markerfacecolor="white", markeredgewidth=.65,
                            zorder=3 if name == "human" else 4)
                ax.axhline(50, color="#aeb7bc", ls=":", lw=.55, zorder=0)
                ax.axhline(75, color="#c3cbd0", ls="--", lw=.5, zorder=0)
                ax.set_ylim(0, 105)
                ax.set_yticks([0, 50, 100])
                ax.tick_params(length=2, pad=2)
                if ci != 1:
                    ax.set_yticklabels([])
                if task == "T2":
                    ax.set_xticks([.05, .20, .40], [".05", ".20", ".40"])
                elif task == "T3":
                    ax.set_xscale("log", base=2)
                    ax.set_xticks([.5, 2, 8, 32, 90], [".5", "2", "8", "32", "90"])
                elif task == "T4":
                    ax.set_xticks([12.5, 50, 100], ["12.5", "50", "100"])
                else:
                    ax.set_xlim(.75, 2.25)
                    ax.set_xticks([1, 2], ["1", "2"])
            if ri == 1:
                ax.set_xlabel(labels[ci], labelpad=3)
        pos = axes[ri, 0].get_position()
        fig.text(.012, (pos.y0 + pos.y1)/2, domain.title(), va="center", ha="center",
                 rotation=90, fontsize=8, weight="bold")
    legend = [Line2D([], [], color=color, ls=ls, lw=1.25 if label == "Human mean" else 1.0,
                     label=label) for label, color, marker, ls in HUMAN_STYLES]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(.55, 1.0), ncol=3,
               handlelength=2.4, columnspacing=1.9)
    fig.text(.020, .957, "Accuracy (%)", fontsize=7.0, va="center")
    pos = axes[1, 0].get_position()
    upper = axes[0, 0].get_position()
    cax = fig.add_axes([pos.x0, (pos.y1 + upper.y0)/2 - .007, pos.width, .014])
    cb = fig.colorbar(heatmap, cax=cax, orientation="horizontal", ticks=[0, 50, 100])
    cb.ax.tick_params(length=1.5, pad=1, labelsize=6)
    cb.ax.set_xticklabels(["0", "50", "100%"])
    cb.outline.set_visible(False)
    fig.savefig(output, metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


ARMS = ("video_off", "video_low", "png_off", "png_low")
QWEN_STYLES = (("Native video · Off", "#2459a4", "--", "o"),
          ("Native video · Low", "#2459a4", "-", "o"),
          ("Ordered PNGs · Off", "#b25427", "--", "s"),
          ("Ordered PNGs · Low", "#b25427", "-", "s"))


def summarize(records, task, levels, domain=None):
    output = {arm: [] for arm in ARMS}
    for level in levels:
        rows = [r for r in records if r["task"] == task and (domain is None or r["domain"] == domain)
                and (r["condition"][1] if task in {"T1", "T5"} else r["condition"][-1]) == level]
        assert len(rows) == {"T1": 22, "T4": 32, "T5": 48}[task]
        for arm in ARMS:
            output[arm].append(sum(r[arm] for r in rows))
    return output


def plot_reasoning(records, output):
    sf = [.5, 1, 2, 4, 8, 16, 32, 64]
    coherence = [i / 8 for i in range(1, 9)]
    stripe = summarize(records, "T1", sf, "classic")
    motion = summarize(records, "T4", coherence)
    matching = summarize(records, "T5", ["single", "dual"])
    assert sum(stripe["png_off"][:3]) == 33 and sum(stripe["png_low"][:3]) == 57
    assert min(motion["video_low"]) >= 24 and min(motion["png_low"]) >= 24
    _, sans = cvpr_font_families()
    plt.rcParams["font.family"] = sans
    fig, axs = plt.subplots(1, 3, figsize=(WIDTH, 2.12))
    fig.subplots_adjust(left=.060, right=.990, bottom=.235, top=.760, wspace=.30)
    for ax in axs:
        ax.set_ylim(0, 105)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.axhline(50, color="#8c949b", lw=.6, ls=":", zorder=0)
        ax.axhline(75, color="#a4aeb6", lw=.6, ls="--", zorder=0)
        ax.tick_params(length=2.5)
    axs[0].set_ylabel("Accuracy (%)")
    for arm, (label, color, ls, marker) in zip(ARMS, QWEN_STYLES):
        for ax, x, cells, denom in ((axs[0], sf, stripe, 22), (axs[1], np.array(coherence)*100, motion, 32)):
            ax.plot(x, np.array(cells[arm])/denom*100, label=label, color=color, ls=ls,
                    marker=marker, markersize=3.4, lw=1.15,
                    markerfacecolor=color if arm.endswith("low") else "white", markeredgewidth=.8)
    axs[0].set_xscale("log", base=2)
    axs[0].set_xticks(sf, [f"{x:g}" for x in sf])
    axs[0].set_xlabel("Spatial frequency (cycles/aperture)")
    axs[0].set_title("(a) T1 · Classic stripes", loc="left", weight="bold", pad=6)
    axs[1].set_xticks([12.5, 37.5, 62.5, 87.5], ["12.5", "37.5", "62.5", "87.5"])
    axs[1].set_xlabel("Motion coherence (%)")
    axs[1].set_title("(b) T4 · Group motion", loc="left", weight="bold", pad=6)
    for i, (arm, (_, color, _, _)) in enumerate(zip(ARMS, QWEN_STYLES)):
        xpos = np.arange(2) + (i-1.5)*.18
        axs[2].bar(xpos, np.array(matching[arm])/48*100, .16, color=color if arm.endswith("low") else "white",
                   edgecolor=color, lw=.9, hatch=None if arm.endswith("low") else "//", zorder=2)
    axs[2].set_xticks([0, 1], ["1 candidate", "2 candidates"])
    axs[2].set_xlabel("Reference matching")
    axs[2].set_title("(c) T5 · Matching", loc="left", weight="bold", pad=6)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.53, 1.01), ncol=4,
               handlelength=2.2, columnspacing=1.6, fontsize=7.1)
    fig.savefig(output)
    plt.close(fig)
    print(json.dumps({"T1_classic_n_per_point":22,"T1_classic":stripe,
                      "T4_pooled_n_per_point":32,"T4":motion,
                      "T5_pooled_n_per_bar":48,"T5":matching}, indent=2))


def plot_axis(study, base, output):
    """Plot the complete native-video comparison and its actual 4-degree frames."""
    import cv2
    import numpy as np
    import pandas as pd
    from matplotlib.lines import Line2D

    spec = read_json(study / 't3-axis-inputs.json')
    data = pd.read_csv(study / 't3-axis-results.csv').rename(columns={'degrees':'level','sample_id':'id'})
    require(len(data)==240 and not data.duplicated(['model','id']).any(), 'Incomplete axis panel')
    for arm in ('base','marked'):
        data[f'{arm}_correct'] = data[arm]==data.truth
    row = spec['inputs'][spec['example_id']]

    with plt.rc_context({"font.family": cvpr_font_families(), "font.size": 7,
                         "pdf.fonttype": 42, "axes.labelcolor": "#243A4A",
                         "text.color": "#243A4A", "savefig.facecolor": "white"}):
        fig = plt.figure(figsize=(3.35, 2.63))
        for left, arm, suffix, label in ((.04, "base", "original", "Original (crop)"),
                                         (.54, "marked", "marked", "Marked axes (same crop)")):
            path = (base if arm == "base" else study) / row[f"{arm}_path"]
            require(sha256(path) == row[f"{arm}_sha256"], f"Changed axis example {path}")
            cap = cv2.VideoCapture(str(path))
            cap.set(cv2.CAP_PROP_POS_FRAMES, row["onset_frame"] + 15)
            ok, frame = cap.read()
            cap.release()
            assert ok and frame.shape == (512, 768, 3)
            ax = fig.add_axes((left, .64, .42, .30))
            # Identical display crop preserves both objects and the original pixels.
            display = frame[190:440, 150:620]
            ax.imshow(cv2.cvtColor(display, cv2.COLOR_BGR2RGB), interpolation="none")
            ax.set_axis_off()
            ax.set_title(label, fontsize=6.6, pad=3)
        for col, (model, label, color) in enumerate((("qwen", "Qwen3.8", "#00568A"),
                                                    ("cosmos", "Cosmos3", "#BD4D20"))):
            ax = fig.add_axes((.13 + col * .47, .18, .36, .32))
            part = data[data.model == model]
            assert len(part) == 120
            objects = sorted(part.object.unique())
            levels = sorted(part.level.unique())
            rng = np.random.default_rng(spec["bootstrap"]["seed"])
            weights = rng.multinomial(len(objects), np.ones(len(objects))/len(objects),
                                      size=spec["bootstrap"]["draws"])
            for arm, line, draw_color in (("base", "--", "#7B858C"), ("marked", "-", color)):
                means, low, high = [], [], []
                for level in levels:
                    cells = part[part.level == level]
                    counts = cells.groupby("object").size().reindex(objects, fill_value=0).values
                    hits = cells.groupby("object")[f"{arm}_correct"].sum().reindex(objects, fill_value=0).values
                    denominator = weights @ counts
                    boot = (weights @ hits)[denominator > 0] / denominator[denominator > 0]
                    means.append(cells[f"{arm}_correct"].mean())
                    lo, hi = np.percentile(boot, [2.5, 97.5])
                    low.append(lo); high.append(hi)
                    assert len(cells) == 12
                ax.fill_between(levels, low, high, color=draw_color, alpha=.08, lw=0)
                ax.plot(levels, means, color=draw_color, ls=line, lw=1.05)
                onset = stable75({level: (12, round(12*mean)) for level, mean in zip(levels, means)})
                ax.plot(onset, means[levels.index(onset)], "o", color=draw_color, ms=3.7,
                        mfc=draw_color if arm == "marked" else "white", mew=.8)
            ax.set_xscale("log", base=2)
            ax.set_xticks([.5, 2, 8, 32], ["0.5", "2", "8", "32"])
            ax.set(xlim=(.42, 38), ylim=(0, 1.04), title=label)
            ax.set_yticks([0, .5, .75, 1], ["0", "50", "75", "100"])
            if col == 0:
                ax.set_ylabel("Accuracy (%)", labelpad=2)
            else:
                ax.tick_params(labelleft=False)
            ax.axhline(.75, color="#9AA9B0", ls=":", lw=.7, zorder=0)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(length=2, pad=2, labelsize=6.3)
        fig.text(.55, .065, "Orientation gap (degrees)", ha="center", fontsize=7)
        fig.legend(handles=[Line2D([], [], color="#7B858C", ls="--", label="Original"),
                            Line2D([], [], color="#243A4A", label="Marked axes")],
                   loc="lower center", bbox_to_anchor=(.55, .55), ncol=2,
                   frameon=False, fontsize=6.6, handlelength=1.8, borderpad=0)
        fig.savefig(output, metadata={"Title": "T3 axis cues: native 180-frame video, Thinking Off"})
        plt.close(fig)




def render(args, curves, panel, predictions):
    illustrations = args.study / "figures"
    subprocess.run([sys.executable, "-m", "paper_core.figure1",
                    "--frames", str(illustrations / "figure-1-frames.npz"),
                    "--inputs", str(illustrations / "figure-1-inputs.json"),
                    "--output", str(args.out / "figure-1")], check=True)
    _, sans = cvpr_font_families()
    with plt.rc_context({"font.family": sans}):
        for number, fig in ((2, plot_orientation(curves)),
                            (3, plot_signal(curves, "T2")),
                            (5, plot_signal(curves, "T4"))):
            fig.savefig(args.out/f"figure-{number}.pdf", metadata={"CreationDate":None})
            plt.close(fig)
    plot_axis(args.study, args.base, args.out / "figure-4.pdf")
    plot_human(read_json(args.study / "human-figure6.json")["cells"], args.out / "figure-6.pdf")
    arms = dict(zip(ARMS, ("off", "low", "png-off", "png-low"), strict=True))
    records = []
    for sid, row in panel.items():
        cond = [row["task"], *factors(row)]
        record = {"task": row["task"], "domain": row["domain"], "condition": cond}
        for key, arm in arms.items():
            record[key] = predictions["qwen", arm, "video", sid] == row["truth"]
        records.append(record)
    plot_reasoning(records, args.out / "figure-7.pdf")

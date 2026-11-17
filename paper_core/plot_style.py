"""Shared style for the visual psychophysics figures."""
from functools import lru_cache
from pathlib import Path
import shutil
import subprocess

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager, pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np

HERE = Path(__file__).resolve().parent
WIDTH = 6.875
INK, MUTED, RULE = "#152a3b", "#475969", "#d2dbe2"
LABELS = {"cosmos": "Cosmos3", "nemotron": "Nemotron3.5", "qwen": "Qwen3.8",
          "gemma": "Gemma4", "glimmer": "Muse Glimmer"}
# figures4papers uses a deep blue anchor and restrained green and red accents.
COLORS = dict(zip(LABELS, ("#0f4d92", "#b64342", "#197762", "#8052a0", "#866416"), strict=True))
MARKERS = dict(zip(LABELS, ("o", "s", "^", "D", "P"), strict=True))
OPEN_STYLES = ("static", "natural", "open")
ACCURACY_COLORS = LinearSegmentedColormap.from_list("accuracy", [
    (0, "#843b2a"), (.40, "#d89970"), (.50, "#f5f2eb"),
    (.75, "#7caac1"), (1, "#174469"),
])

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 7.5,
    "axes.labelsize": 7.4,
    "axes.titlesize": 8.0,
    "axes.labelcolor": INK,
    "axes.edgecolor": MUTED,
    "axes.linewidth": .65,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 6.8,
    "ytick.labelsize": 6.8,
    "xtick.major.width": .65,
    "ytick.major.width": .65,
    "legend.fontsize": 7.0,
    "legend.frameon": False,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "pdf.compression": 9,
    "ps.fonttype": 42,
})


@lru_cache(maxsize=None)
def tex_font_family(stem, styles=("regular", "bold")):
    """Register a font family from the installed TeX Live OpenType fonts."""
    kpse = shutil.which("kpsewhich")
    fallback = "DejaVu Serif" if stem == "texgyretermes" else "DejaVu Sans"
    if kpse is None:
        return fallback
    for style in styles:
        font = subprocess.run([kpse, f"{stem}-{style}.otf"], check=False,
                              capture_output=True, text=True).stdout.strip()
        if not font:
            return fallback
        font_manager.fontManager.addfont(font)
    return font_manager.FontProperties(fname=font).get_name()


@lru_cache(maxsize=1)
def cvpr_font_families():
    """Match CVPR's Times/Helvetica typography using the TeX Live vector fonts."""
    return tuple(tex_font_family(stem) for stem in ("texgyretermes", "texgyreheros"))


def axes_style(ax, grid="y"):
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.6, pad=2.6)
    if grid:
        ax.grid(axis=grid, color=RULE, alpha=.78, linewidth=.45)


def percent_axis(ax, limits=(.32, 1.025), ticks=(.5, .75, 1), chance=True):
    axes_style(ax)
    ax.set_ylim(*limits)
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    if chance:
        ax.axhline(.5, color=MUTED, alpha=.8, lw=.65, linestyle=(0, (2, 2)), zorder=1)

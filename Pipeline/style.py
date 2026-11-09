"""
style.py — Shared chart style for every figure of the project
=============================================================
One place for colors, fonts and the title / subtitle / source layout,
so all charts look like they belong to the same report.

Colors are a nod to the Toronto Tempo (Borealis Blue, Tempo Bordeaux).
The exact shades are approximations, not official brand values.
Font: Poppins if installed (free on Google Fonts), else Helvetica / Arial.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Palette -----------------------------------------------------------------
NAVY = "#1b2a4a"        # text, main lines
INK_2 = "#5b6478"       # secondary text
BOREALIS = "#3f8fd2"    # "good" / played / Toronto main series
BORDEAUX = "#7d2240"    # injuries, Toronto highlight, warnings
SOFT = "#c9cfdb"        # other teams, neutral bars
LIGHT = "#eef1f6"       # guides, background bands
BG = "#ffffff"

# schedule load levels (green / amber / bordeaux)
LOAD_COLORS = {"Standard": "#5aa469", "Adjusted": "#e0a526", "Priority": BORDEAUX}

FOOTER = "Independent analysis - not affiliated with the Toronto Tempo or the WNBA."

plt.rcParams.update({
    "font.family": ["Poppins", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "axes.spines.bottom": False,
    "axes.grid": False,
    "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "xtick.major.size": 0, "ytick.major.size": 0,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": NAVY,
    "axes.titlelocation": "left", "axes.titlepad": 10,
    "font.size": 10, "legend.frameon": False,
})
# silence "font not found" warnings when Poppins is not installed
import logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


def header(fig, title, subtitle=None):
    """Big title (the takeaway) + subtitle (what we are looking at)."""
    fig.text(0.04, 0.975, title, fontsize=19, fontweight="bold", color=NAVY, va="top")
    if subtitle:
        fig.text(0.04, 0.925, subtitle, fontsize=11, color=INK_2, va="top")


def source(fig, text):
    fig.text(0.04, 0.018, f"{text}   {FOOTER}", fontsize=7.5, color=INK_2)


def hgrid(ax, values):
    """Light horizontal guide lines at chosen values."""
    for v in values:
        ax.axhline(v, color=LIGHT, lw=1, zorder=0)


def save(fig, path):
    fig.savefig(path, dpi=200)
    plt.close(fig)

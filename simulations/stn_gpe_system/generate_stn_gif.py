#!/usr/bin/env python3
"""
generate_stn_gif.py
===================
Produces a side-by-side animated GIF comparing STN network activity under
three conditions:

    Normal  |  Parkinson's (PD)  |  PD + DBS

Each column contains two panels:
  Top    – 2-D heatmap of the STN neuron grid, animated with a rolling
           spike-rate window so you can watch activity propagate spatially.
  Bottom – LFP power spectrum (Welch) with the beta band (13–30 Hz)
           highlighted — the pathological beta excess is clearly visible
           in PD and its suppression by DBS.

Usage
-----
Place this script anywhere inside the BG-project tree (it auto-detects the
project root two levels up, same convention as the notebooks).

    python generate_stn_gif.py

The file  stn_comparison.gif  is written to the same directory.

Configuration
-------------
Edit the CONFIGURATION block below to change yaml files, animation speed,
rolling-window width, output DPI, etc.
"""

import os
import sys
import warnings
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap
from scipy import signal

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
#  Project root  (two levels up from this file, same as the notebooks)
# ─────────────────────────────────────────────────────────────────────────────
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = os.getcwd()

PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from stn_gpe import STN_GPe_loop, load_yaml

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION  –  edit these to taste
# ─────────────────────────────────────────────────────────────────────────────
STN_PARAM_DIR = os.path.join(PROJECT_ROOT, "params", "stn_gpe_params")
OUTPUT_GIF    = os.path.join(_HERE, "stn_comparison.gif")

# Yaml file for each condition (relative to STN_PARAM_DIR)
CONDITIONS = {
    "Normal":   "params_Normal.yaml",
    "PD":       "params_PD.yaml",
    "PD + DBS": "params_std_DBS.yaml",
}

# Animation
WINDOW_MS    = 50    # ms  – rolling spike-rate window shown on the grid
STEP_MS      = 20    # ms  – time between successive frames
ANALYSIS_SEC = 1.0   # s   – tail of the simulation to animate
FPS          = 15    # frames per second in the output GIF
OUTPUT_DPI   = 140   # increase for higher-resolution GIF

# Display
BETA_LO, BETA_HI = 10, 35   # Hz  – beta band boundaries
PSD_FMAX         = 40 #60        # Hz  – x-axis limit for power spectrum

# ─────────────────────────────────────────────────────────────────────────────
#  Colour palette  (dark GitHub-style theme)
# ─────────────────────────────────────────────────────────────────────────────
BG          = "#0d1117"
SURFACE     = "#161b22"
BORDER      = "#30363d"
TEXT        = "#c9d1d9"
SUBTEXT     = "#8b949e"

COND_ACCENT = {
    "Normal":   "#4dabf7",   # electric blue
    "PD":       "#ff922b",   # amber-orange
    "PD + DBS": "#cc5de8",   # violet-purple
}

def _make_cmap(hex_col: str) -> LinearSegmentedColormap:
    """Black → dim → full-saturation single-hue colormap."""
    r = int(hex_col[1:3], 16) / 255
    g = int(hex_col[3:5], 16) / 255
    b = int(hex_col[5:7], 16) / 255
    stops = [
        (0.00, (0.05, 0.07, 0.09)),    # near-black background
        (0.30, (r*0.18, g*0.18, b*0.18)),
        (0.60, (r*0.55, g*0.55, b*0.55)),
        (0.85, (r*0.88, g*0.88, b*0.88)),
        (1.00, (r, g, b)),
    ]
    return LinearSegmentedColormap.from_list(hex_col, [(v, c) for v, c in stops])

CMAPS = {k: _make_cmap(v) for k, v in COND_ACCENT.items()}


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _style_ax(ax, xlabel=None, ylabel=None):
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=TEXT, labelsize=7.5, length=2.5, width=0.6)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)
        sp.set_linewidth(0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if xlabel:
        ax.set_xlabel(xlabel, color=TEXT, fontsize=8, labelpad=3)
    if ylabel:
        ax.set_ylabel(ylabel, color=TEXT, fontsize=8, labelpad=3)


def run_condition(yaml_fname: str) -> dict:
    """Run one STN-GPe simulation and return raw arrays."""
    path    = os.path.join(STN_PARAM_DIR, yaml_fname)
    params  = load_yaml(path)
    results = STN_GPe_loop(path)
    # return {
    #     "spike": np.array(results["spike_stn"]),   # (T, N)
    #     "lfp":   np.array(results["lfp_stn"]),      # (T,)
    #     "dt":    float(params.get("dt", 0.1)),       # ms
    # }
    return {
        "spike": np.array(results["spike_stn"]),
        "lfp":   np.array(results["lfp_stn"]),
        "v_stn": np.array(results["v_stn"]),    # ← added
        "dt":    float(params.get("dt", 0.1)),
    }

def preprocess(raw: dict) -> dict:
    """
    From raw simulation output produce:
      grid_frames  – (n_frames, G, G)  rolling spike-rate heatmaps
      freqs, psd   – Welch PSD of the smoothed LFP
      t_axis       – real time (s) at the centre of each frame
      vmax         – consistent colour-scale ceiling for this condition
    """
    spike = raw["spike"]
    lfp   = raw["lfp"]
    dt    = raw["dt"]
    sr    = int(round(1000.0 / dt))    # samples per second

    T, N  = spike.shape
    G     = int(round(np.sqrt(N)))     # grid side  (e.g. 16 → 4×4, 256 → 16×16)

    # Analysis window: last ANALYSIS_SEC seconds
    win_steps = int(ANALYSIS_SEC * sr)
    t_low     = max(0, T - win_steps)
    spike_win = spike[t_low:]          # (win_steps, N)
    lfp_win   = lfp[t_low:]            # (win_steps,)

    # Rolling-window frame extraction
    w  = max(1, int(WINDOW_MS * sr / 1000))   # window in steps
    s  = max(1, int(STEP_MS   * sr / 1000))   # step   in steps
    n_frames   = max(1, (len(spike_win) - w) // s)
    grid_frames = np.zeros((n_frames, G, G), dtype=np.float32)
    for f in range(n_frames):
        chunk             = spike_win[f*s : f*s + w].mean(axis=0)   # (N,)
        grid_frames[f]    = chunk.reshape(G, G)

    # Per-frame time centre (seconds into the analysis window)
    t_axis = np.array([(f * s + w / 2) * dt / 1000.0 for f in range(n_frames)])

    # Colour-scale: 99th percentile to avoid hot-pixel dominance
    vmax = float(np.percentile(grid_frames, 99)) or 1e-6

    # LFP → smoothed → Welch PSD
    lfp_sm = signal.savgol_filter(lfp_win, window_length=11, polyorder=5)
    nperseg = min(sr, len(lfp_sm))
    freqs, psd = signal.welch(lfp_sm, fs=sr, nperseg=nperseg, noverlap=nperseg // 2)
    mask  = freqs <= PSD_FMAX
    freqs = freqs[mask]
    psd   = 10.0 * np.log10(psd[mask] + 1e-12)

    return dict(
        grid_frames = grid_frames,
        t_axis      = t_axis,
        vmax        = vmax,
        freqs       = freqs,
        psd         = psd,
        G           = G,
        n_frames    = n_frames,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # ── 1. Run simulations ───────────────────────────────────────────────────
    print("=" * 60)
    print("STN network GIF generator")
    print("=" * 60)
    raw_results = {}
    for cond, fname in CONDITIONS.items():
        print(f"\n[{cond}]  {fname}")
        raw_results[cond] = run_condition(fname)
        T, N = raw_results[cond]["spike"].shape
        print(f"  → done  T={T}  N={N}  dt={raw_results[cond]['dt']} ms")

    # ── 2. Pre-process ───────────────────────────────────────────────────────
    print("\nPre-processing …")
    data = {cond: preprocess(raw_results[cond]) for cond in CONDITIONS}
    n_frames = min(d["n_frames"] for d in data.values())
    print(f"  → {n_frames} frames  ({n_frames / FPS:.1f} s at {FPS} fps)")

    # ── 3. Build figure ──────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 8.5), facecolor=BG)
    fig.subplots_adjust(
        left=0.05, right=0.97,
        top=0.88,  bottom=0.09,
        wspace=0.32, hspace=0.0,
    )

    outer = gridspec.GridSpec(1, 3, figure=fig, wspace=0.32)

    ax_grid = {}
    ax_psd  = {}
    for col, cond in enumerate(CONDITIONS):
        inner = gridspec.GridSpecFromSubplotSpec(
            2, 1,
            subplot_spec=outer[col],
            height_ratios=[1.7, 1.0],
            hspace=0.42,
        )
        ax_grid[cond] = fig.add_subplot(inner[0])
        ax_psd[cond]  = fig.add_subplot(inner[1])

    # ── 4. Static elements ───────────────────────────────────────────────────

    # ── Figure title ─────────────────────────────────────────────────────────
    fig.text(
        0.51, 0.965,
        "STN Network Activity",
        ha="center", va="top",
        color=TEXT, fontsize=15, fontweight="bold",
    )
    fig.text(
        0.51, 0.940,
        "2-D neuron grid (rolling 50 ms spike rate)  ·  LFP power spectrum",
        ha="center", va="top",
        color=SUBTEXT, fontsize=8.5,
    )

    # ── Column headers ────────────────────────────────────────────────────────
    col_x = [0.19, 0.505, 0.82]   # approximate horizontal centres of columns
    for cx, cond in zip(col_x, CONDITIONS):
        accent = COND_ACCENT[cond]
        fig.text(
            cx, 0.915,
            cond,
            ha="center", va="bottom",
            color=accent, fontsize=13, fontweight="bold",
        )
        # coloured underline
        fig.add_artist(
            plt.Line2D(
                [cx - 0.085, cx + 0.085], [0.913, 0.913],
                transform=fig.transFigure,
                color=accent, linewidth=1.5, alpha=0.6,
            )
        )

    # ── Power spectrum (static per condition) ─────────────────────────────────
    psd_ymins, psd_ymaxs = [], []
    for cond, d in data.items():
        psd_ymins.append(d["psd"].min())
        psd_ymaxs.append(d["psd"].max())
    psd_ymin = min(psd_ymins) - 2
    psd_ymax = max(psd_ymaxs) + 3

    for cond, d in data.items():
        ax  = ax_psd[cond]
        col = COND_ACCENT[cond]
        _style_ax(ax, xlabel="Frequency (Hz)", ylabel="Power (dB)")

        # Beta band shading
        ax.axvspan(BETA_LO, BETA_HI, color=col, alpha=0.10, zorder=0, linewidth=0)
        ax.axvspan(BETA_LO, BETA_HI, color=col, alpha=0.05, zorder=0, linewidth=0)

        # PSD line with gradient-like fill
        ax.plot(d["freqs"], d["psd"], color=col, linewidth=1.5, zorder=3)
        ax.fill_between(d["freqs"], psd_ymin, d["psd"],
                        color=col, alpha=0.15, zorder=2)

        # Beta-band edges
        for fv in (BETA_LO, BETA_HI):
            ax.axvline(fv, color=col, linewidth=0.5, linestyle="--", alpha=0.45, zorder=1)

        # Peak marker in beta
        bmask = (d["freqs"] >= BETA_LO) & (d["freqs"] <= BETA_HI)
        if bmask.any():
            pk_idx = np.argmax(d["psd"][bmask])
            pk_f   = d["freqs"][bmask][pk_idx]
            pk_p   = d["psd"][bmask][pk_idx]
            ax.plot(pk_f, pk_p, "o", color=col, markersize=4, zorder=5)
            ax.text(pk_f + 0.8, pk_p + 0.5, f"{pk_f:.0f} Hz",
                    color=col, fontsize=7.0, va="bottom", zorder=6)

        ax.set_xlim(0, PSD_FMAX)
        ax.set_ylim(psd_ymin, psd_ymax)
        ax.set_title("LFP Power Spectrum", color=TEXT, fontsize=8.0,
                     fontweight="bold", pad=4)

        # Beta label
        ax.text(
            (BETA_LO + BETA_HI) / 2, psd_ymax - 0.5,
            "β", ha="center", va="top",
            color=col, fontsize=8, alpha=0.7,
        )

    # ── Grid axes – initial frame ─────────────────────────────────────────────
    im_obj   = {}
    txt_time = {}
    txt_rate = {}

    for cond, d in data.items():
        ax  = ax_grid[cond]
        col = COND_ACCENT[cond]
        G   = d["G"]
        _style_ax(ax, xlabel="Neuron column", ylabel="Neuron row")

        im = ax.imshow(
            d["grid_frames"][0],
            cmap        = CMAPS[cond],
            vmin        = 0,
            vmax        = d["vmax"],
            interpolation = "nearest",
            aspect      = "equal",
            origin      = "lower",
        )

        # Minor grid lines to show cell boundaries
        ax.set_xticks(np.arange(-0.5, G, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, G, 1), minor=True)
        ax.grid(which="minor", color="#21262d", linewidth=0.4, zorder=5)
        ax.tick_params(which="minor", length=0)

        # Major ticks at a few positions
        tks = sorted({0, G // 4, G // 2, 3 * G // 4, G - 1})
        ax.set_xticks(tks);  ax.set_xticklabels(tks)
        ax.set_yticks(tks);  ax.set_yticklabels(tks)

        # Colourbar
        cb = fig.colorbar(im, ax=ax, pad=0.03, fraction=0.046, shrink=0.90)
        cb.ax.tick_params(colors=TEXT, labelsize=6.5)
        cb.set_label("Spike rate", color=SUBTEXT, fontsize=7)
        cb.outline.set_edgecolor(BORDER)

        ax.set_title(f"STN · {G}×{G} grid  (rolling {WINDOW_MS} ms)",
                     color=TEXT, fontsize=8.0, fontweight="bold", pad=5)

        # Overlay text
        tt = ax.text(
            0.02, 0.975, "t = 0.000 s",
            transform=ax.transAxes,
            color=TEXT, fontsize=7.5, va="top", ha="left",
            fontfamily="monospace", zorder=10,
        )
        tr = ax.text(
            0.98, 0.975, "rate = 0.000",
            transform=ax.transAxes,
            color=col, fontsize=7.5, va="top", ha="right",
            fontfamily="monospace", zorder=10,
        )

        # Colour-coded border
        for sp_name in ("top", "bottom", "left", "right"):
            ax.spines[sp_name].set_visible(True)
            ax.spines[sp_name].set_edgecolor(col)
            ax.spines[sp_name].set_linewidth(1.4)
            ax.spines[sp_name].set_alpha(0.5)

        im_obj[cond]   = im
        txt_time[cond] = tt
        txt_rate[cond] = tr

    # ── Progress bar ──────────────────────────────────────────────────────────
    pb_ax = fig.add_axes([0.05, 0.020, 0.90, 0.010])
    pb_ax.set_xlim(0, n_frames)
    pb_ax.set_ylim(0, 1)
    pb_ax.axis("off")
    pb_bg   = pb_ax.barh(0.5, n_frames, left=0, height=1.0, color="#21262d")
    pb_fill = plt.Rectangle((0, 0), 0, 1, color="#58a6ff",
                             transform=pb_ax.transData, zorder=3)
    pb_ax.add_patch(pb_fill)
    pb_label = pb_ax.text(
        n_frames / 2, -0.8, "",
        ha="center", va="top",
        color=SUBTEXT, fontsize=7, fontfamily="monospace",
        transform=pb_ax.transData,
    )

    # ── 5. Animation update function ─────────────────────────────────────────
    def update(frame: int):
        artists = []
        for cond, d in data.items():
            fi    = min(frame, d["n_frames"] - 1)
            gf    = d["grid_frames"][fi]
            t_val = d["t_axis"][fi]
            r_val = float(gf.mean())

            im_obj[cond].set_data(gf)
            txt_time[cond].set_text(f"t = {t_val:.3f} s")
            txt_rate[cond].set_text(f"μ = {r_val:.4f}")

            # Pulse the border brightness slightly with mean rate
            alpha = 0.35 + min(0.55, r_val * 8.0)
            for sp_name in ("top", "bottom", "left", "right"):
                ax_grid[cond].spines[sp_name].set_alpha(alpha)

            artists += [im_obj[cond], txt_time[cond], txt_rate[cond]]

        # Progress bar
        pb_fill.set_width(frame + 1)
        pct = 100.0 * (frame + 1) / n_frames
        pb_label.set_text(
            f"{pct:5.1f}%  ·  frame {frame + 1}/{n_frames}"
        )
        artists += [pb_fill, pb_label]
        return artists

    # ── 6. Render and save ───────────────────────────────────────────────────
    print(f"\nRendering {n_frames} frames …")
    anim = FuncAnimation(
        fig,
        update,
        frames   = n_frames,
        interval = int(1000 / FPS),
        blit     = True,
    )

    writer = PillowWriter(fps=FPS)
    anim.save(OUTPUT_GIF, writer=writer, dpi=OUTPUT_DPI)
    print(f"\n✓  Saved → {OUTPUT_GIF}")
    plt.close(fig)

    # ─────────────────────────────────────────────────────────────────────────────
    #  SECOND GIF — Single-neuron voltage traces (scrolling oscilloscope)
    #  Outputs: stn_voltage.gif  (same folder as the script)
    # ─────────────────────────────────────────────────────────────────────────────

    VOLT_OUTPUT_GIF = os.path.join(_HERE, "stn_voltage.gif")
    VOLT_WINDOW_MS  = 300   # ms  – width of the visible scrolling window
    VOLT_STEP_MS    = 15    # ms  – how far the window advances each frame
    VOLT_FPS        = 15

    # Three neurons shown per condition (row, col as fractions of grid size).
    # The centre neuron is the "primary" (solid); the others are ghost traces.
    # Having all three overlaid makes synchrony vs asynchrony visible at a glance.
    _NEURON_FRAC   = [(0.25, 0.25), (0.50, 0.50), (0.75, 0.75)]
    _NEURON_ALPHAS = [0.35, 1.00, 0.35]
    _NEURON_WIDTHS = [0.85, 1.80, 0.85]

    # ── extract & pre-process voltage data ───────────────────────────────────────
    print("\nBuilding voltage GIF …")
    volt_data = {}
    for cond in CONDITIONS:
        raw = raw_results[cond]
        v   = raw["v_stn"]           # (T, G, G)
        dt  = raw["dt"]
        sr  = int(round(1000.0 / dt))

        T_full, G, _ = v.shape

        # Tail window matching the main GIF
        win_steps = int(ANALYSIS_SEC * sr)
        t_low_v   = max(0, T_full - win_steps)
        v_win     = v[t_low_v:]      # (win_steps, G, G)

        # Resolve neuron indices
        neurons = [
            (max(0, min(G - 1, int(fy * G))),
            max(0, min(G - 1, int(fx * G))))
            for fy, fx in _NEURON_FRAC
        ]
        traces = [v_win[:, r, c].astype(np.float32) for r, c in neurons]

        # Scrolling-window frame counts
        w_steps  = max(1, int(VOLT_WINDOW_MS * sr / 1000))
        s_steps  = max(1, int(VOLT_STEP_MS   * sr / 1000))
        n_frames = max(1, (v_win.shape[0] - w_steps) // s_steps)

        # Full time axis for the mini-map
        t_full = np.arange(v_win.shape[0]) * dt / 1000.0   # seconds

        # Y-axis range (per condition, consistent across all frames)
        all_v = np.concatenate(traces)
        ymin  = float(np.percentile(all_v, 0.5))
        ymax  = float(np.percentile(all_v, 99.5))
        ypad  = (ymax - ymin) * 0.10

        volt_data[cond] = dict(
            traces   = traces,
            neurons  = neurons,
            t_full   = t_full,
            w_steps  = w_steps,
            s_steps  = s_steps,
            n_frames = n_frames,
            ymin     = ymin - ypad,
            ymax     = ymax + ypad,
        )

    n_frames_v = min(d["n_frames"] for d in volt_data.values())
    print(f"  → {n_frames_v} frames  ({n_frames_v / VOLT_FPS:.1f} s at {VOLT_FPS} fps)")

    # ── build figure ──────────────────────────────────────────────────────────────
    fig2 = plt.figure(figsize=(14, 7.0), facecolor=BG)
    fig2.subplots_adjust(
        left=0.06, right=0.97,
        top=0.87,  bottom=0.16,
        wspace=0.30,
    )

    outer2 = gridspec.GridSpec(1, 3, figure=fig2, wspace=0.30)
    ax_volt = {}
    ax_mini = {}
    for col, cond in enumerate(CONDITIONS):
        inner2 = gridspec.GridSpecFromSubplotSpec(
            2, 1,
            subplot_spec  = outer2[col],
            height_ratios = [3.2, 0.85],
            hspace        = 0.10,
        )
        ax_volt[cond] = fig2.add_subplot(inner2[0])
        ax_mini[cond] = fig2.add_subplot(inner2[1])

    # ── static text ───────────────────────────────────────────────────────────────
    fig2.text(0.51, 0.965, "STN Single-Neuron Voltage Traces",
            ha="center", va="top", color=TEXT, fontsize=14, fontweight="bold")
    fig2.text(
        0.51, 0.942,
        f"{VOLT_WINDOW_MS} ms scrolling window  ·  "
        "3 neurons overlaid  (centre = solid, neighbours = ghost)  ·  "
        "synchrony visible in PD",
        ha="center", va="top", color=SUBTEXT, fontsize=8.0,
    )

    for cx, cond in zip(col_x, CONDITIONS):
        accent = COND_ACCENT[cond]
        fig2.text(cx, 0.910, cond,
                ha="center", va="bottom", color=accent,
                fontsize=13, fontweight="bold")
        fig2.add_artist(plt.Line2D(
            [cx - 0.085, cx + 0.085], [0.908, 0.908],
            transform=fig2.transFigure,
            color=accent, linewidth=1.5, alpha=0.60,
        ))

    # ── per-condition axes setup ──────────────────────────────────────────────────
    line_objs = {}     # {cond: [line, line, line]}
    mini_boxes = {}    # {cond: Rectangle}
    time_txts  = {}
    volt_txts  = {}

    for cond, d in volt_data.items():
        acc = COND_ACCENT[cond]
        ax  = ax_volt[cond]
        axm = ax_mini[cond]

        # ── main voltage panel ────────────────────────────────────────────────────
        _style_ax(ax, ylabel="V (mV)")
        ax.set_xlim(0, VOLT_WINDOW_MS / 1000.0)
        ax.set_ylim(d["ymin"], d["ymax"])
        ax.set_xticklabels([])   # time is shown on mini-map below

        # Coloured border (matches condition accent)
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_edgecolor(acc)
            sp.set_linewidth(1.3)
            sp.set_alpha(0.45)

        r_pri, c_pri = d["neurons"][1]
        ax.set_title(
            f"primary neuron [{r_pri}, {c_pri}]",
            color=TEXT, fontsize=8.0, fontweight="bold", pad=5,
        )

        # Zero-voltage reference
        ax.axhline(0, color=BORDER, linewidth=0.5, linestyle="--", alpha=0.45)

        # Initialise trace lines (empty)
        lines = []
        for al, lw in zip(_NEURON_ALPHAS, _NEURON_WIDTHS):
            ln, = ax.plot([], [], color=acc, alpha=al, linewidth=lw, zorder=3)
            lines.append(ln)
        line_objs[cond] = lines

        # Overlay text
        tt = ax.text(0.98, 0.97, "t = 0.000 s",
                    transform=ax.transAxes, color=TEXT,
                    fontsize=7.5, va="top", ha="right", fontfamily="monospace")
        vt = ax.text(0.02, 0.97, "V = — mV",
                    transform=ax.transAxes, color=acc,
                    fontsize=7.5, va="top", ha="left", fontfamily="monospace")
        time_txts[cond] = tt
        volt_txts[cond] = vt

        # ── mini-map (full tail trace of the centre neuron) ───────────────────────
        _style_ax(axm, xlabel="Time (s)")
        axm.plot(d["t_full"], d["traces"][1],
                color=acc, linewidth=0.6, alpha=0.65)
        axm.set_xlim(d["t_full"][0], d["t_full"][-1])
        axm.set_ylim(d["ymin"], d["ymax"])
        axm.set_yticks([])
        axm.tick_params(labelsize=7.0)

        # Sliding highlight box
        box_w = VOLT_WINDOW_MS / 1000.0
        rect  = plt.Rectangle(
            (d["t_full"][0], d["ymin"]),
            box_w,
            d["ymax"] - d["ymin"],
            color=acc, alpha=0.18, zorder=4, linewidth=0,
        )
        axm.add_patch(rect)
        # Sliding box left edge indicator
        axm.add_patch(plt.Rectangle(
            (d["t_full"][0], d["ymin"]), 0.001,
            d["ymax"] - d["ymin"],
            color=acc, alpha=0.80, zorder=5, linewidth=0,
        ))
        mini_boxes[cond] = rect

    # ── progress bar ──────────────────────────────────────────────────────────────
    pb_ax2 = fig2.add_axes([0.05, 0.025, 0.90, 0.010])
    pb_ax2.set_xlim(0, n_frames_v)
    pb_ax2.set_ylim(0, 1)
    pb_ax2.axis("off")
    pb_ax2.barh(0.5, n_frames_v, left=0, height=1.0, color="#21262d")
    pb2_fill  = plt.Rectangle((0, 0), 0, 1, color="#388bfd",
                                transform=pb_ax2.transData, zorder=3)
    pb_ax2.add_patch(pb2_fill)
    pb2_label = pb_ax2.text(
        n_frames_v / 2, -0.9, "",
        ha="center", va="top", color=SUBTEXT,
        fontsize=7, fontfamily="monospace",
        transform=pb_ax2.transData,
    )

    # ── animation update ──────────────────────────────────────────────────────────
    def update_volt(frame: int):
        artists = []
        for cond, d in volt_data.items():
            fi    = min(frame, d["n_frames"] - 1)
            start = fi * d["s_steps"]
            end   = start + d["w_steps"]

            t_seg = d["t_full"][start:end]
            t_rel = t_seg - t_seg[0]    # 0 … VOLT_WINDOW_MS/1000 s

            for tr, ln in zip(d["traces"], line_objs[cond]):
                ln.set_data(t_rel, tr[start:end])

            mid   = start + d["w_steps"] // 2
            t_now = float(d["t_full"][min(mid, len(d["t_full"]) - 1)])
            v_now = float(d["traces"][1][min(mid, len(d["traces"][1]) - 1)])
            time_txts[cond].set_text(f"t = {t_now:.3f} s")
            volt_txts[cond].set_text(f"V = {v_now:+.1f} mV")

            # Slide the mini-map box
            mini_boxes[cond].set_x(d["t_full"][start])

            artists += line_objs[cond]
            artists += [time_txts[cond], volt_txts[cond], mini_boxes[cond]]

        pb2_fill.set_width(frame + 1)
        pb2_label.set_text(
            f"{100.0 * (frame + 1) / n_frames_v:5.1f}%  ·  "
            f"frame {frame + 1}/{n_frames_v}"
        )
        artists += [pb2_fill, pb2_label]
        return artists

    # ── render ────────────────────────────────────────────────────────────────────
    anim2 = FuncAnimation(
        fig2, update_volt,
        frames   = n_frames_v,
        interval = int(1000 / VOLT_FPS),
        blit     = True,
    )
    anim2.save(VOLT_OUTPUT_GIF, writer=PillowWriter(fps=VOLT_FPS), dpi=OUTPUT_DPI)
    print(f"✓  Saved → {VOLT_OUTPUT_GIF}")
    plt.close(fig2)

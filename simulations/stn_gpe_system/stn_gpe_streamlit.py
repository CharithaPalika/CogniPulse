import os
import sys
import tempfile
import warnings

import numpy as np
import yaml
import streamlit as st

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

from numpy.fft import fft
from scipy import signal

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="STN–GPe Network Simulator",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme / CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Global background */
    .stApp { background-color: #0d1117; color: #e6edf3; }

    # /* Sidebar */
    # [data-testid="stSidebar"] {
    #     background-color: #161b22;
    #     border-right: 1px solid #30363d;
    # }
    # [data-testid="stSidebar"] * { color: #e6edf3 !important; }
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }

    /* Labels only */
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] span {
        color: #e6edf3 !important;
    }

    /* Number inputs */
    .stNumberInput input {
        color: #000000 !important;
        background-color: #ffffff !important;
        -webkit-text-fill-color: #000000 !important;
    }

    /* Disabled number inputs */
    .stNumberInput input:disabled {
        color: #555555 !important;
        -webkit-text-fill-color: #555555 !important;
    }

    /* Selectbox text */
    div[data-baseweb="select"] > div {
        color: #000000 !important;
        background-color: #ffffff !important;
    }

    /* Selectbox input */
    div[data-baseweb="select"] input {
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
    }

    /* Dropdown menu */
    div[role="listbox"] {
        background-color: #ffffff !important;
    }

    div[role="option"] {
        color: #000000 !important;
    }

    /* Buttons inside number inputs */
    .stNumberInput button {
        color: #000000 !important;
    }

    /* Slider values */
    .stSlider span {
        color: #e6edf3 !important;
}



    /* Sidebar section labels */
    .sidebar-section {
        font-size: 0.70rem;
        font-weight: 700;
        letter-spacing: 0.09em;
        text-transform: uppercase;
        color: #8b949e !important;
        margin-top: 1.1rem;
        margin-bottom: 0.25rem;
    }

    /* Run button */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #1f6feb 0%, #388bfd 100%);
        color: #ffffff;
        border: none;
        border-radius: 8px;
        padding: 0.55rem 1.4rem;
        font-weight: 600;
        font-size: 0.95rem;
        width: 100%;
        transition: filter 0.15s ease;
    }
    div.stButton > button[kind="primary"]:hover { filter: brightness(1.15); }

    /* Metric cards */
    [data-testid="stMetric"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 0.8rem 1rem;
    }
    [data-testid="stMetricLabel"] { color: #8b949e !important; font-size: 0.78rem; }
    [data-testid="stMetricValue"] { color: #58a6ff !important; font-size: 1.3rem; font-weight: 700; }

    /* Section title */
    .section-title {
        font-size: 1rem;
        font-weight: 600;
        color: #58a6ff;
        border-bottom: 1px solid #21262d;
        padding-bottom: 0.3rem;
        margin-bottom: 0.8rem;
    }

    /* Mono result box */
    .result-box {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        font-family: "SFMono-Regular", Consolas, monospace;
        font-size: 0.82rem;
        color: #c9d1d9;
        white-space: pre-wrap;
        line-height: 1.65;
    }

    /* Expander header */
    [data-testid="stExpander"] summary {
        color: #8b949e;
        font-size: 0.85rem;
    }

    #MainMenu, footer { visibility: hidden; }
    header[data-testid="stHeader"] { background: transparent; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Project root ──────────────────────────────────────────────────────────────
try:
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    CURRENT_DIR = os.getcwd()

PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from stn_gpe import STN_GPe_loop, Analysis

warnings.filterwarnings("ignore")

# ── Plot palette ──────────────────────────────────────────────────────────────
PLOT_BG      = "#0d1117"
PLOT_SURFACE = "#161b22"
PLOT_GRID    = "#21262d"
PLOT_TEXT    = "#c9d1d9"
PLOT_SPINE   = "#30363d"

# Vibrant signal colours that pop on dark backgrounds
STN_COLOR = "#4dabf7"   # clear blue
GPE_COLOR = "#ff6b6b"   # coral red

# ── Dark spectrogram colormaps ─────────────────────────────────────────────
# STN: black → deep navy → electric blue → cyan
_stn_nodes = [
    (0.00, "#0d1117"),
    (0.20, "#0c2d48"),
    (0.50, "#1158a8"),
    (0.75, "#2684ff"),
    (1.00, "#79c8ff"),
]
cmap_stn_dark = LinearSegmentedColormap.from_list(
    "stn_dark", [(v, c) for v, c in _stn_nodes]
)

# GPe: black → deep burgundy → orange-red → amber
_gpe_nodes = [
    (0.00, "#0d1117"),
    (0.20, "#3d0c0c"),
    (0.50, "#a11515"),
    (0.75, "#e84343"),
    (1.00, "#ffae42"),
]
cmap_gpe_dark = LinearSegmentedColormap.from_list(
    "gpe_dark", [(v, c) for v, c in _gpe_nodes]
)


def _style_ax(ax, ylabel=None, xlabel=None, title=None):
    """Apply unified dark-theme styling to a single Axes."""
    ax.set_facecolor(PLOT_SURFACE)
    ax.tick_params(colors=PLOT_TEXT, labelsize=8, length=3, width=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_edgecolor(PLOT_SPINE)
        ax.spines[spine].set_linewidth(0.7)
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    if ylabel:
        ax.set_ylabel(ylabel, color=PLOT_TEXT, fontsize=8.5)
    if xlabel:
        ax.set_xlabel(xlabel, color=PLOT_TEXT, fontsize=8.5)
    if title:
        ax.set_title(title, color=PLOT_TEXT, fontsize=9.5, fontweight="bold", pad=5)


# ── YAML helper ───────────────────────────────────────────────────────────────
def create_yaml_from_inputs(params: dict) -> str:
    fd, tmp_path = tempfile.mkstemp(suffix=".yaml")
    with os.fdopen(fd, "w") as f:
        yaml.safe_dump(params, f)
    return tmp_path


# ── Core simulation ───────────────────────────────────────────────────────────
def run_stn_gpe_sim(
    stn_gpe_units, time_steps, dt,
    lat_sparse, inter_sparse,
    I_strd2_gpe,
    lat_strength_stn, lat_strength_gpe,
    wsg_strength, wgs_strength,
    I_gpe_ext, I_stn_ext,
    binsize, stn_gpe_noise,
    DBS, DBS_func,
    DBS_freq, DBS_duty, DBS_A1, DBS_A2,
    pulseinterval, center, spread_amplitude, sigma,
):
    # ── Type coercion ─────────────────────────────────────────────────────────
    stn_gpe_units   = int(stn_gpe_units)
    time_steps      = int(time_steps)
    dt              = float(dt)
    lat_sparse      = float(lat_sparse)
    inter_sparse    = float(inter_sparse)
    I_strd2_gpe     = float(I_strd2_gpe)
    lat_strength_stn = float(lat_strength_stn)
    lat_strength_gpe = float(lat_strength_gpe)
    wsg_strength    = float(wsg_strength)
    wgs_strength    = float(wgs_strength)
    I_gpe_ext       = float(I_gpe_ext)
    I_stn_ext       = float(I_stn_ext)
    binsize         = int(binsize)
    stn_gpe_noise   = float(stn_gpe_noise)
    DBS             = bool(DBS)
    DBS_func        = str(DBS_func)
    DBS_freq        = float(DBS_freq)
    DBS_duty        = float(DBS_duty)
    DBS_A1          = float(DBS_A1)
    DBS_A2          = float(DBS_A2)
    pulseinterval   = int(pulseinterval)
    center          = int(center)
    spread_amplitude = float(spread_amplitude)
    sigma           = float(sigma)

    # ── Build config & run ────────────────────────────────────────────────────
    config = dict(
        stn_gpe_units=stn_gpe_units, time=time_steps, dt=dt,
        lat_sparse=lat_sparse, inter_sparse=inter_sparse,
        I_strd2_gpe=I_strd2_gpe,
        lat_strength_stn=lat_strength_stn, lat_strength_gpe=lat_strength_gpe,
        wsg_strength=wsg_strength, wgs_strength=wgs_strength,
        I_gpe_ext=I_gpe_ext, I_stn_ext=I_stn_ext,
        binsize=binsize, stn_gpe_noise=stn_gpe_noise,
        DBS=DBS, DBS_func=DBS_func,
        DBS_freq=DBS_freq, DBS_duty=DBS_duty,
        DBS_A1=DBS_A1, DBS_A2=DBS_A2,
        pulseinterval=pulseinterval,
        center=center, spread_amplitude=spread_amplitude, sigma=sigma,
    )
    yaml_path = create_yaml_from_inputs(config)
    results   = STN_GPe_loop(yaml_path)

    # ── Unpack results ────────────────────────────────────────────────────────
    V_stn_time_all    = np.array(results["v_stn"])
    V_gpe_time_all    = np.array(results["v_gpe"])
    spike_monitor_stn = results["spike_stn"]
    spike_monitor_gpe = results["spike_gpe"]
    lfp_stn           = np.array(results["lfp_stn"])
    lfp_gpe           = np.array(results["lfp_gpe"])

    h   = dt
    sr  = int(round(1000.0 / h)) if h > 0 else 10000

    steps_for_one_second = int(round(1000.0 / h))
    window_steps = min(steps_for_one_second, time_steps)
    t_high  = time_steps
    t_low   = max(0, t_high - window_steps)
    t_chunk = np.linspace(0, (t_high - t_low) * h / 1000.0, t_high - t_low)

    # ── Analysis ──────────────────────────────────────────────────────────────
    analysis_STN = Analysis(spike_monitor_stn[t_low:t_high])
    analysis_GPe = Analysis(spike_monitor_gpe[t_low:t_high])

    pad = min(steps_for_one_second, t_low)
    lfp_smooth_stn = signal.savgol_filter(
        lfp_stn[t_low - pad:t_high], window_length=11, polyorder=5,
        deriv=0, delta=1.0, axis=-1, mode="interp", cval=0.0,
    )
    lfp_smooth_gpe = signal.savgol_filter(
        lfp_gpe[t_low - pad:t_high], window_length=11, polyorder=5,
        deriv=0, delta=1.0, axis=-1, mode="interp", cval=0.0,
    )

    fs          = sr
    window_size = sr
    nperseg     = window_size
    noverlap    = int(0.95 * window_size)

    f_stn, t_spec_stn, Sxx_stn = signal.spectrogram(
        lfp_smooth_stn, fs=fs, nperseg=nperseg, noverlap=noverlap, window="hamming"
    )
    f_gpe, t_spec_gpe, Sxx_gpe = signal.spectrogram(
        lfp_smooth_gpe, fs=fs, nperseg=nperseg, noverlap=noverlap, window="hamming"
    )

    avg_entropy_stn = analysis_STN.spectral_entropy(
        signal=lfp_stn[t_low:t_high], fs=fs, nperseg=nperseg, fmax=35, normalize=True
    )
    avg_entropy_gpe = analysis_GPe.spectral_entropy(
        signal=lfp_gpe[t_low:t_high], fs=fs, nperseg=nperseg, fmax=35, normalize=True
    )

    _, Ravg_stn = analysis_STN.synchrony()
    _, Ravg_gpe = analysis_GPe.synchrony()

    rate_data = analysis_STN.spike_rate(binsize=binsize)
    mean_std  = rate_data["mean_std"]

    frequency_avg_STN, _, _ = analysis_STN.frequency(dt=h)
    frequency_avg_GPe, _, _ = analysis_GPe.frequency(dt=h)

    # ── Build figure ──────────────────────────────────────────────────────────
    fig, axs = plt.subplots(
        4, 2,
        figsize=(11, 9),
        facecolor=PLOT_BG,
        gridspec_kw={"height_ratios": [1, 1, 1.2, 1.4]},
    )
    fig.subplots_adjust(wspace=0.32, left=0.07, right=0.97,
                        top=0.93, bottom=0.08, hspace=0.55)

    # Safe neuron indices
    stn_i = min(15, V_stn_time_all.shape[1] - 1)
    stn_j = min(15, V_stn_time_all.shape[2] - 1)
    gpe_i = min(7,  V_gpe_time_all.shape[1] - 1)
    gpe_j = min(7,  V_gpe_time_all.shape[2] - 1)

    # ── Row 0: Membrane voltage ───────────────────────────────────────────────
    _style_ax(axs[0, 0], ylabel="V (mV)", xlabel="Time (s)", title="STN Voltage")
    axs[0, 0].plot(
        t_chunk, V_stn_time_all[t_low:t_high, stn_i, stn_j],
        color=STN_COLOR, linewidth=0.8, alpha=0.9,
    )
    axs[0, 0].fill_between(
        t_chunk, V_stn_time_all[t_low:t_high, stn_i, stn_j],
        alpha=0.08, color=STN_COLOR,
    )

    _style_ax(axs[0, 1], ylabel="V (mV)", xlabel="Time (s)", title="GPe Voltage")
    axs[0, 1].plot(
        t_chunk, V_gpe_time_all[t_low:t_high, gpe_i, gpe_j],
        color=GPE_COLOR, linewidth=0.8, alpha=0.9,
    )
    axs[0, 1].fill_between(
        t_chunk, V_gpe_time_all[t_low:t_high, gpe_i, gpe_j],
        alpha=0.08, color=GPE_COLOR,
    )

    # ── Row 1: LFP ───────────────────────────────────────────────────────────
    _style_ax(axs[1, 0], ylabel="LFP (mV)", xlabel="Time (s)", title="STN LFP")
    axs[1, 0].plot(t_chunk, lfp_stn[t_low:t_high], color=STN_COLOR, linewidth=0.85)
    axs[1, 0].fill_between(t_chunk, lfp_stn[t_low:t_high], alpha=0.10, color=STN_COLOR)

    _style_ax(axs[1, 1], ylabel="LFP (mV)", xlabel="Time (s)", title="GPe LFP")
    axs[1, 1].plot(t_chunk, lfp_gpe[t_low:t_high], color=GPE_COLOR, linewidth=0.85)
    axs[1, 1].fill_between(t_chunk, lfp_gpe[t_low:t_high], alpha=0.10, color=GPE_COLOR)

    # ── Row 2: Spike rasters ──────────────────────────────────────────────────
    spike_array_stn  = np.array(spike_monitor_stn[t_low:t_high])
    num_neurons_stn  = spike_array_stn.shape[1]
    t_raster_stn     = np.linspace(0, spike_array_stn.shape[0] * h / 1000.0, spike_array_stn.shape[0])

    _style_ax(axs[2, 0], ylabel="Neuron", xlabel="t (s)", title="STN Raster")
    for n_idx in range(num_neurons_stn):
        spike_times = t_raster_stn[spike_array_stn[:, n_idx] == 1]
        if len(spike_times):
            axs[2, 0].scatter(
                spike_times,
                np.full_like(spike_times, n_idx + 1),
                color=STN_COLOR, s=1.2, alpha=0.75, linewidths=0,
            )
    axs[2, 0].set_ylim(0.5, num_neurons_stn + 0.5)
    axs[2, 0].set_yticks([1, num_neurons_stn // 2, num_neurons_stn])

    spike_array_gpe  = np.array(spike_monitor_gpe[t_low:t_high])
    num_neurons_gpe  = spike_array_gpe.shape[1]
    t_raster_gpe     = np.linspace(0, spike_array_gpe.shape[0] * h / 1000.0, spike_array_gpe.shape[0])

    _style_ax(axs[2, 1], ylabel="Neuron", xlabel="t (s)", title="GPe Raster")
    for n_idx in range(num_neurons_gpe):
        spike_times = t_raster_gpe[spike_array_gpe[:, n_idx] == 1]
        if len(spike_times):
            axs[2, 1].scatter(
                spike_times,
                np.full_like(spike_times, n_idx + 1),
                color=GPE_COLOR, s=1.2, alpha=0.75, linewidths=0,
            )
    axs[2, 1].set_ylim(0.5, num_neurons_gpe + 0.5)
    axs[2, 1].set_yticks([1, num_neurons_gpe // 2, num_neurons_gpe])

    # ── Row 3: Spectrograms ───────────────────────────────────────────────────
    _style_ax(axs[3, 0], ylabel="Frequency (Hz)", xlabel="Time (s)", title="STN Spectrogram")
    pcm_stn = axs[3, 0].pcolormesh(
        t_spec_stn, f_stn,
        10 * np.log10(Sxx_stn + 1e-12),
        cmap=cmap_stn_dark, shading="gouraud",
    )
    axs[3, 0].set_ylim(0, 40)
    cb_stn = fig.colorbar(pcm_stn, ax=axs[3, 0], pad=0.02, fraction=0.045)
    cb_stn.ax.tick_params(colors=PLOT_TEXT, labelsize=7)
    cb_stn.set_label("dB", color=PLOT_TEXT, fontsize=7.5)
    cb_stn.outline.set_edgecolor(PLOT_SPINE)

    _style_ax(axs[3, 1], ylabel="Frequency (Hz)", xlabel="Time (s)", title="GPe Spectrogram")
    pcm_gpe = axs[3, 1].pcolormesh(
        t_spec_gpe, f_gpe,
        10 * np.log10(Sxx_gpe + 1e-12),
        cmap=cmap_gpe_dark, shading="gouraud",
    )
    axs[3, 1].set_ylim(0, 40)
    cb_gpe = fig.colorbar(pcm_gpe, ax=axs[3, 1], pad=0.02, fraction=0.045)
    cb_gpe.ax.tick_params(colors=PLOT_TEXT, labelsize=7)
    cb_gpe.set_label("dB", color=PLOT_TEXT, fontsize=7.5)
    cb_gpe.outline.set_edgecolor(PLOT_SPINE)

    # ── Column header labels ──────────────────────────────────────────────────
    for ax, label, color in [
        (axs[0, 0], "● STN", STN_COLOR),
        (axs[0, 1], "● GPe", GPE_COLOR),
    ]:
        ax.annotate(
            label, xy=(0.01, 1.08), xycoords="axes fraction",
            color=color, fontsize=9, fontweight="bold",
        )

    # ── Shared legend ─────────────────────────────────────────────────────────
    legend_elements = [
        Line2D([0], [0], color=STN_COLOR, label="STN", linewidth=2.5),
        Line2D([0], [0], color=GPE_COLOR, label="GPe", linewidth=2.5),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center", bbox_to_anchor=(0.5, 0.005),
        ncol=2, frameon=False,
        labelcolor=PLOT_TEXT, fontsize=9,
    )

    # ── Metrics dict ─────────────────────────────────────────────────────────
    metrics = {
        "entropy_stn":   float(avg_entropy_stn),
        "entropy_gpe":   float(avg_entropy_gpe),
        "synchrony_stn": float(Ravg_stn),
        "synchrony_gpe": float(Ravg_gpe),
        "rate_std":      float(mean_std),
        "freq_stn":      float(frequency_avg_STN),
        "freq_gpe":      float(frequency_avg_GPe),
    }

    return fig, metrics


# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown("# ⚡ STN–GPe Network Simulation")
st.markdown(
    "<span style='color:#8b949e;font-size:0.9rem;'>"
    "Biophysical basal-ganglia model · LFP · Raster · Spectrogram"
    "</span>",
    unsafe_allow_html=True,
)
st.markdown("---")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Parameters")

    # ── Network ──────────────────────────────────────────────────────────────
    st.markdown('<p class="sidebar-section">Network</p>', unsafe_allow_html=True)
    stn_gpe_units = st.slider("Units (STN & GPe)", 1, 64, value=16, step=1)
    time_steps    = st.number_input("Time steps", value=50000, step=1000, min_value=1000)
    dt            = st.number_input("dt (ms)", value=0.1, step=0.05, format="%.3f")

    # ── Connectivity ─────────────────────────────────────────────────────────
    st.markdown('<p class="sidebar-section">Connectivity</p>', unsafe_allow_html=True)
    lat_sparse      = st.number_input("lat_sparse",      value=0.10, step=0.05, format="%.3f")
    inter_sparse    = st.number_input("inter_sparse",    value=0.10, step=0.05, format="%.3f")
    lat_strength_stn = st.number_input("lat_strength_stn", value=0.02, step=0.01, format="%.3f")
    lat_strength_gpe = st.number_input("lat_strength_gpe", value=0.10, step=0.01, format="%.3f")
    wsg_strength    = st.number_input("wsg_strength",    value=0.10, step=0.01, format="%.3f")
    wgs_strength    = st.number_input("wgs_strength",    value=0.10, step=0.01, format="%.3f")

    # ── Currents ─────────────────────────────────────────────────────────────
    st.markdown('<p class="sidebar-section">Currents & Noise</p>', unsafe_allow_html=True)
    I_strd2_gpe   = st.number_input("I_strd2_gpe",   value=5.0,  step=0.5,  format="%.2f")
    I_gpe_ext     = st.number_input("I_gpe_ext",     value=6.0,  step=0.5,  format="%.2f")
    I_stn_ext     = st.number_input("I_stn_ext",     value=12.0, step=0.5,  format="%.2f")
    stn_gpe_noise = st.number_input("stn_gpe_noise", value=0.0,  step=0.5,  format="%.2f")

    # ── Spike analysis ────────────────────────────────────────────────────────
    st.markdown('<p class="sidebar-section">Spike Analysis</p>', unsafe_allow_html=True)
    binsize = st.number_input("binsize", value=100, step=10, min_value=10)

    # ── DBS ───────────────────────────────────────────────────────────────────
    st.markdown('<p class="sidebar-section">DBS</p>', unsafe_allow_html=True)
    DBS     = st.checkbox("Enable DBS", value=False)
    DBS_func = st.selectbox(
        "DBS waveform",
        ["monophasicDBS", "biphasicDBS", "biphasicDBS_uninoise", "biphasicDBS_normalnoise"],
        index=1,
        disabled=not DBS,
    )
    dbs_col1, dbs_col2 = st.columns(2)
    with dbs_col1:
        DBS_freq  = st.number_input("Freq (Hz)", value=130.0, step=5.0,  format="%.1f", disabled=not DBS)
        DBS_A1    = st.number_input("A1",        value=250.0, step=10.0, format="%.1f", disabled=not DBS)
        pulseinterval = st.number_input("Pulse interval", value=10, step=1, min_value=1, disabled=not DBS)
    with dbs_col2:
        DBS_duty  = st.number_input("Duty",      value=0.052, step=0.005, format="%.3f", disabled=not DBS)
        DBS_A2    = st.number_input("A2",        value=-250.0, step=10.0, format="%.1f", disabled=not DBS)

    # ── DBS spread ────────────────────────────────────────────────────────────
    st.markdown('<p class="sidebar-section">DBS Spatial Spread</p>', unsafe_allow_html=True)
    center           = st.number_input("center",           value=7,   step=1,   min_value=0, disabled=not DBS)
    spread_amplitude = st.number_input("spread_amplitude", value=1.0, step=0.1, format="%.2f", disabled=not DBS)
    sigma            = st.number_input("sigma",            value=3.0, step=0.5, format="%.2f", disabled=not DBS)

    st.markdown("---")
    run_clicked = st.button("▶  Run Simulation", type="primary")


# ── Main panel ────────────────────────────────────────────────────────────────
if run_clicked:
    with st.spinner("Running simulation — may take a minute…"):
        try:
            fig, metrics = run_stn_gpe_sim(
                stn_gpe_units, time_steps, dt,
                lat_sparse, inter_sparse,
                I_strd2_gpe,
                lat_strength_stn, lat_strength_gpe,
                wsg_strength, wgs_strength,
                I_gpe_ext, I_stn_ext,
                binsize, stn_gpe_noise,
                DBS, DBS_func,
                DBS_freq, DBS_duty, DBS_A1, DBS_A2,
                pulseinterval, center, spread_amplitude, sigma,
            )
            st.session_state["fig"]     = fig
            st.session_state["metrics"] = metrics
        except Exception as e:
            st.error(f"Simulation failed: {e}")
            st.stop()

if "fig" in st.session_state:
    fig     = st.session_state["fig"]
    metrics = st.session_state["metrics"]

    # ── Metric cards ──────────────────────────────────────────────────────────
    st.markdown('<p class="section-title">Summary Metrics</p>', unsafe_allow_html=True)
    c1, c2, = st.columns(2)
    # c1.metric("Entropy — STN",   f"{metrics['entropy_stn']:.3f}")
    # c2.metric("Entropy — GPe",   f"{metrics['entropy_gpe']:.3f}")
    c1.metric("Synchrony — STN", f"{metrics['synchrony_stn']:.3f}")
    # c4.metric("Synchrony — GPe", f"{metrics['synchrony_gpe']:.3f}")
    c2.metric("Rate Std Dev",    f"{metrics['rate_std']:.3f}")

    st.markdown("&nbsp;", unsafe_allow_html=True)

    # ── Plot ──────────────────────────────────────────────────────────────────
    st.markdown('<p class="section-title">Simulation Results</p>', unsafe_allow_html=True)
    st.pyplot(fig, use_container_width=True)

    # ── Frequency expander ────────────────────────────────────────────────────
    with st.expander("Firing frequency", expanded=False):
        fc1, fc2 = st.columns(2)
        fc1.metric("Avg freq — STN (Hz)", f"{metrics['freq_stn']:.2f}")
        fc2.metric("Avg freq — GPe (Hz)", f"{metrics['freq_gpe']:.2f}")

    # ── Raw metrics ───────────────────────────────────────────────────────────
    with st.expander("Raw metrics", expanded=False):
        raw = (
            f"Spectral entropy  STN : {metrics['entropy_stn']:.5f}\n"
            f"Spectral entropy  GPe : {metrics['entropy_gpe']:.5f}\n\n"
            f"Synchrony (Ravg)  STN : {metrics['synchrony_stn']:.5f}\n"
            f"Synchrony (Ravg)  GPe : {metrics['synchrony_gpe']:.5f}\n\n"
            f"Rate std dev      STN : {metrics['rate_std']:.5f}\n\n"
            f"Avg firing freq   STN : {metrics['freq_stn']:.3f} Hz\n"
            f"Avg firing freq   GPe : {metrics['freq_gpe']:.3f} Hz\n"
        )
        st.markdown(f'<div class="result-box">{raw}</div>', unsafe_allow_html=True)

else:
    # ── Empty state ───────────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="margin-top:4rem;text-align:center;color:#8b949e;">
            <div style="font-size:3rem;margin-bottom:1rem;">⚡</div>
            <div style="font-size:1.1rem;font-weight:600;color:#58a6ff;">
                Configure the network in the sidebar, then press <em>Run Simulation</em>.
            </div>
            <div style="font-size:0.85rem;margin-top:0.5rem;">
                Voltage traces · LFP · Spike rasters · Spectrograms will appear here.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

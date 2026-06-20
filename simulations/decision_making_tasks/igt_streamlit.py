import os
import sys
import warnings

import numpy as np
import torch
import streamlit as st

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

# ── Page config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="Iowa Gambling Task",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme / CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Global background ── */
    .stApp {
        background-color: #0d1117;
        color: #e6edf3;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    [data-testid="stSidebar"] * {
        color: #e6edf3 !important;
    }

    /* ── Sidebar section headers ── */
    .sidebar-section {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #8b949e !important;
        margin-top: 1.2rem;
        margin-bottom: 0.3rem;
    }

    /* ── Primary button ── */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #238636 0%, #2ea043 100%);
        color: #ffffff;
        border: none;
        border-radius: 8px;
        padding: 0.55rem 1.4rem;
        font-weight: 600;
        font-size: 0.95rem;
        width: 100%;
        transition: filter 0.15s ease;
    }
    div.stButton > button[kind="primary"]:hover {
        filter: brightness(1.15);
    }

    /* ── Metric cards ── */
    [data-testid="stMetric"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 0.8rem 1rem;
    }
    [data-testid="stMetricLabel"] { color: #8b949e !important; font-size: 0.8rem; }
    [data-testid="stMetricValue"] { color: #58a6ff !important; font-size: 1.4rem; font-weight: 700; }

    /* ── Info/result box ── */
    .result-box {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        font-family: "SFMono-Regular", Consolas, monospace;
        font-size: 0.83rem;
        color: #c9d1d9;
        white-space: pre-wrap;
        line-height: 1.6;
    }

    /* ── Section divider ── */
    .section-title {
        font-size: 1rem;
        font-weight: 600;
        color: #58a6ff;
        border-bottom: 1px solid #21262d;
        padding-bottom: 0.3rem;
        margin-bottom: 0.8rem;
    }

    /* ── Hide Streamlit chrome ── */
    #MainMenu, footer { visibility: hidden; }
    header[data-testid="stHeader"] { background: transparent; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Project root setup ────────────────────────────────────────────────────────
try:
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    CURRENT_DIR = os.getcwd()

PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from envs import *            # defines IGTEnv
from basal_ganglia import *   # defines train(...)
from stn_gpe import load_yaml

warnings.filterwarnings("ignore")

IGT_PARAMS_PATH = os.path.join(
    PROJECT_ROOT, "params", "decision_making_task_params", "igt_params.yaml"
)
STN_PARAM_DIR = os.path.join(PROJECT_ROOT, "params", "stn_gpe_params")

# ── Plot style helpers ────────────────────────────────────────────────────────
PLOT_BG      = "#0d1117"
PLOT_SURFACE = "#161b22"
PLOT_GRID    = "#21262d"
PLOT_TEXT    = "#000000"
PLOT_SPINE   = "#30363d"

COLOR_A = "#ff7b7b"   # disadvantageous – coral red
COLOR_B = "#ff4f4f"   # disadvantageous – deeper red
COLOR_C = "#3fb950"   # advantageous    – bright green
COLOR_D = "#56d364"   # advantageous    – lighter green
COLOR_BAR = "#58a6ff" # IGT score bars  – blue accent
COLOR_ERR = "#e6edf3" # error bars      – near-white

def _style_ax(ax):
    """Apply dark-theme styling to a matplotlib Axes."""
    ax.set_facecolor(PLOT_SURFACE)
    ax.tick_params(colors=PLOT_TEXT, labelsize=9)
    ax.xaxis.label.set_color(PLOT_TEXT)
    ax.yaxis.label.set_color(PLOT_TEXT)
    ax.title.set_color(PLOT_TEXT)
    for spine in ax.spines.values():
        spine.set_edgecolor(PLOT_SPINE)
    ax.grid(color=PLOT_GRID, linestyle="--", linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)


# ── Core simulation ───────────────────────────────────────────────────────────
def run_igt_task(use_stn_gpe, stn_mode, d1_amp, d2_amp, var, gpi_mean):
    d1_amp   = float(d1_amp)
    d2_amp   = float(d2_amp)
    var      = float(var)
    gpi_mean = float(gpi_mean)

    params       = load_yaml(IGT_PARAMS_PATH)
    TRIALS       = int(params["TRIALS"])
    EPOCHS       = int(params["EPOCHS"])
    NUM_BINS     = int(params["NUM_BINS"])
    LR           = float(params["LR"])
    NUM_ARMS     = int(params["NUM_ARMS"])
    SCALING_FACTOR = float(params["SCALING_FACTOR"])

    REW_STD  = 0
    LOSS_STD = 0

    mean_reward = np.array([100, 100, 50,   50])   / SCALING_FACTOR
    std_reward  = np.array([REW_STD] * 4)          / SCALING_FACTOR
    mean_loss   = np.array([-250, -1250, -50, -250]) / SCALING_FACTOR
    std_loss    = np.array([LOSS_STD] * 4)          / SCALING_FACTOR

    env = IGTEnv(
        mean_reward=mean_reward,
        std_reward=std_reward,
        mean_loss=mean_loss,
        std_loss=std_loss,
    )

    if use_stn_gpe:
        mode_to_file = {
            "normal":   "params_Normal.yaml",
            "PD":       "params_PD.yaml",
            "std_DBS":  "params_std_DBS.yaml",
        }
        STN_DATA_path = os.path.join(STN_PARAM_DIR, mode_to_file[stn_mode])
    else:
        STN_DATA_path = None

    reward_monitor, arm_chosen_monitor, avg_counts, ip_monitor, dp_monitor, ep_monitor, _ = train(
        env,
        trails=TRIALS,
        epochs=EPOCHS,
        lr=LR,
        bins=NUM_BINS,
        STN_data=STN_DATA_path,
        d1_amp=d1_amp,
        d2_amp=d2_amp,
        gpi_threshold=0.15,
        max_gpi_iters=50,
        del_lim=None,
        train_IP=False,
        del_med=None,
        printing=False,
        gpi_mean=gpi_mean,
        ep_0=var,
    )

    A_picks = avg_counts[0]
    B_picks = avg_counts[1]
    C_picks = avg_counts[2]
    D_picks = avg_counts[3]

    Avg_A = torch.mean(A_picks, dim=0).detach().cpu().numpy()
    Avg_B = torch.mean(B_picks, dim=0).detach().cpu().numpy()
    Avg_C = torch.mean(C_picks, dim=0).detach().cpu().numpy()
    Avg_D = torch.mean(D_picks, dim=0).detach().cpu().numpy()

    IGT_score = torch.mean(
        torch.add(C_picks, D_picks) - torch.add(A_picks, B_picks), dim=0
    )
    IGT_dev = torch.std(
        torch.add(C_picks, D_picks) - torch.add(A_picks, B_picks), dim=0
    ) / torch.sqrt(torch.tensor(EPOCHS, dtype=torch.float32))

    IGT_score_np = IGT_score.detach().cpu().numpy().reshape(-1)
    IGT_dev_np   = IGT_dev.detach().cpu().numpy().reshape(-1)
    BINS         = np.arange(NUM_BINS) + 1

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
    fig.patch.set_facecolor(PLOT_BG)

    # Left: avg picks per bin
    _style_ax(ax1)
    ax1.plot(BINS, Avg_A, label="Deck A", color=COLOR_A, marker="o",
             linewidth=2, markersize=5, markeredgecolor="white", markeredgewidth=0.4)
    ax1.plot(BINS, Avg_B, label="Deck B", color=COLOR_B, marker="s",
             linewidth=2, markersize=5, markeredgecolor="white", markeredgewidth=0.4)
    ax1.plot(BINS, Avg_C, label="Deck C", color=COLOR_C, marker="D",
             linewidth=2, markersize=5, markeredgecolor="white", markeredgewidth=0.4)
    ax1.plot(BINS, Avg_D, label="Deck D", color=COLOR_D, marker="^",
             linewidth=2, markersize=5, markeredgecolor="white", markeredgewidth=0.4)
    ax1.set_title("Average card picks per bin", fontsize=11, fontweight="bold", pad=10)
    ax1.set_xlabel("Bin", fontsize=10)
    ax1.set_ylabel("Avg picks", fontsize=10)
    legend = ax1.legend(loc="upper right", framealpha=0.25, edgecolor=PLOT_SPINE,
                        labelcolor=PLOT_TEXT, fontsize=9)
    legend.get_frame().set_facecolor(PLOT_SURFACE)

    # Right: IGT score per bin
    _style_ax(ax2)
    bars = ax2.bar(BINS, IGT_score_np, color=COLOR_BAR, alpha=0.85,
                   edgecolor=PLOT_SPINE, linewidth=0.6, zorder=3)
    # Colour negative bars differently
    for bar, val in zip(bars, IGT_score_np):
        if val < 0:
            bar.set_color("#f85149")
            bar.set_alpha(0.85)
    ax2.errorbar(BINS, IGT_score_np, yerr=IGT_dev_np,
                 fmt="none", ecolor=COLOR_ERR, elinewidth=1.5, capsize=4, capthick=1.5, zorder=4)
    ax2.axhline(0, color=PLOT_SPINE, linewidth=1, linestyle="--", zorder=2)
    ax2.set_title("IGT score per bin", fontsize=11, fontweight="bold", pad=10)
    ax2.set_xlabel("Bin", fontsize=10)
    ax2.set_ylabel("IGT score", fontsize=10)

    fig.tight_layout(pad=2.0)

    overall_mean = float(IGT_score_np.mean())
    overall_se   = float(np.sqrt((IGT_dev_np ** 2).mean()))

    metrics = {
        "IGT score per bin":  IGT_score_np,
        "IGT SE per bin":     IGT_dev_np,
        "Overall mean score": overall_mean,
        "Approx. overall SE": overall_se,
        "Mode": stn_mode if use_stn_gpe else "Noise only",
        "STN–GPe enabled": use_stn_gpe,
    }

    return fig, metrics


# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown("# 🧠 Iowa Gambling Task Simulation")
st.markdown(
    "<span style='color:#8b949e;font-size:0.9rem;'>"
    "Basal-ganglia model · STN–GPe drive or noise · Decision-making task"
    "</span>",
    unsafe_allow_html=True,
)
st.markdown("---")

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Parameters")

    st.markdown('<p class="sidebar-section">STN–GPe Drive</p>', unsafe_allow_html=True)
    use_stn_gpe = st.checkbox("Use STN–GPe simulation", value=False,
                               help="When enabled, beta-band LFP from a simulated STN–GPe network replaces additive noise.")
    stn_mode = st.selectbox(
        "Condition",
        ["normal", "PD", "std_DBS"],
        disabled=not use_stn_gpe,
        help="Normal healthy state, Parkinsonian, or DBS-treated.",
    )

    st.markdown('<p class="sidebar-section">Model Parameters</p>', unsafe_allow_html=True)
    d1_amp   = st.number_input("D1 amplitude",        value=0.50, step=0.01, format="%.3f")
    d2_amp   = st.number_input("D2 amplitude",        value=0.02, step=0.01, format="%.3f")
    gpi_mean = st.number_input("GPI mean",            value=1.00, step=0.10, format="%.2f")
    var      = st.number_input("ε₀ / reward std",     value=0.00, step=0.10, format="%.2f",
                                help="Initial exploration parameter / reward/loss noise std.")

    st.markdown("---")
    run_clicked = st.button("▶  Run Simulation", type="primary")

# ── Main panel ────────────────────────────────────────────────────────────────
if run_clicked:
    with st.spinner("Running simulation — this may take a moment…"):
        try:
            fig, metrics = run_igt_task(use_stn_gpe, stn_mode, d1_amp, d2_amp, var, gpi_mean)
            st.session_state["fig"]     = fig
            st.session_state["metrics"] = metrics
        except Exception as e:
            st.error(f"Simulation failed: {e}")
            st.stop()

if "fig" in st.session_state:
    fig     = st.session_state["fig"]
    metrics = st.session_state["metrics"]

    # ── Metric cards ──────────────────────────────────────────────────────────
    st.markdown('<p class="section-title">Summary</p>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    col1.metric("Lastbin IGT Score", f"{metrics['IGT score per bin'][-1]:.3f}")
    col2.metric("Approx. SE",        f"{metrics['Approx. overall SE']:.3f}")
    col3.metric("Condition",         metrics["Mode"])

    st.markdown("&nbsp;", unsafe_allow_html=True)

    # ── Plots ─────────────────────────────────────────────────────────────────
    st.markdown('<p class="section-title">Results</p>', unsafe_allow_html=True)
    st.pyplot(fig, use_container_width=True)

    # ── Per-bin breakdown ─────────────────────────────────────────────────────
    with st.expander("Per-bin breakdown", expanded=False):
        score_np = metrics["IGT score per bin"]
        se_np    = metrics["IGT SE per bin"]
        rows = [
            {"Bin": int(i + 1),
             "IGT Score": f"{s:.4f}",
             "SE":        f"{e:.4f}"}
            for i, (s, e) in enumerate(zip(score_np, se_np))
        ]
        st.table(rows)

    # ── Raw metrics text ──────────────────────────────────────────────────────
    with st.expander("Raw metrics", expanded=False):
        raw = (
            f"STN–GPe enabled : {metrics['STN–GPe enabled']}\n"
            f"Condition        : {metrics['Mode']}\n\n"
            f"IGT score/bin    : {score_np}\n"
            f"IGT SE/bin       : {se_np}\n\n"
            f"Overall mean     : {metrics['Overall mean score']:.5f}\n"
            f"Approx. SE       : {metrics['Approx. overall SE']:.5f}\n"
        )
        st.markdown(f'<div class="result-box">{raw}</div>', unsafe_allow_html=True)

else:
    # ── Empty-state prompt ────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="
            margin-top: 4rem;
            text-align: center;
            color: #8b949e;
        ">
            <div style="font-size: 3rem; margin-bottom: 1rem;">🧠</div>
            <div style="font-size: 1.1rem; font-weight: 600; color: #58a6ff;">
                Configure parameters in the sidebar, then press <em>Run Simulation</em>.
            </div>
            <div style="font-size: 0.85rem; margin-top: 0.5rem;">
                Results — deck picks, IGT score, and per-bin breakdown — will appear here.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

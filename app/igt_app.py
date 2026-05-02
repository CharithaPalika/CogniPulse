import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─────────────────────────────────────────────
# IGT Environment (unchanged from your code)
# ─────────────────────────────────────────────
class IGTEnv():
    def __init__(self,
                 mean_reward: np.ndarray,
                 std_reward: np.ndarray,
                 mean_loss: np.ndarray,
                 std_loss: np.ndarray):
        self.num_arms = 4
        self.mean_reward = mean_reward
        self.std_reward = std_reward
        self.mean_loss = mean_loss
        self.std_loss = std_loss
        self.loss_tstamps = {0: np.array([0,1,0,1,0,1,0,1,0,1]),
                             1: np.array([0,0,0,0,1,0,0,0,0,0]),
                             2: np.array([0,1,0,1,0,1,0,1,0,1]),
                             3: np.array([0,0,0,0,1,0,0,0,0,0])}
        for key in self.loss_tstamps:
            np.random.shuffle(self.loss_tstamps[key])
        self.counts = np.zeros((self.num_arms))
        assert self.num_arms == self.mean_reward.shape[0] == self.std_reward.shape[0] == self.mean_loss.shape[0] == self.std_loss.shape[0]
        self.arms = dict(enumerate(zip(self.mean_reward, self.std_reward, self.mean_loss, self.std_loss)))

    def step(self, chosen_arm):
        arm_mean_rew, arm_dev_rew, arm_mean_loss, arm_dev_loss = self.arms[chosen_arm]
        gain = np.random.normal(arm_mean_rew, arm_dev_rew)
        loss = np.random.normal(arm_mean_loss, arm_dev_loss)
        loss = loss * self.loss_tstamps[chosen_arm][int(self.counts[chosen_arm])]
        net  = gain + loss
        self.counts[chosen_arm] += 1
        if self.counts[chosen_arm] == 10:
            self.counts[chosen_arm] = 0
            np.random.shuffle(self.loss_tstamps[chosen_arm])
        return gain, loss, net

    def reset(self):
        self.loss_tstamps = {0: np.array([0,1,0,1,0,1,0,1,0,1]),
                             1: np.array([0,0,0,0,1,0,0,0,0,0]),
                             2: np.array([0,1,0,1,0,1,0,1,0,1]),
                             3: np.array([0,0,0,0,1,0,0,0,0,0])}
        for key in self.loss_tstamps:
            np.random.shuffle(self.loss_tstamps[key])
        self.counts = np.zeros((self.num_arms))
        self.arms = dict(enumerate(zip(self.mean_reward, self.std_reward, self.mean_loss, self.std_loss)))

# ─────────────────────────────────────────────
# Standard Bechara et al. IGT parameters
# Deck A & C → high reward, high & frequent loss (bad decks)
# Deck B & D → low reward, high but infrequent loss (good decks)
# ─────────────────────────────────────────────
DEFAULT_PARAMS = dict(
    mean_reward = np.array([100.0, 100.0, 50.0, 50.0]),
    std_reward  = np.array([10.0,  5.0,  10.0,  5.0]),
    mean_loss   = np.array([-250.0, -1250.0, -50.0, -250.0]),
    std_loss    = np.array([10.0,   10.0,   10.0,   10.0]),
)
MAX_TRIALS  = 100
STARTING_BANKROLL = 2000.0
DECK_LABELS = ["A", "B", "C", "D"]
DECK_COLORS = ["#E05C5C", "#5C9BE0", "#E0A85C", "#5CB87A"]

# ─────────────────────────────────────────────
# Session state bootstrap
# ─────────────────────────────────────────────
def init_state():
    if "env" not in st.session_state:
        env = IGTEnv(**DEFAULT_PARAMS)
        st.session_state.env         = env
        st.session_state.bankroll    = STARTING_BANKROLL
        st.session_state.history     = []       # list of (trial, deck_label, reward, bankroll)
        st.session_state.trial       = 0
        st.session_state.last_result = None     # dict with trial info for the flash card
        st.session_state.game_over   = False

def reset_game():
    st.session_state.env.reset()
    st.session_state.bankroll    = STARTING_BANKROLL
    st.session_state.history     = []
    st.session_state.trial       = 0
    st.session_state.last_result = None
    st.session_state.game_over   = False

def pick_deck(arm_idx: int):
    if st.session_state.game_over:
        return
    gain, loss, net = st.session_state.env.step(arm_idx)
    st.session_state.bankroll += net
    st.session_state.trial    += 1
    result = dict(
        trial    = st.session_state.trial,
        deck     = DECK_LABELS[arm_idx],
        arm      = arm_idx,
        gain     = gain,
        loss     = loss,
        reward   = net,
        bankroll = st.session_state.bankroll,
    )
    st.session_state.history.append(result)
    st.session_state.last_result = result
    if st.session_state.trial >= MAX_TRIALS:
        st.session_state.game_over = True

# ─────────────────────────────────────────────
# Page config & global CSS
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Iowa Gambling Task",
    page_icon="🃏",
    layout="centered",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0f0f13;
    color: #e8e4dc;
}

/* ── header ── */
.igt-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.6rem;
    font-weight: 900;
    letter-spacing: -0.02em;
    color: #f5f0e8;
    line-height: 1.1;
    margin-bottom: 0;
}
.igt-subtitle {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.18em;
    color: #7a7468;
    text-transform: uppercase;
    margin-top: 4px;
    margin-bottom: 24px;
}

/* ── bankroll badge ── */
.bankroll-box {
    background: #1a1a22;
    border: 1px solid #2e2e3a;
    border-radius: 12px;
    padding: 14px 24px;
    display: flex;
    align-items: baseline;
    gap: 10px;
    margin-bottom: 28px;
}
.bankroll-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    color: #5a5860;
    text-transform: uppercase;
}
.bankroll-amount {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: -0.01em;
}
.bankroll-trial {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: #5a5860;
    margin-left: auto;
    letter-spacing: 0.12em;
}

/* ── deck buttons ── */
.deck-row {
    display: flex;
    gap: 14px;
    justify-content: center;
    margin-bottom: 24px;
}
.deck-btn {
    flex: 1;
    max-width: 130px;
    aspect-ratio: 2/3;
    border-radius: 14px;
    border: none;
    cursor: pointer;
    font-family: 'Playfair Display', serif;
    font-size: 2.4rem;
    font-weight: 900;
    color: #fff;
    transition: transform 0.12s ease, box-shadow 0.12s ease;
    position: relative;
    box-shadow: 0 6px 20px rgba(0,0,0,0.4);
    display: flex;
    align-items: center;
    justify-content: center;
    flex-direction: column;
}
.deck-btn:hover   { transform: translateY(-5px) scale(1.03); box-shadow: 0 14px 32px rgba(0,0,0,0.55); }
.deck-btn:active  { transform: translateY(0px) scale(0.98); }
.deck-btn .deck-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.55rem;
    letter-spacing: 0.2em;
    opacity: 0.7;
    margin-top: 4px;
    text-transform: uppercase;
}

/* ── result flash card ── */
.result-card {
    background: #1a1a22;
    border-radius: 12px;
    border: 1px solid #2e2e3a;
    padding: 16px 20px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 16px;
}
.result-deck-badge {
    width: 42px; height: 56px;
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-family: 'Playfair Display', serif;
    font-size: 1.5rem; font-weight: 900; color: white;
    flex-shrink: 0;
}
.result-value {
    font-family: 'DM Mono', monospace;
    font-size: 1.4rem;
    font-weight: 500;
}
.result-desc {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.75rem;
    color: #7a7468;
    margin-top: 2px;
}

/* ── game over banner ── */
.gameover-banner {
    background: linear-gradient(135deg, #1e1e2e, #2a1a2e);
    border: 1px solid #4a2a5a;
    border-radius: 14px;
    padding: 28px 24px;
    text-align: center;
    margin-bottom: 24px;
}
.gameover-title {
    font-family: 'Playfair Display', serif;
    font-size: 2rem; font-weight: 900;
    color: #c89edc;
    margin-bottom: 8px;
}
.gameover-score {
    font-family: 'DM Mono', monospace;
    font-size: 1.1rem;
    color: #e8e4dc;
}

/* ── progress bar ── */
.progress-wrap {
    background: #1a1a22;
    border-radius: 999px;
    height: 6px;
    margin-bottom: 28px;
    overflow: hidden;
}
.progress-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #5C9BE0, #5CB87A);
    transition: width 0.3s ease;
}

/* Streamlit overrides */
.block-container { max-width: 680px; padding-top: 2rem; }
div[data-testid="stButton"] button {
    background: transparent;
    border: 1px solid #2e2e3a;
    color: #9a9490;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.15em;
    border-radius: 8px;
    padding: 8px 20px;
    transition: all 0.15s ease;
}
div[data-testid="stButton"] button:hover {
    border-color: #5C9BE0;
    color: #5C9BE0;
    background: rgba(92, 155, 224, 0.07);
}
section[data-testid="stSidebar"] { background: #0d0d10; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Init
# ─────────────────────────────────────────────
init_state()

# ─────────────────────────────────────────────
# Sidebar — instructions
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='font-family:"Playfair Display",serif; font-size:1.3rem; font-weight:700; color:#f5f0e8; margin-bottom:8px;'>Instructions</div>
    <div style='font-family:"DM Sans",sans-serif; font-size:0.82rem; color:#9a9490; line-height:1.7;'>
    You start with <b style='color:#e8e4dc'>$2,000</b> in play money.<br><br>
    Each round, pick one of the four decks — <b>A, B, C, or D</b>.<br><br>
    Every pick gives you a <b style='color:#5CB87A'>reward</b>. Some picks also carry a <b style='color:#E05C5C'>penalty</b>.<br><br>
    Your goal is to <b style='color:#e8e4dc'>maximize your bankroll</b> over <b>100 trials</b>.<br><br>
    The decks differ in how often and how severely they penalize you — you'll have to figure out the pattern.
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # Deck selection stats (only shown after game over or with data)
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        counts = df["deck"].value_counts().reindex(DECK_LABELS, fill_value=0)
        st.markdown("<div style='font-family:\"DM Mono\",monospace; font-size:0.65rem; letter-spacing:0.2em; color:#5a5860; text-transform:uppercase; margin-bottom:8px;'>Deck selections</div>", unsafe_allow_html=True)
        for i, label in enumerate(DECK_LABELS):
            pct = int(counts[label] / len(df) * 100) if len(df) > 0 else 0
            st.markdown(f"""
            <div style='display:flex; align-items:center; gap:8px; margin-bottom:6px;'>
                <div style='width:22px; height:22px; background:{DECK_COLORS[i]}; border-radius:4px; display:flex; align-items:center; justify-content:center; font-family:"Playfair Display",serif; font-weight:700; font-size:0.8rem; color:white;'>{label}</div>
                <div style='flex:1; background:#1a1a22; border-radius:999px; height:5px; overflow:hidden;'>
                    <div style='width:{pct}%; background:{DECK_COLORS[i]}; height:100%; border-radius:999px;'></div>
                </div>
                <div style='font-family:"DM Mono",monospace; font-size:0.65rem; color:#5a5860; width:28px; text-align:right;'>{counts[label]}</div>
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Main layout
# ─────────────────────────────────────────────
st.markdown('<div class="igt-title">Iowa Gambling Task</div>', unsafe_allow_html=True)
st.markdown('<div class="igt-subtitle">Decision-making under uncertainty</div>', unsafe_allow_html=True)

# ── Bankroll ──
trial    = st.session_state.trial
bankroll = st.session_state.bankroll
br_color = "#5CB87A" if bankroll >= STARTING_BANKROLL else "#E05C5C"
st.markdown(f"""
<div class="bankroll-box">
    <span class="bankroll-label">Balance</span>
    <span class="bankroll-amount" style="color:{br_color};">${bankroll:,.0f}</span>
    <span class="bankroll-trial">Trial {trial} / {MAX_TRIALS}</span>
</div>
""", unsafe_allow_html=True)

# ── Progress bar ──
pct = int(trial / MAX_TRIALS * 100)
st.markdown(f"""
<div class="progress-wrap">
    <div class="progress-fill" style="width:{pct}%;"></div>
</div>
""", unsafe_allow_html=True)

# ── Last result flash ──
if st.session_state.last_result:
    r          = st.session_state.last_result
    gain       = r["gain"]
    loss       = r["loss"]
    net        = r["reward"]
    deck_color = DECK_COLORS[r["arm"]]
    net_color  = "#5CB87A" if net >= 0 else "#E05C5C"

    gain_str = f"+${gain:,.0f}"
    loss_str = f"-${abs(loss):,.0f}" if loss != 0 else "—"
    loss_color = "#E05C5C" if loss != 0 else "#5a5860"
    net_str  = f"+${net:,.0f}" if net >= 0 else f"-${abs(net):,.0f}"

    st.markdown(f"""
    <div class="result-card" style="align-items:center; gap:18px;">
        <div class="result-deck-badge" style="background:{deck_color}; flex-shrink:0;">{r['deck']}</div>
        <div style="flex:1;">
            <div style="font-family:'DM Mono',monospace; font-size:0.6rem; letter-spacing:0.18em; color:#5a5860; text-transform:uppercase; margin-bottom:8px;">
                Trial {r['trial']}
            </div>
            <div style="display:flex; gap:24px; align-items:baseline;">
                <div>
                    <div style="font-family:'DM Mono',monospace; font-size:0.58rem; color:#5a5860; letter-spacing:0.12em; text-transform:uppercase; margin-bottom:2px;">Gain</div>
                    <div style="font-family:'DM Mono',monospace; font-size:1.05rem; color:#5CB87A;">{gain_str}</div>
                </div>
                <div style="color:#2e2e3a; font-size:1.2rem;">|</div>
                <div>
                    <div style="font-family:'DM Mono',monospace; font-size:0.58rem; color:#5a5860; letter-spacing:0.12em; text-transform:uppercase; margin-bottom:2px;">Loss</div>
                    <div style="font-family:'DM Mono',monospace; font-size:1.05rem; color:{loss_color};">{loss_str}</div>
                </div>
                <div style="color:#2e2e3a; font-size:1.2rem;">|</div>
                <div>
                    <div style="font-family:'DM Mono',monospace; font-size:0.58rem; color:#5a5860; letter-spacing:0.12em; text-transform:uppercase; margin-bottom:2px;">Net</div>
                    <div style="font-family:'DM Mono',monospace; font-size:1.15rem; font-weight:600; color:{net_color};">{net_str}</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Game over banner ──
if st.session_state.game_over:
    final = st.session_state.bankroll
    delta = final - STARTING_BANKROLL
    sign  = "+" if delta >= 0 else ""
    outcome_word = "Well played." if delta >= 0 else "Better luck next time."
    st.markdown(f"""
    <div class="gameover-banner">
        <div class="gameover-title">Game Over</div>
        <div class="gameover-score">Final balance: ${final:,.0f} &nbsp;·&nbsp; {sign}${delta:,.0f} vs start</div>
        <div style='font-family:"DM Sans",sans-serif; font-size:0.82rem; color:#9a9490; margin-top:8px;'>{outcome_word}</div>
    </div>
    """, unsafe_allow_html=True)

# ── Deck buttons ──
if not st.session_state.game_over:
    cols = st.columns(4, gap="small")
    for i, (label, color) in enumerate(zip(DECK_LABELS, DECK_COLORS)):
        with cols[i]:
            # Use st.button but style it via injected HTML
            st.markdown(f"""
            <style>
            div[data-testid="column"]:nth-child({i+1}) div[data-testid="stButton"] button {{
                background: {color}18;
                border: 2px solid {color}55;
                color: {color};
                font-family: 'Playfair Display', serif;
                font-size: 2rem;
                font-weight: 900;
                width: 100%;
                aspect-ratio: 2/3;
                border-radius: 14px;
                padding: 0;
                letter-spacing: 0;
                transition: all 0.12s ease;
            }}
            div[data-testid="column"]:nth-child({i+1}) div[data-testid="stButton"] button:hover {{
                background: {color}35;
                border-color: {color};
                transform: translateY(-3px);
                box-shadow: 0 10px 28px {color}40;
            }}
            </style>
            """, unsafe_allow_html=True)
            if st.button(label, key=f"deck_{i}", use_container_width=True):
                pick_deck(i)
                st.rerun()

# ── Equity curve chart ──
if len(st.session_state.history) > 1:
    df = pd.DataFrame(st.session_state.history)
    
    fig, ax = plt.subplots(figsize=(7, 2.8))
    fig.patch.set_facecolor("#0f0f13")
    ax.set_facecolor("#0f0f13")

    # shade above/below start
    ax.axhline(STARTING_BANKROLL, color="#2e2e3a", linewidth=0.8, linestyle="--")
    ax.fill_between(df["trial"], STARTING_BANKROLL, df["bankroll"],
                    where=(df["bankroll"] >= STARTING_BANKROLL),
                    color="#5CB87A", alpha=0.12)
    ax.fill_between(df["trial"], STARTING_BANKROLL, df["bankroll"],
                    where=(df["bankroll"] < STARTING_BANKROLL),
                    color="#E05C5C", alpha=0.12)

    # color segments by gain/loss
    for j in range(len(df) - 1):
        seg_color = "#5CB87A" if df["bankroll"].iloc[j+1] >= STARTING_BANKROLL else "#E05C5C"
        ax.plot(df["trial"].iloc[j:j+2], df["bankroll"].iloc[j:j+2],
                color=seg_color, linewidth=1.8, solid_capstyle="round")

    # scatter colored by deck
    for arm_idx, color in enumerate(DECK_COLORS):
        sub = df[df["arm"] == arm_idx]
        ax.scatter(sub["trial"], sub["bankroll"], color=color,
                   s=22, zorder=5, alpha=0.85, linewidths=0)

    ax.set_xlim(1, max(MAX_TRIALS, df["trial"].max()))
    ax.tick_params(colors="#5a5860", labelsize=7)
    ax.spines[["top","right","left","bottom"]].set_color("#2e2e3a")
    ax.yaxis.set_tick_params(color="#2e2e3a")
    ax.xaxis.set_tick_params(color="#2e2e3a")
    ax.set_xlabel("Trial", color="#5a5860", fontsize=7, labelpad=4)
    ax.set_ylabel("Balance ($)", color="#5a5860", fontsize=7, labelpad=4)
    ax.tick_params(axis='both', colors='#5a5860')

    # legend for decks
    patches = [mpatches.Patch(color=DECK_COLORS[i], label=f"Deck {DECK_LABELS[i]}") for i in range(4)]
    ax.legend(handles=patches, loc="upper left", framealpha=0,
              labelcolor="#9a9490", fontsize=6.5, ncol=4,
              handlelength=1, handleheight=0.8)

    plt.tight_layout(pad=0.5)
    st.pyplot(fig, use_container_width=True)
    plt.close()

# ── Controls ──
st.markdown("<br>", unsafe_allow_html=True)
col_r, col_s = st.columns([1, 3])
with col_r:
    if st.button("↺  New Game", use_container_width=True):
        reset_game()
        st.rerun()

# ── Post-game deck reveal ──
if st.session_state.game_over and len(st.session_state.history) > 0:
    st.divider()
    df = pd.DataFrame(st.session_state.history)
    st.markdown("""
    <div style='font-family:"DM Mono",monospace; font-size:0.65rem; letter-spacing:0.2em; color:#5a5860; text-transform:uppercase; margin-bottom:12px;'>
    Deck reveal
    </div>
    """, unsafe_allow_html=True)
    
    reveal_rows = [
        ("A", "Bad", "High reward, frequent heavy penalties. Net negative over time.", DECK_COLORS[0]),
        ("B", "Bad", "High reward, frequent heavy penalties. Net negative over time.", DECK_COLORS[1]),
        ("C", "Good", "Low reward, rare heavy penalties. Net positive over time.", DECK_COLORS[2]),
        ("D", "Good", "Low reward, rare heavy penalties. Net positive over time.", DECK_COLORS[3]),
    ]
    for deck_label, verdict, desc, color in reveal_rows:
        sub   = df[df["deck"] == deck_label]
        count = len(sub)
        avg   = sub["reward"].mean() if count > 0 else 0
        v_color = "#5CB87A" if verdict == "Good" else "#E05C5C"
        st.markdown(f"""
        <div style='display:flex; align-items:center; gap:14px; padding:12px 16px; background:#1a1a22; border-radius:10px; margin-bottom:8px; border:1px solid #2e2e3a;'>
            <div style='width:36px; height:48px; background:{color}; border-radius:7px; display:flex; align-items:center; justify-content:center; font-family:"Playfair Display",serif; font-weight:900; font-size:1.2rem; color:white; flex-shrink:0;'>{deck_label}</div>
            <div style='flex:1;'>
                <div style='font-family:"DM Sans",sans-serif; font-size:0.8rem; color:#e8e4dc; margin-bottom:2px;'>
                    <span style='color:{v_color}; font-weight:500;'>{verdict} deck</span> — {desc}
                </div>
                <div style='font-family:"DM Mono",monospace; font-size:0.65rem; color:#5a5860;'>
                    Picked {count}× · avg {("+" if avg>=0 else "")}{avg:,.0f} / trial
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

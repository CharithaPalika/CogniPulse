
import streamlit as st
import random
import time
import matplotlib.pyplot as plt

# -------------------------
# CONFIG
# -------------------------
TOTAL_ROUNDS = 30
REVERSAL_ROUND = None

PROB_A = 0.7
PROB_B = 0.3

POS_REWARD = 10
NEG_REWARD = -2

DELAY = 0.2 #1.5  # slightly increased so reward is visible

# -------------------------
# SESSION STATE INIT
# -------------------------
if "round" not in st.session_state:
    st.session_state.round = 1
    st.session_state.score = 0
    st.session_state.history = []
    st.session_state.prob_A = PROB_A
    st.session_state.prob_B = PROB_B
    st.session_state.game_over = False

    st.session_state.pending_reward = None
    st.session_state.pending_choice = None
    st.session_state.reveal = False

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="EarnQuest", layout="centered")

# -------------------------
# HEADER
# -------------------------
st.markdown("<h1 style='text-align:center;'>🎮 Pick and Earn</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center;'>Choose A button or B button to earn money</h4>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center;'>You are given only 50 chances to maximise the earnings...!!</h4>", unsafe_allow_html=True)
# -------------------------
# GAME LOGIC
# -------------------------
def play(choice):
    if st.session_state.game_over or st.session_state.reveal:
        return

    # reversal
    if REVERSAL_ROUND and st.session_state.round == REVERSAL_ROUND:
        st.session_state.prob_A, st.session_state.prob_B = (
            st.session_state.prob_B,
            st.session_state.prob_A,
        )

    prob = st.session_state.prob_A if choice == "A" else st.session_state.prob_B
    reward = POS_REWARD if random.random() < prob else NEG_REWARD

    st.session_state.pending_reward = reward
    st.session_state.pending_choice = choice
    st.session_state.reveal = True

# -------------------------
# SCORE DISPLAY
# -------------------------
col1, col2 = st.columns(2)

with col1:
    st.markdown(f"### 🔢 Round: {st.session_state.round-1}/{TOTAL_ROUNDS}")

with col2:
    st.markdown(
        f"<h3 style='text-align:right;'>⭐ Total returns earned: {st.session_state.score}</h3>",
        unsafe_allow_html=True
    )

st.divider()

# -------------------------
# REWARD DISPLAY
# -------------------------
reward_placeholder = st.empty()

# -------------------------
# BUTTONS
# -------------------------
if not st.session_state.game_over:
    col1, col2 = st.columns(2)

    with col1:
        st.button(
            "🟦 Choose A",
            use_container_width=True,
            on_click=play,
            args=("A",),
            disabled=st.session_state.reveal
        )

    with col2:
        st.button(
            "🟥 Choose B",
            use_container_width=True,
            on_click=play,
            args=("B",),
            disabled=st.session_state.reveal
        )

# -------------------------
# REVEAL REWARD (VISIBLE STEP)
# -------------------------
if st.session_state.reveal:

    reward = st.session_state.pending_reward
    choice = st.session_state.pending_choice

    # update state FIRST
    st.session_state.score += reward
    st.session_state.history.append((st.session_state.round, choice, reward))
    st.session_state.round += 1

    if st.session_state.round > TOTAL_ROUNDS:
        st.session_state.game_over = True

    # 🔥 SHOW REWARD CLEARLY
    if reward > 0:
        reward_placeholder.markdown(
            f"<h1 style='text-align:center; color:green;'>+{reward}</h1>",
            unsafe_allow_html=True
        )
    else:
        reward_placeholder.markdown(
            f"<h1 style='text-align:center; color:red;'>{reward}</h1>",
            unsafe_allow_html=True
        )

    # 🔥 HOLD SCREEN (critical for perception)
    time.sleep(DELAY)

    # clear state
    st.session_state.pending_reward = None
    st.session_state.pending_choice = None
    st.session_state.reveal = False

    # now move to next step
    st.rerun()

# -------------------------
# GAME OVER
# -------------------------
if st.session_state.game_over:
    st.markdown("<h2 style='text-align:center;'>🎉 Game Over!</h2>", unsafe_allow_html=True)
    st.markdown(
        f"<h3 style='text-align:center;'>Final Score: {st.session_state.score}</h3>",
        unsafe_allow_html=True
    )

# -------------------------
# SCATTER PLOT AFTER GAME
# -------------------------
if st.session_state.game_over and len(st.session_state.history) > 0:

    rounds = [r for r, c, rew in st.session_state.history]
    choices = [1 if c == "A" else 0 for _, c, _ in st.session_state.history]

    fig, ax = plt.subplots(figsize = (5,1.5))

    ax.scatter(rounds, choices, marker='|', s=200)

    ax.set_yticks([0, 1])
    ax.set_yticklabels(["B", "A"])
    ax.set_xlabel("Round")
    ax.set_ylabel("Choice")
    ax.set_title("Choice Pattern Over Trials")

    st.pyplot(fig)

# -------------------------
# HISTORY
# -------------------------
with st.expander("📊 Show History"):
    for r, c, rew in st.session_state.history:
        st.write(f"Round {r}: Choice {c} → {rew}")

# -------------------------
# RESET
# -------------------------
if st.button("🔄 Restart Game"):
    st.session_state.clear()
    st.rerun()


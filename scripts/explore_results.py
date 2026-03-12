"""
Streamlit app for exploring VLM tangram reference game results.

Run from the project root:
    streamlit run scripts/explore_results.py
"""

import ast
import os
import sys

import pandas as pd
import streamlit as st

# Allow imports from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.lm import SYSTEM_PROMPT
from src.utils import preprocess_messages

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
LOGPROBS_DIR = os.path.join(DATA_DIR, "logprobs")
GRID_IMAGE_PATH = os.path.join(DATA_DIR, "compiled_grid.png")
CHOICES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]

st.set_page_config(page_title="VLM Results Explorer", layout="wide")
st.title("VLM Tangram Reference Game — Results Explorer")

# ── Sidebar: file selection and filters ──────────────────────────────────────

st.sidebar.header("Data Selection")

subdirs = sorted(
    d for d in os.listdir(LOGPROBS_DIR) if os.path.isdir(os.path.join(LOGPROBS_DIR, d))
)
selected_subdir = st.sidebar.selectbox("Result category", subdirs)

subdir_path = os.path.join(LOGPROBS_DIR, selected_subdir)
csv_files = sorted(
    f
    for f in os.listdir(subdir_path)
    if f.endswith("_logprobs.csv") and "checkpoint" not in f
)
if not csv_files:
    st.warning("No result CSV files found in this directory.")
    st.stop()

selected_file = st.sidebar.selectbox("Result file", csv_files)


@st.cache_data
def load_data(path):
    return pd.read_csv(path)


df = load_data(os.path.join(subdir_path, selected_file))

# Derive model_prediction if missing
if "model_prediction" not in df.columns and "model_logprobs" in df.columns:

    def _predict(lp_str):
        try:
            lp = ast.literal_eval(lp_str)
            return max(lp, key=lp.get)
        except Exception:
            return None

    df["model_prediction"] = df["model_logprobs"].apply(_predict)

# Derive trialNum from orig_trialNum if needed (human_yoked format)
if "trialNum" not in df.columns and "orig_trialNum" in df.columns:
    df["trialNum"] = df["orig_trialNum"]

# Compute correctness column
if "target" in df.columns and "model_prediction" in df.columns:
    df["_correct"] = df["model_prediction"] == df["target"]
else:
    df["_correct"] = None

# ── Filters ──────────────────────────────────────────────────────────────────

st.sidebar.header("Filters")

if "gameId" in df.columns:
    game_ids = ["(all)"] + sorted(df["gameId"].dropna().unique().tolist())
    selected_game = st.sidebar.selectbox("Game ID", game_ids)
else:
    selected_game = "(all)"

if "trialNum" in df.columns:
    trial_nums = ["(all)"] + sorted(int(t) for t in df["trialNum"].dropna().unique())
    selected_trial = st.sidebar.selectbox("Trial number", trial_nums)
else:
    selected_trial = "(all)"

if "condition" in df.columns:
    conditions = ["(all)"] + sorted(df["condition"].dropna().unique().tolist())
    selected_condition = st.sidebar.selectbox("Condition", conditions)
else:
    selected_condition = "(all)"

correctness_filter = st.sidebar.radio(
    "Correctness", ["All", "Correct only", "Incorrect only"]
)

# Apply filters
filtered = df.copy()
if selected_game != "(all)":
    filtered = filtered[filtered["gameId"] == selected_game]
if selected_trial != "(all)":
    filtered = filtered[filtered["trialNum"] == int(selected_trial)]
if selected_condition != "(all)":
    filtered = filtered[filtered["condition"] == selected_condition]
if correctness_filter == "Correct only":
    filtered = filtered[filtered["_correct"] == True]
elif correctness_filter == "Incorrect only":
    filtered = filtered[filtered["_correct"] == False]

filtered = filtered.reset_index(drop=True)

if len(filtered) == 0:
    st.info("No trials match the current filters.")
    st.stop()

# ── Navigation ───────────────────────────────────────────────────────────────

if "trial_idx" not in st.session_state:
    st.session_state.trial_idx = 0

# Clamp index to current filtered range
st.session_state.trial_idx = min(st.session_state.trial_idx, len(filtered) - 1)

col_prev, col_counter, col_next = st.columns([1, 2, 1])
with col_prev:
    if st.button("← Previous", disabled=st.session_state.trial_idx == 0):
        st.session_state.trial_idx -= 1
        st.rerun()
with col_next:
    if st.button("Next →", disabled=st.session_state.trial_idx >= len(filtered) - 1):
        st.session_state.trial_idx += 1
        st.rerun()
with col_counter:
    new_idx = st.number_input(
        "Trial",
        min_value=1,
        max_value=len(filtered),
        value=st.session_state.trial_idx + 1,
        step=1,
        label_visibility="collapsed",
    )
    if new_idx - 1 != st.session_state.trial_idx:
        st.session_state.trial_idx = new_idx - 1
        st.rerun()
    st.caption(f"Trial {st.session_state.trial_idx + 1} of {len(filtered)}")

row = filtered.iloc[st.session_state.trial_idx]

# ── Metadata ─────────────────────────────────────────────────────────────────

st.divider()
meta_cols = st.columns(6)
meta_fields = [
    ("Game ID", row.get("gameId", "—")),
    ("Trial #", row.get("trialNum", "—")),
    ("Condition", row.get("condition", "—")),
    ("Target", row.get("target", "—")),
    ("Prediction", row.get("model_prediction", "—")),
    (
        "Correct?",
        "Yes"
        if row.get("_correct")
        else "No"
        if row.get("_correct") is not None
        else "—",
    ),
]
for col, (label, value) in zip(meta_cols, meta_fields):
    col.metric(label, value)

if "original_pct_correct" in row.index and pd.notna(row["original_pct_correct"]):
    st.caption(
        f"Original human accuracy on this trial: {row['original_pct_correct']:.1%}"
    )

# ── Main content: image + messages side by side ──────────────────────────────

left_col, right_col = st.columns([1, 2])

# Tangram grid image
with left_col:
    st.subheader("Tangram Grid")
    if os.path.exists(GRID_IMAGE_PATH):
        st.image(GRID_IMAGE_PATH, use_container_width=True)
    else:
        st.warning("Grid image not found.")

# Formatted messages
with right_col:
    st.subheader("Messages Sent to Model")

    try:
        chat_messages = preprocess_messages(row)
    except Exception as e:
        st.error(f"Error preprocessing messages: {e}")
        chat_messages = []

    # System prompt
    with st.chat_message("user", avatar="⚙️"):
        st.markdown(f"**System prompt:**\n\n{SYSTEM_PROMPT.strip()}")

    print("chat_messages:", chat_messages)
    # Conversation turns
    for msg in chat_messages:
        role = msg["role"]
        content = msg["content"]
        # Replace newlines with markdown line breaks to preserve formatting
        display_content = content.replace("\n", "  \n")
        if role == "assistant":
            with st.chat_message("assistant"):
                st.markdown(display_content)
        else:
            # User messages (describer/matcher dialogue + feedback)
            with st.chat_message("user"):
                st.markdown(display_content)

# ── Logprobs bar chart ───────────────────────────────────────────────────────

st.subheader("Model Log-Probabilities")

if "model_logprobs" in row.index and pd.notna(row["model_logprobs"]):
    try:
        logprobs = ast.literal_eval(str(row["model_logprobs"]))
        chart_data = pd.DataFrame(
            {
                "Choice": CHOICES,
                "Log-Prob": [logprobs.get(c, float("nan")) for c in CHOICES],
            }
        )
        chart_data = chart_data.dropna(subset=["Log-Prob"])

        # Color bars by role: target, prediction, or neither
        target = row.get("target", "")
        pred = row.get("model_prediction", "")

        def bar_color(choice):
            if choice == target and choice == pred:
                return "Correct prediction"
            elif choice == target:
                return "Target"
            elif choice == pred:
                return "Prediction"
            return "Other"

        chart_data["Type"] = chart_data["Choice"].apply(bar_color)

        import altair as alt

        color_scale = alt.Scale(
            domain=["Correct prediction", "Target", "Prediction", "Other"],
            range=["#2ca02c", "#1f77b4", "#ff7f0e", "#cccccc"],
        )

        chart = (
            alt.Chart(chart_data)
            .mark_bar()
            .encode(
                x=alt.X("Choice:N", sort=CHOICES, title="Choice"),
                y=alt.Y("Log-Prob:Q", title="Log-Probability"),
                color=alt.Color("Type:N", scale=color_scale, title=""),
                tooltip=["Choice", "Log-Prob", "Type"],
            )
            .properties(height=300)
        )
        st.altair_chart(chart, use_container_width=True)
    except Exception as e:
        st.error(f"Error parsing logprobs: {e}")
else:
    st.info("No logprobs available for this trial.")

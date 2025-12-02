# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Indy12 Joint Data Dashboard", layout="wide")

st.title("Indy12 Synthetic Joint Data Dashboard")

# ========================================
#  1) 데이터 로드 (3개: ideal / real / cleaned)
# ========================================
@st.cache_data
def load_data():
    df_ideal = pd.read_csv("ideal_sim.csv")
    df_real = pd.read_csv("real_like.csv")
    df_clean = pd.read_csv("real_like_cleaned.csv")
    return df_ideal, df_real, df_clean

df_ideal, df_real, df_clean = load_data()

joint_ids = [1, 2, 3, 4, 5, 6]

# ========================================
#  2) 사이드바 설정
# ========================================
st.sidebar.header("설정")

view_mode = st.sidebar.radio(
    "보기 모드 선택",
    ["단일 데이터 보기", "비교 보기 (ideal vs real vs cleaned)"]
)

joint_id = st.sidebar.selectbox(
    "Joint 선택",
    options=joint_ids,
    index=0,
)

var_name = st.sidebar.selectbox(
    "변수 선택",
    options=["position", "velocity", "torque"],
    index=0,
)

# 시간 범위
t_min = float(df_ideal["timestamp"].min())
t_max = float(df_ideal["timestamp"].max())

time_range = st.sidebar.slider(
    "시간 범위 [s]",
    min_value=t_min,
    max_value=t_max,
    value=(t_min, t_max),
    step=0.1,
)

downsample = st.sidebar.selectbox(
    "다운샘플링 (표본 간격)",
    options=[1, 2, 5, 10],
    index=2,
)

# ========================================
#  3) 공통: 시간 필터 적용
# ========================================
def filter_time(df, t_start, t_end, step):
    mask = (df["timestamp"] >= t_start) & (df["timestamp"] <= t_end)
    return df.loc[mask].iloc[::step, :].copy()

df_ideal_sel = filter_time(df_ideal, time_range[0], time_range[1], downsample)
df_real_sel  = filter_time(df_real,  time_range[0], time_range[1], downsample)
df_clean_sel = filter_time(df_clean, time_range[0], time_range[1], downsample)

# 변수→컬럼 매핑
col_map = {
    "position": "joint_position_{}",
    "velocity": "joint_velocity_{}",
    "torque":   "joint_torque_{}",
}
col_name = col_map[var_name].format(joint_id)

y_label_map = {
    "position": "Position [rad]",
    "velocity": "Velocity [rad/s]",
    "torque":   "Torque [Nm]",
}
y_label = y_label_map[var_name]

# ========================================
#  4) 단일 데이터 보기 모드
# ========================================
if view_mode == "단일 데이터 보기":
    dataset_name = st.sidebar.selectbox(
        "데이터셋 선택",
        ("ideal_sim", "real_like", "real_like_cleaned"),
        index=0,
    )

    if dataset_name == "ideal_sim":
        df_sel = df_ideal_sel
    elif dataset_name == "real_like":
        df_sel = df_real_sel
    else:
        df_sel = df_clean_sel

    time = df_sel["timestamp"].values
    y = df_sel[col_name].values
    state = df_sel["robot_state"].values

    st.subheader(
        f"[단일 보기] Dataset: {dataset_name}  |  Joint {joint_id}  |  {var_name}"
    )

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(time, y)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(y_label)
    ax.grid(True)
    st.pyplot(fig)

    st.markdown("#### Robot State (0: stop, 1: run, 2: idle/pause)")

    fig2, ax2 = plt.subplots(figsize=(12, 2.5))
    ax2.step(time, state, where="post")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("state")
    ax2.set_yticks([0, 1, 2])
    ax2.grid(True)
    st.pyplot(fig2)

    st.markdown("#### Raw data (부분)")
    st.dataframe(df_sel[[ "timestamp", "robot_state", col_name ]].head(50))

# ========================================
#  5) 비교 보기 모드 (ideal vs real vs cleaned)
# ========================================
else:
    st.subheader(
        f"[비교 보기] Joint {joint_id}  |  {var_name}  |  t = {time_range[0]:.1f} ~ {time_range[1]:.1f} s"
    )

    t_i = df_ideal_sel["timestamp"].values
    y_i = df_ideal_sel[col_name].values

    t_r = df_real_sel["timestamp"].values
    y_r = df_real_sel[col_name].values

    t_c = df_clean_sel["timestamp"].values
    y_c = df_clean_sel[col_name].values

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t_i, y_i, label="ideal_sim", linewidth=2)
    ax.plot(t_r, y_r, "--", label="real_like (raw)")
    ax.plot(t_c, y_c, label="real_like (cleaned)", linewidth=1.5)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(y_label)
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    st.markdown("#### 부분 데이터 (ideal / real / cleaned)")

    # 세 데이터셋을 timestamp 기준으로 병합해서 일부만 보기
    merged = df_ideal_sel[["timestamp", col_name]].rename(columns={col_name: "ideal"})
    merged = merged.merge(
        df_real_sel[["timestamp", col_name]].rename(columns={col_name: "real_raw"}),
        on="timestamp",
        how="outer"
    )
    merged = merged.merge(
        df_clean_sel[["timestamp", col_name]].rename(columns={col_name: "real_clean"}),
        on="timestamp",
        how="outer"
    )
    merged = merged.sort_values("timestamp").reset_index(drop=True)

    st.dataframe(merged.head(50))

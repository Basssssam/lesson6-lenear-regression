import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# -----------------------------
# 기본 설정
# -----------------------------
st.set_page_config(
    page_title="서울 기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 서울 기온 예측기")
st.write(
    "서울의 과거 연평균기온을 분석하고, "
    "선형 회귀로 선택한 연도의 예상 기온을 계산합니다."
)

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)

# -----------------------------
# 데이터 불러오기
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    df["날짜"] = pd.to_datetime(
        df["날짜"], errors="coerce"
    )
    df["평균기온"] = pd.to_numeric(
        df["평균기온"], errors="coerce"
    )

    df = df.dropna(subset=["날짜", "평균기온"]).copy()

    # 2025년까지의 자료만 사용
    df = df[df["날짜"].dt.year <= 2025].copy()
    df["연도"] = df["날짜"].dt.year

    # 연도별 평균기온과 관측일 수 계산
    yearly = (
        df.groupby("연도")
        .agg(
            평균기온=("평균기온", "mean"),
            관측일수=("평균기온", "count")
        )
        .reset_index()
    )

    # 관측일 300일 이상인 연도만 사용
    yearly = yearly[yearly["관측일수"] >= 300].copy()

    return yearly.sort_values("연도").reset_index(drop=True)


try:
    yearly = load_data()
except Exception as e:
    st.error(f"데이터를 불러올 수 없습니다: {e}")
    st.stop()

if len(yearly) < 2:
    st.error("회귀 분석에 필요한 데이터가 부족합니다.")
    st.stop()

# -----------------------------
# 전체 기간 선형 회귀
# -----------------------------
x = yearly["연도"].to_numpy(dtype=float)
y = yearly["평균기온"].to_numpy(dtype=float)

slope, intercept = np.polyfit(x, y, 1)
correlation = np.corrcoef(x, y)[0, 1]

start_year = int(yearly["연도"].min())
end_year = int(yearly["연도"].max())
year_count = len(yearly)

# 100년당 기온 변화량
slope_100 = slope * 100

# -----------------------------
# 최근 20년 선형 회귀
# -----------------------------
# 마지막 관측 연도부터 20년 범위
recent_start = end_year - 19
recent = yearly[
    (yearly["연도"] >= recent_start)
    & (yearly["연도"] <= end_year)
].copy()

if len(recent) >= 2:
    recent_x = recent["연도"].to_numpy(dtype=float)
    recent_y = recent["평균기온"].to_numpy(dtype=float)

    recent_slope, recent_intercept = np.polyfit(
        recent_x, recent_y, 1
    )
    recent_correlation = np.corrcoef(
        recent_x, recent_y
    )[0, 1]
    recent_slope_100 = recent_slope * 100
else:
    recent_slope = None
    recent_intercept = None
    recent_correlation = None
    recent_slope_100 = None

# -----------------------------
# 기울기 비교
# -----------------------------
st.subheader("📈 100년당 기온 변화 비교")

st.caption(
    f"전체 기간: {start_year}~{end_year}년 "
    f"({year_count}개 연도 사용)"
)

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 전체 기간")
    st.metric(
        label="100년당 기온 변화",
        value=f"{slope_100:+.2f}℃"
    )
    st.metric(
        label="상관계수",
        value=f"{correlation:.4f}"
    )
    st.caption(
        f"분석 기간: {start_year}~{end_year}년"
    )

with col2:
    st.markdown("### 최근 20년")
    if recent_slope_100 is not None:
        st.metric(
            label="100년당 기온 변화",
            value=f"{recent_slope_100:+.2f}℃"
        )
        st.metric(
            label="상관계수",
            value=f"{recent_correlation:.4f}"
        )
        st.caption(
            f"분석 기간: {int(recent['연도'].min())}"
            f"~{int(recent['연도'].max())}년 "
            f"({len(recent)}개 연도 사용)"
        )
    else:
        st.warning("최근 기간의 자료가 부족합니다.")

st.info(
    "양수(+)는 상승 추세, 음수(-)는 하락 추세입니다. "
    "100년당 변화량은 회귀 직선의 기울기에 100을 곱한 값입니다. "
    "최근 20년의 변화량은 짧은 기간의 추세를 환산한 것이므로 "
    "실제 100년간의 변화량을 뜻하지 않습니다."
)

# -----------------------------
# 산점도와 회귀 직선
# -----------------------------
st.subheader("🌡️ 연도별 평균기온과 회귀 직선")

fig = go.Figure()

# 실제 관측 데이터
fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["평균기온"],
        mode="markers",
        name="실제 연평균기온",
        marker=dict(size=7),
        customdata=yearly[["관측일수"]].to_numpy(),
        hovertemplate=(
            "연도: %{x}년<br>"
            "평균기온: %{y:.2f}℃<br>"
            "관측일수: %{customdata[0]}일"
            "<extra></extra>"
        )
    )
)

# 전체 기간 회귀 직선
line_years = np.arange(1900, 2101)
line_temps = slope * line_years + intercept

fig.add_trace(
    go.Scatter(
        x=line_years,
        y=line_temps,
        mode="lines",
        name="전체 기간 회귀 직선",
        line=dict(width=3),
        hovertemplate=(
            "연도: %{x}년<br>"
            "추정 기온: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)

# 최근 20년 회귀 직선
if recent_slope is not None:
    recent_line_years = np.arange(
        int(recent["연도"].min()),
        int(recent["연도"].max()) + 1
    )
    recent_line_temps = (
        recent_slope * recent_line_years
        + recent_intercept
    )

    fig.add_trace(
        go.Scatter(
            x=recent_line_years,
            y=recent_line_temps,
            mode="lines",
            name="최근 20년 회귀 직선",
            line=dict(width=3, dash="dash"),
            hovertemplate=(
                "연도: %{x}년<br>"
                "추정 기온: %{y:.2f}℃"
                "<extra></extra>"
            )
        )
    )

fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    xaxis=dict(range=[1900, 2100]),
    hovermode="closest",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0
    )
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# 연도별 예상 기온
# -----------------------------
st.subheader("🔮 선택한 연도의 예상 기온")

selected_year = st.slider(
    "연도를 선택하세요",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1
)

# 전체 기간 회귀 직선을 기준으로 예상
predicted_temp = slope * selected_year + intercept

st.markdown(f"### {selected_year}년 예상 연평균기온")

st.markdown(
    f"""
    <div style="text-align:center; padding:15px;">
        <span style="font-size:60px; font-weight:bold;">
            {predicted_temp:.2f}℃
        </span>
    </div>
    """,
    unsafe_allow_html=True
)

# 선택한 연도의 예상값 표시
fig_selected = go.Figure(fig)

fig_selected.add_trace(
    go.Scatter(
        x=[selected_year],
        y=[predicted_temp],
        mode="markers",
        name="선택한 연도의 예상 기온",
        marker=dict(size=15, symbol="diamond"),
        hovertemplate=(
            f"{selected_year}년<br>"
            "예상 기온: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)

st.plotly_chart(
    fig_selected,
    use_container_width=True
)

# -----------------------------
# 데이터 표
# -----------------------------
with st.expander("📋 분석에 사용한 연도별 데이터"):
    st.dataframe(
        yearly,
        use_container_width=True,
        hide_index=True
    )

st.caption(
    "이 앱의 예상 기온은 과거 자료에 맞춘 선형 회귀 결과입니다. "
    "실제 미래 기온을 확정적으로 예측하는 값은 아닙니다."
)

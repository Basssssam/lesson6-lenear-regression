import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# -----------------------------
# 기본 설정
# -----------------------------
st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 서울 기온 예측기")
st.write(
    "서울의 과거 연평균기온을 분석하고, "
    "선형 회귀를 이용해 선택한 연도의 예상 기온을 확인합니다."
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

    # 날짜와 기온 데이터 정리
    df["날짜"] = pd.to_datetime(
        df["날짜"],
        errors="coerce"
    )
    df["평균기온"] = pd.to_numeric(
        df["평균기온"],
        errors="coerce"
    )

    df = df.dropna(subset=["날짜", "평균기온"]).copy()

    # 연도 추출
    df["연도"] = df["날짜"].dt.year

    # 2025년까지의 데이터만 사용
    df = df[df["연도"] <= 2025].copy()

    # 연도별 평균기온과 관측일 수 계산
    yearly = (
        df.groupby("연도")
        .agg(
            평균기온=("평균기온", "mean"),
            관측일수=("평균기온", "count")
        )
        .reset_index()
    )

    # 관측일이 300일 이상인 연도만 사용
    yearly = yearly[yearly["관측일수"] >= 300].copy()

    yearly["평균기온"] = yearly["평균기온"].round(3)

    return yearly.sort_values("연도").reset_index(drop=True)


# -----------------------------
# 데이터 처리 및 오류 확인
# -----------------------------
try:
    yearly = load_data()
except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()

if len(yearly) < 2:
    st.error("회귀 분석에 사용할 수 있는 연도가 2개 미만입니다.")
    st.stop()

# -----------------------------
# 선형 회귀 계산
# -----------------------------
x = yearly["연도"].to_numpy(dtype=float)
y = yearly["평균기온"].to_numpy(dtype=float)

# y = 기울기 * x + 절편
slope, intercept = np.polyfit(x, y, 1)

# 상관계수
correlation = np.corrcoef(x, y)[0, 1]

start_year = int(yearly["연도"].min())
end_year = int(yearly["연도"].max())
year_count = len(yearly)

# -----------------------------
# 회귀 분석 결과 표시
# -----------------------------
st.subheader("📊 회귀 분석 정보")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("분석에 사용한 연도 수", f"{year_count}개")

with col2:
    st.metric("시작 연도", f"{start_year}년")

with col3:
    st.metric("끝 연도", f"{end_year}년")

with col4:
    st.metric("상관계수", f"{correlation:.4f}")

st.caption(
    f"회귀 직선: 평균기온 = {slope:.5f} × 연도 "
    f"+ ({intercept:.3f})"
)

# -----------------------------
# 연도별 산점도 + 회귀 직선
# -----------------------------
st.subheader("📈 연도별 평균기온과 회귀 직선")

fig = go.Figure()

# 실제 연평균기온 산점도
fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["평균기온"],
        mode="markers",
        name="실제 연평균기온",
        marker=dict(size=7),
        customdata=yearly[["관측일수"]],
        hovertemplate=(
            "연도: %{x}년<br>"
            "평균기온: %{y:.2f}℃<br>"
            "관측일수: %{customdata[0]}일"
            "<extra></extra>"
        )
    )
)

# 1900~2100년 회귀 직선
line_years = np.arange(1900, 2101)
line_temps = slope * line_years + intercept

fig.add_trace(
    go.Scatter(
        x=line_years,
        y=line_temps,
        mode="lines",
        name="선형 회귀 직선",
        line=dict(width=3),
        hovertemplate=(
            "연도: %{x}년<br>"
            "회귀 추정 기온: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)

fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
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
# 연도 선택 슬라이더
# -----------------------------
st.subheader("🌡️ 원하는 연도의 예상 기온 확인")

selected_year = st.slider(
    "예상 기온을 확인할 연도",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1
)

# 회귀 직선에 따른 예상 기온
predicted_temp = slope * selected_year + intercept

st.markdown(f"### {selected_year}년 예상 연평균기온")

st.markdown(
    f"<h1 style='text-align: center; font-size: 60px;'>"
    f"{predicted_temp:.2f}℃</h1>",
    unsafe_allow_html=True
)

# 선택한 연도를 그래프에 표시
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

fig_selected.update_layout(
    xaxis=dict(range=[1900, 2100])
)

st.plotly_chart(fig_selected, use_container_width=True)

# -----------------------------
# 분석에 사용한 데이터 확인
# -----------------------------
with st.expander("분석에 사용한 연도별 데이터 보기"):
    st.dataframe(
        yearly,
        use_container_width=True,
        hide_index=True
    )

st.caption(
    "주의: 예상 기온은 2025년까지의 관측 자료에 맞춘 "
    "선형 회귀 직선의 계산 결과입니다. "
    "실제 미래 기온이나 기후 변화를 정확하게 예측하는 값은 아닙니다."
)

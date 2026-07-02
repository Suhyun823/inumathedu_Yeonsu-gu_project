import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# 1. 페이지 설정 및 제목
st.set_page_config(page_title="심전도와 푸리에 급수 시각화", layout="wide")
st.title("🔬 연구실 살인사건: 심전도 역문제와 푸리에 근사 이론")
st.markdown("### 의생명·수학 융합 교육 프로그램 (1교시 시각화 검증)")
st.write("삼각함수의 항($n$)을 늘려가며 매끄러운 곡선들이 어떻게 뾰족한 QRS파를 대수적으로 합성하는지 확인해봅시다.")

# 사이드바: 인터랙티브 컨트롤러
st.sidebar.header("⚙️ 푸리에 급수 제어 매개변수")
n_terms = st.sidebar.slider("푸리에 급수의 항의 개수 (n)", min_value=1, max_value=50, value=5, step=1)
signal_type = st.sidebar.radio("신호 형태 선택", ["가상 심전도 파형(첨점 포함)", "이상적인 사각파(Gibbs 현상 극대화)"])

# 2. 수학적 배경 데이터 및 함수 정의
x = np.linspace(-np.pi, np.pi, 1000)

def generate_target_signal(x, type):
    """학생들이 근사할 목표 신호(가상 ECG 혹은 사각파)를 생성합니다."""
    if type == "가상 심전도 파형(첨점 포함)":
        # R파(QRS)의 날카로운 첨점을 묘사하기 위한 가상 주기 함수
        y = np.zeros_like(x)
        # 중심부에 뾰족한 첨점 배치
        y[abs(x) < 0.2] = 1.0 - abs(x[abs(x) < 0.2]) / 0.2
        # P파, T파 느낌의 완만한 언덕 추가
        y += 0.1 * np.exp(-((x - 1.0)/0.3)**2)  # T파 표현
        y += 0.05 * np.exp(-((x + 1.0)/0.2)**2) # P파 표현
        return y
    else:
        # 깁스 현상을 가장 극명하게 볼 수 있는 사각파
        return np.sign(np.sin(x))

def fourier_approximation(x, n, type):
    """n개의 항을 활용한 푸리에 급수 전개 식을 계산합니다."""
    f_approx = np.zeros_like(x)
    
    if type == "가상 심전도 파형(첨점 포함)":
        # 수치적으로 목표 신호의 푸리에 계수를 근사 계산 (학생 이해용 단순화 모델)
        target = generate_target_signal(x, type)
        # a0 구하기
        a0 = np.mean(target)
        f_approx += a0
        for i in range(1, n + 1):
            # 계수 an, bn 유도 (첨점이 존재하므로 1/n^2 수준으로 느리게 감소함)
            an = np.mean(target * np.cos(i * x)) * 2
            bn = np.mean(target * np.sin(i * x)) * 2
            f_approx += an * np.cos(i * x) + bn * np.sin(i * x)
    else:
        # 사각파 공식: (4/pi) * sum( sin(nx)/n ) (홀수 n만 존재)
        for i in range(1, n + 1):
            if i % 2 == 1:
                f_approx += (4 / np.pi) * (np.sin(i * x) / i)
    return f_approx

# 데이터 생성
y_target = generate_target_signal(x, signal_type)
y_approx = fourier_approximation(x, n_terms, signal_type)
error = y_target - y_approx

# 3. Streamlit 화면 레이아웃 구성 (2단 컬럼 구조)
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 그래프 시각화 분석")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, y_target, label="실제 생체 데이터 (Original)", color="black", linestyle="--", alpha=0.7)
    ax.plot(x, y_approx, label=f"푸리에 근사 합산 곡선 (n={n_terms})", color="red", linewidth=2)
    ax.set_title(f"Fourier Series Approximation (n={n_terms})", fontsize=14)
    ax.grid(True, linestyle=":")
    ax.legend(loc="upper right")
    st.pyplot(fig)

with col2:
    st.subheader("💡 수리해석학적 관찰 노트")
    
    # 깁스 현상 및 한계 설명 동적 피드백
    if n_terms < 5:
        st.warning("⚠️ 현재 **n이 너무 낮아** 뾰족한 순간변화율(QRS파)을 완벽히 표현하지 못하고 부드러운 곡선 형태로 뭉개집니다.")
    elif n_terms >= 30:
        st.success("✨ **고주파 성분($n \ge 30$)이 누적**되면서 급격히 꺾이는 첨점이 원본과 유사하게 합성되었습니다!")
    else:
        st.info("🔄 중간 빈도의 파동들이 상쇄·보강 간섭을 일으키며 점차 수렴해가는 과정입니다.")
        
    st.markdown("---")
    # 수치적 지표 제시 (최대 오차값 계산을 통한 오버슈트/깁스현상 시각화 보조)
    max_error = np.max(np.abs(error))
    st.metric(label="🎯 현재 설정된 n에서의 최대 절대 오차(Max Error)", value=f"{max_error:.4f}")
    
    st.markdown("""
    **📝 학생 안내 미션**
    1. 슬라이더의 `n`을 **1에서 50까지** 천천히 늘려보세요.
    2. 뾰족하게 꺾이는 부분 주변에서 사라지지 않는 작은 진동인 **'깁스 현상(Gibbs Phenomenon)'**을 관찰해보세요.
    3. 수학에서 '완전히 같다'는 것보다, 실용적인 공학에서 **'필요한 만큼 가까워지는 것'**의 가치를 토론해봅시다.
    """)

# 4. 하단부 오차 그래프 추가 공유
st.markdown("---")
st.subheader("📉 원래 함수와의 오차(Error) 추이 분석")
fig_err, ax_err = plt.subplots(figsize=(12, 2.5))
ax_err.plot(x, error, color="purple", label="Error (Original - Approx)")
ax_err.fill_between(x, error, color="purple", alpha=0.1)
ax_err.grid(True, linestyle=":")
ax_err.legend()
st.pyplot(fig_err)

# 본 시각화 도구는 인천대학교 수학교육과 '연구실 살인사건: 심전도 역문제와 과학수사' 융합 교육 프로그램의 일환으로 제작되었습니다.
st.caption("본 시각화 도구는 인천대학교 수학교육과 '연구실 살인사건: 심전도 역문제와 과학수사' 융합 교육 프로그램의 일환으로 제작되었습니다.")
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# 1. 페이지 설정 및 제목
st.set_page_config(page_title="ECG & Fourier Series", layout="wide")
st.title("🔬 Lab Murder Case: ECG Inverse Problem & Fourier Approximation")
st.markdown("### Biomedical-Mathematics Convergence Program (Session 1)")
st.write("Adjust the number of terms ($n$) to see how smooth trigonometric functions algebraically synthesize the sharp QRS complex.")

# 사이드바: 인터랙티브 컨트롤러
st.sidebar.header("⚙️ Fourier Series Parameters")
n_terms = st.sidebar.slider("Number of terms (n)", min_value=1, max_value=50, value=5, step=1)
signal_type = st.sidebar.radio("Select Signal Type", ["Virtual ECG Waveform", "Ideal Square Wave (Max Gibbs)"])

# 2. 수학적 배경 데이터 및 함수 정의
x = np.linspace(-np.pi, np.pi, 1000)

def generate_target_signal(x, type):
    """Generates the target signal to approximate."""
    if type == "Virtual ECG Waveform":
        y = np.zeros_like(x)
        y[abs(x) < 0.2] = 1.0 - abs(x[abs(x) < 0.2]) / 0.2
        y += 0.1 * np.exp(-((x - 1.0)/0.3)**2)  # T wave
        y += 0.05 * np.exp(-((x + 1.0)/0.2)**2) # P wave
        return y
    else:
        return np.sign(np.sin(x))

def fourier_approximation(x, n, type):
    """Calculates the Fourier series approximation."""
    f_approx = np.zeros_like(x)
    
    if type == "Virtual ECG Waveform":
        target = generate_target_signal(x, type)
        a0 = np.mean(target)
        f_approx += a0
        for i in range(1, n + 1):
            an = np.mean(target * np.cos(i * x)) * 2
            bn = np.mean(target * np.sin(i * x)) * 2
            f_approx += an * np.cos(i * x) + bn * np.sin(i * x)
    else:
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
    st.subheader("📊 Graphical Visualization")
    fig, ax = plt.subplots(figsize=(10, 5))
    # 한글 깨짐을 방지하기 위해 범례(label)를 영어로 수정했습니다.
    ax.plot(x, y_target, label="Original Signal", color="black", linestyle="--", alpha=0.7)
    ax.plot(x, y_approx, label=f"Fourier Sum (n={n_terms})", color="red", linewidth=2)
    ax.set_title(f"Fourier Series Approximation (n={n_terms})", fontsize=14)
    ax.grid(True, linestyle=":")
    ax.legend(loc="upper right")
    st.pyplot(fig)

with col2:
    st.subheader("💡 Mathematical Observation Note")
    
    if n_terms < 5:
        st.warning("⚠️ **n is too low:** The sharp QRS complex cannot be represented well and looks blurred as a smooth curve.")
    elif n_terms >= 30:
        st.success("✨ **High-frequency components accumulated:** The sharp peak close to the original signal is successfully synthesized!")
    else:
        st.info("🔄 Intermediate frequencies are causing constructive/destructive interference to converge.")
        
    st.markdown("---")
    max_error = np.max(np.abs(error))
    st.metric(label="🎯 Max Absolute Error", value=f"{max_error:.4f}")
    
    st.markdown("""
    **📝 Student Mission**
    1. Slowly increase the slider `n` from **1 to 50**.
    2. Observe the **'Gibbs Phenomenon'** (the persistent oscillation near the sharp peak).
    3. Discuss the practical value of being **'close enough'** in engineering versus 'exactly equal' in pure math.
    """)

# 4. 하단부 오차 그래프 추가 공유
st.markdown("---")
st.subheader("📉 Error Trend (Original - Approximation)")
fig_err, ax_err = plt.subplots(figsize=(12, 2.5))
ax_err.plot(x, error, color="purple", label="Error")
ax_err.fill_between(x, error, color="purple", alpha=0.1)
ax_err.grid(True, linestyle=":")
ax_err.legend()
st.pyplot(fig_err)

st.caption("This tool was developed as part of the Incheon National University Mathematics Education convergence program.")

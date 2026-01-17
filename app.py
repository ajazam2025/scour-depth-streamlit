import streamlit as st
import numpy as np
import pickle

# =================================================
# Load trained models and scaler
# =================================================
rf_model = pickle.load(open("rf_model.pkl", "rb"))
gpr_model = pickle.load(open("gpr_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# =================================================
# Page configuration
# =================================================
st.set_page_config(
    page_title="Scour Depth Predictor",
    layout="centered"
)

# =================================================
# Custom CSS for beautification
# =================================================
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(to right, #e3f2fd, #ffffff);
        font-family: 'Segoe UI', sans-serif;
    }

    h1 {
        color: #0d47a1;
        text-align: center;
    }

    h2, h3 {
        color: #1565c0;
    }

    label {
        font-weight: bold;
        color: #263238;
    }

    div.stButton > button {
        background-color: #1976d2;
        color: white;
        border-radius: 12px;
        height: 3em;
        width: 100%;
        font-size: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =================================================
# Title and description
# =================================================
st.title("🌊 GUI Tool for Ice-Covered Scour Depth Prediction ")
st.markdown(
    """
    ### ML based prediction of scour depth under ice-covered scenarios  
    **Models used:** Random Forest (RF) & Gaussian Process Regression (GPR)
    """
)

# =================================================
# Model selection
# =================================================
model_choice = st.selectbox(
    "🔍 Select Prediction Model",
    ["Random Forest", "Gaussian Process Regression"]
)

# =================================================
# Input parameters (two-column layout)
# =================================================
st.header("🔹 Input Parameters")

col1, col2 = st.columns(2)

with col1:
    U = st.number_input("🌊 Flow Velocity, U (m/s)", min_value=0.01, value=1.5)
    H = st.number_input("📏 Flow Depth, H (m)", min_value=0.01, value=0.6)
    Fr = st.number_input("⚡ Froude Number, Fr", min_value=0.01, value=0.5)

with col2:
    D = st.number_input("🟦 Pier Diameter, D (m)", min_value=0.01, value=0.3)
    d50 = st.number_input("🪨 Median Grain Size, d50 (m)", min_value=0.00001, value=0.001)

# =================================================
# Derived parameters (card style)
# =================================================
H_D = H / D
D_d50 = D / d50

st.markdown(
    f"""
    <div style="
        background-color:#ffffff;
        padding:15px;
        border-radius:12px;
        box-shadow:0px 4px 10px rgba(0,0,0,0.1);
        margin-top:10px;
    ">
    <h4 style="color:#0d47a1;">🔸 Derived Parameters</h4>
    <p><b>H / D</b> = {H_D:.3f}</p>
    <p><b>D / d<sub>50</sub></b> = {D_d50:.1f}</p>
    </div>
    """,
    unsafe_allow_html=True
)

# =================================================
# Governing equations (optional expander)
# =================================================
with st.expander("📐 Governing Equations & Definitions"):
    st.latex(r"Fr = \frac{U}{\sqrt{gH}}")
    st.latex(r"\frac{H}{D} = \text{Flow depth ratio}")
    st.latex(r"\frac{D}{d_{50}} = \text{Relative pier size}")

# =================================================
# Prediction section
# =================================================
st.markdown("---")

if st.button("🚀 Predict Scour Depth"):

    X = np.array([[U, H, D, Fr, d50, H_D, D_d50]])
    X_scaled = scaler.transform(X)

    if model_choice == "Random Forest":
        ds = rf_model.predict(X_scaled)[0]

        st.markdown(
            f"""
            <div style="
                background:#e8f5e9;
                padding:20px;
                border-radius:15px;
                text-align:center;
                font-size:22px;
                color:#1b5e20;
                margin-top:15px;
            ">
            🌊 <b>Predicted Scour Depth (RF)</b><br>
            <span style="font-size:32px;">{ds:.4f} m</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:
        ds, sigma = gpr_model.predict(X_scaled, return_std=True)

        st.markdown(
            f"""
            <div style="
                background:#e3f2fd;
                padding:20px;
                border-radius:15px;
                text-align:center;
                font-size:22px;
                color:#0d47a1;
                margin-top:15px;
            ">
            🌊 <b>Predicted Scour Depth (GPR)</b><br>
            <span style="font-size:32px;">{ds[0]:.4f} m</span><br>
            <span style="font-size:18px;">Uncertainty (±1σ): {sigma[0]:.4f} m</span>
            </div>
            """,
            unsafe_allow_html=True
        )

# =================================================
# Footer
# =================================================
st.markdown("---")
st.caption(
    "🔬 Developed by Ajaz Ahmad Mir, Research Scholar, "
    "Dr B R Ambedkar National Institute of Technology Jalandhar | "
    "Streamlit-based research GUI for scour depth prediction using "
    "Random Forest and Gaussian Process Regression models")

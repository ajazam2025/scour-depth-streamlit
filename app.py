# -------------------------
# Custom CSS for styling
# -------------------------
st.markdown(
    """
    <style>
    /* Background */
    .stApp {
        background: linear-gradient(to right, #e3f2fd, #ffffff);
        font-family: 'Segoe UI', sans-serif;
    }

    /* Title */
    h1 {
        color: #0d47a1;
        text-align: center;
    }

    /* Section headers */
    h2, h3 {
        color: #1565c0;
    }

    /* Buttons */
    div.stButton > button {
        background-color: #1976d2;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 18px;
    }

    /* Success box */
    .stAlert.success {
        background-color: #e8f5e9;
        color: #1b5e20;
        border-radius: 10px;
    }

    /* Info box */
    .stAlert.info {
        background-color: #e3f2fd;
        color: #0d47a1;
        border-radius: 10px;
    }

    /* Input labels */
    label {
        font-weight: bold;
        color: #263238;
    }
    </style>
    """,
    unsafe_allow_html=True
)
import streamlit as st
import numpy as np
import pickle

# Load models and scaler
rf_model = pickle.load(open("rf_model.pkl", "rb"))
gpr_model = pickle.load(open("gpr_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.set_page_config(page_title="Scour Depth Predictor", layout="centered")

st.title("🌊 Ice Covered Scour Depth GUI")
st.markdown("Machine Learning based prediction using RF and GPR")

model_choice = st.selectbox(
    "Select Prediction Model",
    ["Random Forest", "Gaussian Process Regression"]
)

st.header("Input Parameters")

U = st.number_input("Flow Velocity U (m/s)", min_value=0.01, value=1.5)
H = st.number_input("Flow Depth H (m)", min_value=0.01, value=0.6)
D = st.number_input("Pier Diameter D (m)", min_value=0.01, value=0.3)
Fr = st.number_input("Froude Number Fr", min_value=0.01, value=0.5)
d50 = st.number_input("Median Grain Size d50 (m)", min_value=0.00001, value=0.001)

H_D = H / D
D_d50 = D / d50

st.write(f"H/D = {H_D:.3f}")
st.write(f"D/d50 = {D_d50:.3f}")

if st.button("Predict Scour Depth"):
    X = np.array([[U, H, D, Fr, d50, H_D, D_d50]])
    X_scaled = scaler.transform(X)

    if model_choice == "Random Forest":
        ds = rf_model.predict(X_scaled)[0]
        st.success(f"Predicted Scour Depth (RF) = {ds:.4f} m")
    else:
        ds, sigma = gpr_model.predict(X_scaled, return_std=True)
        st.success(f"Predicted Scour Depth (GPR) = {ds[0]:.4f} m")
        st.info(f"Uncertainty (±1σ) = {sigma[0]:.4f} m")

import streamlit as st
import numpy as np
import pandas as pd
from io import StringIO
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostRegressor

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(layout="centered")

# -------------------------------
# CSS (STRONG VISUAL CHANGE)
# -------------------------------
st.markdown("""
<style>

/* Whole background */
.stApp {
    background: linear-gradient(to bottom, #e3f2fd, #ffffff);
}

/* Title block */
.title-box {
    background:#0d47a1;
    padding:20px;
    border-radius:15px;
    text-align:center;
    color:white;
    margin-bottom:10px;
}

.subtitle {
    font-size:15px;
}

/* Section box */
.section {
    background:white;
    padding:15px;
    border-radius:12px;
    margin-top:10px;
    box-shadow:0px 3px 10px rgba(0,0,0,0.1);
}

/* Result box */
.result-box {
    background:#c8e6c9;
    padding:20px;
    border-radius:15px;
    text-align:center;
    font-size:26px;
    font-weight:bold;
    color:#1b5e20;
}

/* Button */
.stButton>button {
    width:100%;
    height:3em;
    font-size:16px;
    border-radius:10px;
    background:#1565c0;
    color:white;
}

/* Footer */
.footer {
    text-align:center;
    font-size:12px;
    color:#555;
    margin-top:15px;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------
# HEADER (NOW CLEARLY DIFFERENT)
# -------------------------------
st.markdown("""
<div class="title-box">
<h2>🌊 Hydraulic Roughness Predictor</h2>
<div class="subtitle">
AI Applications in Geomorphology
</div>
</div>
""", unsafe_allow_html=True)

# -------------------------------
# DATA
# -------------------------------
DATA_CSV = """Fr,Re,H_D,LD,Slope_S,u_star,Manning_n
0.5015,543478,1.67,2,0.0005,0.3395,0.00368
0.2094,217391,1.64,2.5,0.0005,0.3358,0.00888
0.5537,652173,1.75,2.5,0.0005,0.3466,0.00328
"""

df = pd.read_csv(StringIO(DATA_CSV))
X = df.drop(columns=["Manning_n"])
y = df["Manning_n"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = AdaBoostRegressor()
model.fit(X_scaled, y)

# -------------------------------
# INPUT SECTION
# -------------------------------
st.markdown('<div class="section">', unsafe_allow_html=True)
st.subheader("🔹 Input Parameters")

Fr = st.number_input("Froude Number (Fr)", value=0.5)
Re = st.number_input("Reynolds Number (Re)", value=500000.0)
HD = st.number_input("Relative Submergence (H/D)", value=1.6)
LD = st.number_input("Spacing Ratio (λ/D)", value=2.5)
Slope = st.number_input("Channel Slope (S)", value=0.0005)
u_star = st.number_input("Shear Velocity (u*)", value=0.3)

st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------
# DERIVED BLOCK (VISUALLY DIFFERENT)
# -------------------------------
ratio = u_star / Fr

st.markdown(f"""
<div class="section">
<h4>📊 Derived Indicator</h4>
<b>u* / Fr</b> = {ratio:.3f}
</div>
""", unsafe_allow_html=True)

# -------------------------------
# EQUATIONS (OPTIONAL)
# -------------------------------
with st.expander("📐 Governing Equations"):
    st.latex(r"Fr = \\frac{U}{\\sqrt{gH}}")
    st.latex(r"n = \\frac{R^{2/3} S^{1/2}}{U}")

# -------------------------------
# BUTTON
# -------------------------------
st.markdown("---")
predict = st.button("🚀 Predict Manning’s n")

# -------------------------------
# RESULT (BIG CENTERED)
# -------------------------------
if predict:
    X_input = np.array([[Fr, Re, HD, LD, Slope, u_star]])
    pred = model.predict(scaler.transform(X_input))[0]

    st.markdown(f"""
    <div class="result-box">
    Predicted Manning’s n<br><br>
    {pred:.6f}
    </div>
    """, unsafe_allow_html=True)

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("""
<div class="footer">
Developed by Ajaz Mir<br>
Research Scholar<br>
Dr B R Ambedkar National Institute of Technology Jalandhar
</div>
""", unsafe_allow_html=True)

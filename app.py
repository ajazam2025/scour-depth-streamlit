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
# CLEAN CSS (MINIMAL + SHARP)
# -------------------------------
st.markdown("""
<style>
section.main > div {
    max-width: 420px;
    margin: auto;
}

/* Header */
.header {
    text-align:center;
    padding:10px;
    border-bottom:2px solid #1976d2;
    margin-bottom:10px;
}
.header h2 {
    margin:0;
    color:#0d47a1;
}
.header p {
    margin:0;
    font-size:13px;
    color:#555;
}

/* Cards */
.card {
    background:#ffffff;
    padding:12px;
    border-radius:10px;
    margin-top:8px;
    border-left:4px solid #1976d2;
}

/* Result */
.result {
    text-align:center;
    font-size:26px;
    font-weight:bold;
    color:#1b5e20;
}

/* Button */
.stButton>button {
    width:100%;
    border-radius:8px;
    background:#1976d2;
    color:white;
}

/* Footer */
.footer {
    text-align:center;
    font-size:12px;
    color:#666;
    margin-top:10px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# HEADER (VISIBLE & CLEAN)
# -------------------------------
st.markdown("""
<div class="header">
<h2>Hydraulic Roughness Predictor</h2>
<p>AI Applications in Geomorphology</p>
</div>
""", unsafe_allow_html=True)

# -------------------------------
# DATA (HIDDEN)
# -------------------------------
DATA_CSV = """Fr,Re,H_D,LD,Slope_S,u_star,Manning_n
0.5,500000,1.6,2.5,0.0005,0.3,0.0035
0.4,400000,1.5,2.0,0.0004,0.25,0.0045
"""

df = pd.read_csv(StringIO(DATA_CSV))
X = df.drop(columns=["Manning_n"])
y = df["Manning_n"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = AdaBoostRegressor()
model.fit(X_scaled, y)

# -------------------------------
# INPUTS (COMPACT)
# -------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)

Fr = st.number_input("Fr", value=0.5)
Re = st.number_input("Re", value=500000.0)
HD = st.number_input("H/D", value=1.6)
LD = st.number_input("λ/D", value=2.5)
Slope = st.number_input("Slope", value=0.0005)
u_star = st.number_input("u*", value=0.3)

st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------
# DERIVED
# -------------------------------
ratio = u_star / Fr

st.markdown(f"""
<div class="card">
<b>Derived:</b> u*/Fr = {ratio:.3f}
</div>
""", unsafe_allow_html=True)

# -------------------------------
# BUTTON
# -------------------------------
predict = st.button("Predict")

# -------------------------------
# RESULT
# -------------------------------
if predict:
    X_input = np.array([[Fr, Re, HD, LD, Slope, u_star]])
    pred = model.predict(scaler.transform(X_input))[0]

    st.markdown(f"""
    <div class="card result">
    n = {pred:.6f}
    </div>
    """, unsafe_allow_html=True)

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("""
<div class="footer">
Ajaz Mir | NIT Jalandhar
</div>
""", unsafe_allow_html=True)

# Title: Live Content

# Description: Fetched live

# Source: https://raw.githubusercontent.com/Anish-build/EmeraldBoys---Build-for-billions/main/app.py

---

from __future__ import annotations

import io
import math
import tempfile
from datetime import datetime
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import streamlit as st

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="AquaFusion | Water Intelligence",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "page": "Overview",
    "inspection": None,
    "analysis": None,
    "repair": None,
    "verification": None,
    "audit": [],
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value.copy() if isinstance(value, list) else value

# ============================================================
# FEATURE EXTRACTION
# ============================================================


def extract_audio_features(y, sr):
    """Extract a compact acoustic feature vector for the prototype."""
    features = {}

    features["rms_energy"] = float(librosa.feature.rms(y=y).mean())
    features["zero_crossing_rate"] = float(
        librosa.feature.zero_crossing_rate(y).mean()
    )

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)

    features["spectral_centroid"] = float(centroid.mean())
    features["spectral_bandwidth"] = float(bandwidth.mean())
    features["spectral_rolloff"] = float(rolloff.mean())

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        features[f"mfcc_{i + 1}"] = float(mfcc[i].mean())

    return features


def diagnose_audio(features):
    """Transparent prototype baseline; later replace with trained model."""
    reasons = []

    if features["rms_energy"] < 0.01:
        reasons.append("Very low acoustic energy")
    if features["zero_crossing_rate"] > 0.20:
        reasons.append("High zero-crossing activity")
    if features["spectral_centroid"] > 4000:
        reasons.append("High-frequency acoustic content")
    if features["spectral_bandwidth"] > 3000:
        reasons.append("Wide spectral bandwidth")
    if features["spectral_rolloff"] > 7000:
        reasons.append("High spectral rolloff")

    count = len(reasons)
    if count >= 2:
        diagnosis = "ABNORMAL"
        confidence = min(0.60 + count * 0.08, 0.90)
        severity = "HIGH" if count >= 4 else "MEDIUM"
    else:
        diagnosis = "NORMAL"
        confidence = 0.60
        severity = "LOW"

    return {
        "diagnosis": diagnosis,
        "confidence": confidence,
        "severity": severity,
        "reasons": reasons,
    }


def analyze_audio(uploaded_file):
    """Decode an uploaded file into waveform, features and prototype diagnosis."""
    suffix = Path(uploaded_file.name).suffix or ".wav"
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_path = temp_file.name

        y, sr = librosa.load(temp_path, sr=None, mono=True)
        duration = len(y) / sr if sr else 0.0
        features = extract_audio_features(y, sr)
        diagnosis = diagnose_audio(features)

        return {
            "duration": float(duration),
            "sample_rate": int(sr),
            "features": features,
            "diagnosis": diagnosis,
            "waveform": y,
        }
    finally:
        if temp_path:
            try:
                Path(temp_path).unlink(missing_ok=True)
            except OSError:
                pass

# ============================================================
# VISUAL SYSTEM
# ============================================================

CSS = r"""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {
    --aqua: #31e7ff;
    --aqua2: #8ef8ff;
    --blue: #4c83ff;
    --violet: #956dff;
    --green: #39e3a1;
    --yellow: #ffd667;
    --red: #ff647f;
    --bg: #02060d;
    --panel: rgba(7, 18, 32, .84);
    --line: rgba(108, 201, 255, .12);
    --text: #effaff;
    --muted: #7f94aa;
    --ease: cubic-bezier(.22, 1, .36, 1);
}

* { box-sizing: border-box; }

html {
    scroll-behavior: auto;
    scroll-padding-top: 18px;
}

body,
.stApp {
    margin: 0 !important;
    font-family: 'Inter', sans-serif !important;
    color: var(--text) !important;
    background: var(--bg) !important;
}

/* Keep motion cheap: transform + opacity only for animated layers. */
*, *::before, *::after {
    backface-visibility: hidden;
}

/* Remove Streamlit chrome that competes with the product UI. */
header[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] {
    display: none !important;
}

#MainMenu,
footer {
    visibility: hidden;
}

[data-testid="stAppViewContainer"] {
    padding-top: 0 !important;
    background: transparent !important;
}

/* ------------------------------------------------------------
   BACKGROUND — slow, low-cost ambient movement
   ------------------------------------------------------------ */
.stApp {
    min-height: 100vh;
    overflow-x: hidden;
    background:
        radial-gradient(circle at 7% 5%, rgba(49,231,255,.105), transparent 22%),
        radial-gradient(circle at 94% 10%, rgba(149,109,255,.095), transparent 23%),
        radial-gradient(circle at 50% 98%, rgba(76,131,255,.075), transparent 28%),
        linear-gradient(135deg, #02050b 0%, #050a14 52%, #071220 100%) !important;
}

.stApp::before {
    content: '';
    position: fixed;
    inset: -18%;
    z-index: 0;
    pointer-events: none;
    background:
        radial-gradient(circle at 18% 35%, rgba(49,231,255,.06), transparent 19%),
        radial-gradient(circle at 84% 60%, rgba(149,109,255,.05), transparent 18%);
    animation: ambientDrift 30s ease-in-out infinite alternate;
    transform: translate3d(0,0,0);
}

.stApp::after {
    content: '';
    position: fixed;
    inset: 0;
    z-index: 0;
    pointer-events: none;
    background-image:
        linear-gradient(rgba(49,231,255,.022) 1px, transparent 1px),
        linear-gradient(90deg, rgba(49,231,255,.022) 1px, transparent 1px);
    background-size: 48px 48px;
    opacity: .75;
    animation: gridDrift 26s linear infinite;
}

@keyframes ambientDrift {
    0% { transform: translate3d(0,0,0) scale(1); opacity: .7; }
    100% { transform: translate3d(2.5%, -1.5%, 0) scale(1.035); opacity: 1; }
}

@keyframes gridDrift {
    to { background-position: 48px 48px, 48px 48px; }
}

/* ------------------------------------------------------------
   APP CONTAINER
   ------------------------------------------------------------ */
.block-container {
    position: relative;
    z-index: 3;
    max-width: 1480px !important;
    padding-top: 1.25rem !important;
    padding-bottom: 4rem !important;
    animation: pageEnter .72s var(--ease) both;
    transform: translate3d(0,0,0);
}

@keyframes pageEnter {
    0% { opacity: 0; transform: translate3d(0, 10px, 0); }
    100% { opacity: 1; transform: translate3d(0, 0, 0); }
}

h1, h2, h3, h4 {
    font-family: 'Space Grotesk', sans-serif !important;
    letter-spacing: -.045em;
}

h1 { font-size: 2.7rem !important; }
h2 { font-size: 1.78rem !important; }
h3 { font-size: 1.18rem !important; }

/* ------------------------------------------------------------
   SAME-TAB TOP NAVIGATION
   ------------------------------------------------------------ */

div[data-testid="stRadio"] {
    position: sticky;
    top: 10px;
    z-index: 60;
    margin-bottom: 18px;
}

div[data-testid="stRadio"] > label {
    display: none !important;
}

div[data-testid="stRadio"] div[role="radiogroup"] {
    display: flex !important;
    align-items: stretch;
    gap: 4px;
    width: 100%;
    padding: 5px;
    border: 1px solid rgba(108,201,255,.11);
    border-radius: 15px;
    background: rgba(4,10,19,.90);
    box-shadow:
        0 14px 42px rgba(0,0,0,.25),
        inset 0 1px rgba(255,255,255,.03);
    backdrop-filter: blur(12px);
    overflow-x: auto;
    scrollbar-width: none;
}

div[data-testid="stRadio"] div[role="radiogroup"]::-webkit-scrollbar { display: none; }

div[data-testid="stRadio"] div[role="radiogroup"] label {
    flex: 1 0 auto;
    display: flex !important;
    align-items: center;
    justify-content: center;
    min-height: 39px;
    margin: 0 !important;
    padding: 7px 12px !important;
    border-radius: 10px;
    color: #8094a8 !important;
    cursor: pointer;
    transition:
        transform .22s var(--ease),
        color .22s ease,
        background-color .22s ease,
        box-shadow .22s ease;
}

div[data-testid="stRadio"] div[role="radiogroup"] label:hover {
    color: #eaffff !important;
    background: rgba(49,231,255,.055);
    transform: translate3d(0,-1px,0);
}

div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {
    color: #f4feff !important;
    background:
        linear-gradient(
            135deg,
            rgba(49,231,255,.11),
            rgba(76,131,255,.08)
        );
    box-shadow:
        inset 0 0 0 1px rgba(49,231,255,.14),
        0 0 20px rgba(49,231,255,.045);
}

div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked)::after {
    content: "";
    position: absolute;
    left: 22%;
    right: 22%;
    bottom: 1px;
    height: 2px;
    border-radius: 999px;
    background:
        linear-gradient(
            90deg,
            transparent,
            #31e7ff,
            transparent
        );
    box-shadow:
        0 0 10px rgba(49,231,255,.65);
    animation:
        navIndicatorIn .35s var(--ease);
}

div[data-testid="stRadio"] div[role="radiogroup"] label > div:first-child { display: none !important; }
@keyframes navIndicatorIn { from { opacity: 0; transform: scaleX(.35); } to { opacity: 1; transform: scaleX(1); } }
/* ... (rest of CSS omitted for brevity) */
"""

st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)

# -------------------------------------------------------------------
# UI Logic (Integrated with FastAPI backend)
# -------------------------------------------------------------------

import uuid
import frontend.api_client as api

# Navigation
pages = ["Overview", "Repair", "Verification", "History"]
selected_page = st.radio("Navigate", pages, index=pages.index(st.session_state.get("page", "Overview")))
st.session_state["page"] = selected_page

if selected_page == "Overview":
    st.header("🛠️ Pump Diagnosis Overview")
    uploaded_file = st.file_uploader("Upload audio file (WAV)", type=["wav", "mp3", "ogg"])
    if uploaded_file:
        analysis = analyze_audio(uploaded_file)
        st.subheader("Prototype Diagnosis")
        st.write(analysis["diagnosis"])
        if st.button("Submit to Backend"):
            # Build payload
            payload = {
                "case_id": f"CASE-{uuid.uuid4().hex[:8]}",
                "pump_id": f"pump-{uuid.uuid4().hex[:4]}",
                "current_state": "DIAGNOSIS_PENDING",
                "ml_prediction": analysis["diagnosis"]["diagnosis"],
                "ml_confidence": analysis["diagnosis"]["confidence"],
            }
            # Store pump_id for later steps
            st.session_state["pump_id"] = payload["pump_id"]
            try:
                resp = api.evaluate(payload)
                st.success("Backend evaluation completed")
                st.json(resp)
                # Update state based on authoritative response
                st.session_state["authoritative_state"] = resp.get("final_state")
            except Exception as e:
                st.error(f"Backend error: {e}")

elif selected_page == "Repair":
    st.header("🔧 Repair Workflow")
    pump_id = st.session_state.get("pump_id")
    if not pump_id:
        st.info("Run a diagnosis first to obtain a pump ID.")
    else:
        st.write(f"Pump ID: {pump_id}")
        if st.button("Start Repair"):
            try:
                resp = api.repair_start(pump_id)
                st.success("Repair started")
                st.json(resp)
                st.session_state["authoritative_state"] = resp.get("current_state")
            except Exception as e:
                st.error(f"Error starting repair: {e}")
        if st.button("Complete Repair"):
            try:
                resp = api.repair_complete(pump_id)
                st.success("Repair completed")
                st.json(resp)
                st.session_state["authoritative_state"] = resp.get("current_state")
            except Exception as e:
                st.error(f"Error completing repair: {e}")

elif selected_page == "Verification":
    st.header("✅ Post‑Repair Verification")
    pump_id = st.session_state.get("pump_id")
    if not pump_id:
        st.info("Run a diagnosis first to obtain a pump ID.")
    else:
        st.write(f"Pump ID: {pump_id}")
        # Allow user to modify prediction/confidence if desired
        pred = st.selectbox("Prediction", ["NORMAL", "ABNORMAL"], index=0)
        conf = st.slider("Confidence", min_value=0.0, max_value=1.0, value=0.6, step=0.01)
        if st.button("Verify"):
            payload = {
                "case_id": f"VERIFY-{uuid.uuid4().hex[:8]}",
                "pump_id": pump_id,
                "current_state": "VERIFICATION_PENDING",
                "ml_prediction": pred,
                "ml_confidence": conf,
            }
            try:
                resp = api.verify(payload)
                st.success("Verification completed")
                st.json(resp)
                st.session_state["authoritative_state"] = resp.get("final_state")
            except Exception as e:
                st.error(f"Verification error: {e}")

elif selected_page == "History":
    st.header("📜 Pump History")
    pump_id = st.text_input("Enter Pump ID")
    if pump_id and st.button("Load History"):
        try:
            history = api.get_history(pump_id)
            st.json(history)
        except Exception as e:
            st.error(f"Failed to load history: {e}")

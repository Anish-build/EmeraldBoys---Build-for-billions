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
# CANONICAL AUDIO PROCESSING & DIAGNOSIS PIPELINE
# ============================================================
from audio_processing import (
    load_audio,
    extract_audio_features,
    diagnose_audio,
    process_audio_file,
    analyze_audio,
    AudioProcessingError,
    get_model_status,
    FEATURE_KEYS,
)

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

div[data-testid="stRadio"] div[role="radiogroup"]::-webkit-scrollbar {
    display: none;
}

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

div[data-testid="stRadio"] div[role="radiogroup"] label > div:first-child {
    display: none !important;
}

@keyframes navIndicatorIn {
    from {
        opacity: 0;
        transform: scaleX(.35);
    }

    to {
        opacity: 1;
        transform: scaleX(1);
    }
}

/* ------------------------------------------------------------
   TOP BRAND BAR
   ------------------------------------------------------------ */
.aqua-topbar {
    position: relative;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 18px;
    width: 100%;
    min-height: 58px;
    padding: 10px 14px;
    margin-bottom: 10px;
    border: 1px solid rgba(108,201,255,.12);
    border-radius: 18px;
    background: linear-gradient(110deg, rgba(10,26,44,.88), rgba(4,11,22,.93));
    box-shadow: 0 14px 50px rgba(0,0,0,.24), inset 0 1px rgba(255,255,255,.04);
    isolation: isolate;
}

.aqua-topbar::after {
    content: '';
    position: absolute;
    top: 0;
    bottom: 0;
    left: -25%;
    width: 20%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,.06), transparent);
    transform: skewX(-18deg) translate3d(0,0,0);
    animation: topSweep 11s ease-in-out infinite;
    pointer-events: none;
}

@keyframes topSweep {
    0%, 74% { left: -25%; opacity: 0; }
    82% { opacity: .8; }
    94%, 100% { left: 125%; opacity: 0; }
}

.aqua-home-link {
    position: relative;
    z-index: 2;
    display: inline-flex;
    align-items: center;
    gap: 9px;
    min-width: 145px;
    color: #eefcff !important;
    text-decoration: none !important;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 17px;
    font-weight: 700;
    letter-spacing: -.025em;
    transition: transform .22s var(--ease), color .22s ease;
}

.aqua-home-link:hover {
    color: var(--aqua2) !important;
    transform: translate3d(0,-1px,0) scale(1.01);
}

.aqua-top-copy {
    position: relative;
    z-index: 2;
    text-align: center;
    flex: 1;
}

.aqua-top-copy-label {
    color: #60778f;
    font-size: 8px;
    font-weight: 800;
    letter-spacing: 1.7px;
}

.aqua-top-copy-value {
    margin-top: 4px;
    color: #eafcff;
    font-size: 11px;
    font-weight: 700;
}

.aqua-live {
    position: relative;
    z-index: 2;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 7px;
    min-width: 110px;
    padding: 7px 10px;
    border-radius: 999px;
    border: 1px solid rgba(57,227,161,.18);
    background: rgba(57,227,161,.04);
    color: #76f2c4;
    font-size: 8px;
    font-weight: 800;
    letter-spacing: .9px;
    text-align: center;
    animation: liveBreathe 3.5s ease-in-out infinite;
}

.aqua-live::before {
    content: '';
    width: 6px;
    height: 6px;
    flex: 0 0 6px;
    border-radius: 50%;
    background: currentColor;
    box-shadow: 0 0 10px currentColor;
    animation: liveDot 1.7s ease-in-out infinite;
}

@keyframes liveBreathe {
    50% { box-shadow: 0 0 24px rgba(57,227,161,.10); }
}

@keyframes liveDot {
    50% { transform: scale(1.35); opacity: .62; }
}

/* ------------------------------------------------------------
   TOP NAV — stable layout, smooth click/hover states
   ------------------------------------------------------------ */
.aqua-nav {
    position: sticky;
    top: 10px;
    z-index: 50;
    display: flex;
    align-items: stretch;
    gap: 4px;
    width: 100%;
    overflow-x: auto;
    overflow-y: hidden;
    padding: 5px;
    margin: 0 0 18px;
    border: 1px solid rgba(108,201,255,.11);
    border-radius: 15px;
    background: rgba(4,10,19,.88);
    box-shadow: 0 14px 42px rgba(0,0,0,.25), inset 0 1px rgba(255,255,255,.03);
    backdrop-filter: blur(12px);
    scrollbar-width: none;
}

.aqua-nav::-webkit-scrollbar { display: none; }

.aqua-nav-link {
    position: relative;
    flex: 1 0 auto;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 7px;
    min-height: 39px;
    padding: 7px 12px;
    border-radius: 10px;
    color: #8094a8 !important;
    text-decoration: none !important;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: .02em;
    white-space: nowrap;
    transition:
        transform .24s var(--ease),
        color .22s ease,
        background-color .22s ease,
        box-shadow .22s ease;
}

.aqua-nav-link:hover {
    color: #eaffff !important;
    background: rgba(49,231,255,.055);
    transform: translate3d(0,-1px,0);
}

.aqua-nav-link:active {
    transform: translate3d(0,1px,0) scale(.975);
}

.aqua-nav-link::after {
    content: '';
    position: absolute;
    left: 18%;
    right: 18%;
    bottom: 1px;
    height: 2px;
    border-radius: 999px;
    background: linear-gradient(90deg, transparent, var(--aqua), transparent);
    opacity: 0;
    transform: scaleX(.35);
    transition: transform .28s var(--ease), opacity .22s ease;
}

.aqua-nav-link:hover::after {
    opacity: .55;
    transform: scaleX(1);
}

.aqua-nav-link.active {
    color: #f4feff !important;
    background: linear-gradient(135deg, rgba(49,231,255,.10), rgba(76,131,255,.08));
    box-shadow: inset 0 0 0 1px rgba(49,231,255,.14), 0 0 20px rgba(49,231,255,.045);
}

.aqua-nav-link.active::after {
    opacity: 1;
    transform: scaleX(1);
}

.aqua-nav-dot {
    width: 5px;
    height: 5px;
    flex: 0 0 5px;
    border-radius: 50%;
    background: #52687e;
    transition: background-color .22s ease, box-shadow .22s ease, transform .22s ease;
}

.aqua-nav-link:hover .aqua-nav-dot,
.aqua-nav-link.active .aqua-nav-dot {
    background: var(--aqua);
    box-shadow: 0 0 9px rgba(49,231,255,.8);
    transform: scale(1.08);
}

/* ------------------------------------------------------------
   HERO
   ------------------------------------------------------------ */
.aqua-hero {
    position: relative;
    overflow: hidden;
    padding: 18px 20px 19px;
    margin: 0 0 20px;
    border: 1px solid rgba(108,201,255,.11);
    border-radius: 20px;
    background:
        radial-gradient(circle at 88% 50%, rgba(49,231,255,.07), transparent 20%),
        linear-gradient(110deg, rgba(10,23,39,.83), rgba(4,11,22,.86));
    box-shadow: 0 18px 60px rgba(0,0,0,.22), inset 0 1px rgba(255,255,255,.03);
}

.aqua-hero::before {
    content: '';
    position: absolute;
    left: -10%;
    right: -10%;
    top: 50%;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(49,231,255,.18), transparent);
    opacity: .7;
}

.aqua-kicker {
    position: relative;
    z-index: 1;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 8px;
    border: 1px solid rgba(49,231,255,.13);
    border-radius: 999px;
    background: rgba(49,231,255,.035);
    color: #79efff;
    font-size: 8px;
    font-weight: 800;
    letter-spacing: 1.45px;
}

.aqua-hero-title {
    position: relative;
    z-index: 1;
    margin-top: 9px;
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(2rem, 4vw, 3.3rem);
    line-height: .96;
    font-weight: 700;
    letter-spacing: -.065em;
    color: #f4feff;
}

.aqua-hero-copy {
    position: relative;
    z-index: 1;
    margin-top: 9px;
    max-width: 780px;
    color: #7d91a7;
    font-size: 12px;
    line-height: 1.55;
}

/* ------------------------------------------------------------
   PANELS
   ------------------------------------------------------------ */
.aqua-panel {
    position: relative;
    overflow: hidden;
    border: 1px solid rgba(108,201,255,.10);
    border-radius: 18px;
    background: linear-gradient(145deg, rgba(12,27,45,.79), rgba(4,11,22,.77));
    box-shadow: 0 22px 62px rgba(0,0,0,.22), inset 0 1px rgba(255,255,255,.028);
    transition: transform .28s var(--ease), border-color .28s ease, box-shadow .28s ease;
}

.aqua-panel::before {
    content: '';
    position: absolute;
    top: 0;
    left: -120%;
    width: 45%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,.035), transparent);
    transform: skewX(-16deg) translate3d(0,0,0);
    animation: panelLight 12s ease-in-out infinite;
    pointer-events: none;
}

@keyframes panelLight {
    0%, 70% { left: -120%; opacity: 0; }
    80% { opacity: .55; }
    94%, 100% { left: 150%; opacity: 0; }
}

.aqua-panel:hover {
    transform: translate3d(0,-4px,0);
    border-color: rgba(49,231,255,.19);
    box-shadow: 0 30px 80px rgba(0,0,0,.30), 0 0 26px rgba(49,231,255,.045);
}

.aqua-panel-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 14px;
    padding: 13px 15px;
    border-bottom: 1px solid rgba(108,201,255,.07);
}

.aqua-panel-label {
    color: #60778e;
    font-size: 8px;
    font-weight: 800;
    letter-spacing: 1.5px;
}

.aqua-panel-title {
    margin-top: 4px;
    color: #eefcff;
    font-size: 13px;
    font-weight: 700;
}

.aqua-panel-body { padding: 15px; }

/* ------------------------------------------------------------
   CHIPS
   ------------------------------------------------------------ */
.js-chip {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 5px;
    min-height: 22px;
    padding: 4px 7px;
    border: 1px solid rgba(108,201,255,.12);
    border-radius: 999px;
    background: rgba(108,201,255,.035);
    color: #9ab0c2;
    font-size: 7px;
    font-weight: 800;
    letter-spacing: .8px;
    white-space: nowrap;
}

.js-chip-green { color: #79efc4; border-color: rgba(57,227,161,.18); background: rgba(57,227,161,.045); }
.js-chip-blue { color: #91dfff; border-color: rgba(49,231,255,.18); background: rgba(49,231,255,.045); }
.js-chip-yellow { color: #ffdc77; border-color: rgba(255,214,103,.17); background: rgba(255,214,103,.045); }
.js-chip-red { color: #ff8ca0; border-color: rgba(255,100,127,.18); background: rgba(255,100,127,.045); }

/* ------------------------------------------------------------
   METRICS
   ------------------------------------------------------------ */
div[data-testid="stMetric"] {
    position: relative;
    overflow: hidden;
    min-height: 112px;
    padding: 1.18rem !important;
    border-radius: 18px;
    border: 1px solid rgba(108,201,255,.10);
    background: linear-gradient(145deg, rgba(14,29,49,.84), rgba(4,12,24,.80));
    box-shadow: 0 18px 55px rgba(0,0,0,.23), inset 0 1px rgba(255,255,255,.035);
    transition: transform .28s var(--ease), border-color .28s ease, box-shadow .28s ease;
}

div[data-testid="stMetric"]::before {
    content: '';
    position: absolute;
    width: 180px;
    height: 180px;
    top: -110px;
    right: -70px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(49,231,255,.14), transparent 68%);
    transition: transform .4s var(--ease);
}

div[data-testid="stMetric"]::after {
    content: '';
    position: absolute;
    top: -10%;
    left: -125%;
    width: 58%;
    height: 120%;
    transform: skewX(-18deg) translate3d(0,0,0);
    background: linear-gradient(90deg, transparent, rgba(255,255,255,.045), transparent);
    animation: metricSweep 12s ease-in-out infinite;
}

@keyframes metricSweep {
    0%, 76% { left: -125%; opacity: 0; }
    84% { opacity: .65; }
    96%, 100% { left: 150%; opacity: 0; }
}

div[data-testid="stMetric"]:hover {
    transform: translate3d(0,-6px,0) scale(1.012);
    border-color: rgba(49,231,255,.28);
    box-shadow: 0 26px 72px rgba(0,0,0,.32), 0 0 25px rgba(49,231,255,.055);
}

div[data-testid="stMetric"]:hover::before { transform: scale(1.35); }
div[data-testid="stMetricLabel"] { color: #8195aa !important; }
div[data-testid="stMetricValue"] { font-family: 'Space Grotesk', sans-serif !important; color: #f1fcff !important; font-weight: 700 !important; }

/* ------------------------------------------------------------
   BUTTONS
   ------------------------------------------------------------ */
.stButton > button {
    position: relative;
    overflow: hidden;
    min-height: 46px;
    border-radius: 13px;
    border: 1px solid rgba(49,231,255,.18);
    background: linear-gradient(120deg, rgba(49,231,255,.07), rgba(76,131,255,.09), rgba(149,109,255,.07));
    color: #eafcff;
    font-weight: 700;
    transition: transform .20s var(--ease), border-color .20s ease, box-shadow .20s ease;
}

.stButton > button::after {
    content: '';
    position: absolute;
    top: -20%;
    bottom: -20%;
    left: -120%;
    width: 68%;
    transform: skewX(-18deg) translate3d(0,0,0);
    background: linear-gradient(90deg, transparent, rgba(255,255,255,.11), transparent);
    transition: left .5s var(--ease);
}

.stButton > button:hover {
    transform: translate3d(0,-3px,0);
    border-color: rgba(49,231,255,.48);
    box-shadow: 0 10px 32px rgba(0,0,0,.25), 0 0 22px rgba(49,231,255,.07);
}

.stButton > button:hover::after { left: 150%; }
.stButton > button:active { transform: translate3d(0,1px,0) scale(.98); }

/* ------------------------------------------------------------
   INPUTS + UPLOADER
   ------------------------------------------------------------ */
.stTextInput input,
.stTextArea textarea,
.stSelectbox div[data-baseweb="select"] {
    background: rgba(4,12,24,.84) !important;
    color: #effcff !important;
    border: 1px solid rgba(108,201,255,.11) !important;
    border-radius: 12px !important;
    transition: transform .20s var(--ease), border-color .20s ease, box-shadow .20s ease;
}

.stTextInput input:hover,
.stTextArea textarea:hover,
.stSelectbox div[data-baseweb="select"]:hover {
    border-color: rgba(49,231,255,.28) !important;
}

.stTextInput input:focus,
.stTextArea textarea:focus {
    transform: translate3d(0,-1px,0);
    border-color: rgba(49,231,255,.55) !important;
    box-shadow: 0 0 0 3px rgba(49,231,255,.045), 0 0 24px rgba(49,231,255,.065);
}

section[data-testid="stFileUploaderDropzone"] {
    min-height: 150px;
    border-radius: 18px !important;
    border: 1px dashed rgba(49,231,255,.27) !important;
    background: radial-gradient(circle at 50% 45%, rgba(49,231,255,.06), transparent 50%), rgba(4,12,24,.72) !important;
    box-shadow: inset 0 0 44px rgba(49,231,255,.016);
    transition: transform .25s var(--ease), border-color .25s ease, box-shadow .25s ease;
}

section[data-testid="stFileUploaderDropzone"]:hover {
    transform: translate3d(0,-4px,0);
    border-color: rgba(49,231,255,.68) !important;
    box-shadow: 0 20px 55px rgba(0,0,0,.25), 0 0 30px rgba(49,231,255,.07), inset 0 0 42px rgba(49,231,255,.028);
}

/* ------------------------------------------------------------
   TABLES / CHARTS / ALERTS
   ------------------------------------------------------------ */
div[data-testid="stDataFrame"],
div[data-testid="stVegaLiteChart"] {
    border-radius: 16px;
    overflow: hidden;
    border: 1px solid rgba(108,201,255,.08);
    box-shadow: 0 18px 55px rgba(0,0,0,.18);
    transition: transform .26s var(--ease), box-shadow .26s ease, border-color .26s ease;
}

div[data-testid="stDataFrame"]:hover,
div[data-testid="stVegaLiteChart"]:hover {
    transform: translate3d(0,-3px,0);
    border-color: rgba(49,231,255,.17);
    box-shadow: 0 24px 68px rgba(0,0,0,.25), 0 0 22px rgba(49,231,255,.04);
}

div[data-testid="stAlert"] {
    border-radius: 14px !important;
    box-shadow: 0 12px 35px rgba(0,0,0,.18);
    animation: alertEnter .42s var(--ease) both;
}

@keyframes alertEnter {
    from { opacity: 0; transform: translate3d(0,6px,0); }
    to { opacity: 1; transform: none; }
}

hr {
    border: 0 !important;
    height: 1px;
    margin: 1.45rem 0 !important;
    background: linear-gradient(90deg, transparent, rgba(49,231,255,.21), rgba(149,109,255,.18), transparent) !important;
}

/* ------------------------------------------------------------
   FLOATING BACKGROUND FX
   ------------------------------------------------------------ */
.aqua-fx-layer {
    position: fixed;
    inset: 0;
    z-index: 1;
    pointer-events: none;
    overflow: hidden;
}

.aqua-orb {
    position: absolute;
    border-radius: 50%;
    opacity: .075;
    mix-blend-mode: screen;
}

.aqua-orb-1 {
    width: 180px;
    height: 180px;
    left: 4%;
    top: 24%;
    background: radial-gradient(circle, var(--aqua), transparent 70%);
    animation: orb1 24s ease-in-out infinite;
}

.aqua-orb-2 {
    width: 200px;
    height: 200px;
    right: 5%;
    top: 48%;
    background: radial-gradient(circle, var(--violet), transparent 70%);
    animation: orb2 29s ease-in-out infinite;
}

.aqua-orb-3 {
    width: 130px;
    height: 130px;
    left: 46%;
    bottom: 4%;
    background: radial-gradient(circle, var(--blue), transparent 70%);
    animation: orb3 31s ease-in-out infinite;
}

@keyframes orb1 {
    0%,100% { transform: translate3d(0,0,0) scale(.96); }
    50% { transform: translate3d(110px,-60px,0) scale(1.08); }
}

@keyframes orb2 {
    0%,100% { transform: translate3d(0,0,0) scale(1); }
    50% { transform: translate3d(-120px,70px,0) scale(1.09); }
}

@keyframes orb3 {
    0%,100% { transform: translate3d(0,0,0); }
    50% { transform: translate3d(70px,-45px,0) scale(1.08); }
}

/* ------------------------------------------------------------
   SIGNAL / AI DETAILS
   ------------------------------------------------------------ */
.ai-thinking {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    color: var(--aqua2);
    font-weight: 700;
}

.ai-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--aqua);
    box-shadow: 0 0 10px rgba(49,231,255,.75);
    animation: aiDot 1.7s ease-in-out infinite;
}

@keyframes aiDot {
    50% { transform: scale(1.35); opacity: .6; }
}

/* ------------------------------------------------------------
   RESPONSIVE
   ------------------------------------------------------------ */
@media (max-width: 1000px) {
    .aqua-topbar { padding: 9px 10px; }
    .aqua-top-copy { display: none; }
    .aqua-home-link { min-width: auto; }
    .aqua-nav { top: 5px; }
    .aqua-nav-link { flex: 0 0 auto; }
    .block-container { padding-left: 1rem !important; padding-right: 1rem !important; }
}

@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: .01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: .01ms !important;
        scroll-behavior: auto !important;
    }
}
"""


st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


# ============================================================
# AMBIENT UI EFFECTS
# ============================================================

# ============================================================
# TOP BRAND BAR + SAME-TAB TOP NAVIGATION
# ============================================================

NAV = [
    "Overview",
    "New Inspection",
    "AI Diagnosis",
    "Agent Workflow",
    "Repair Dispatch",
    "Verification",
    "Pump History",
    "Audit & Safety",
]

if "page_nav" not in st.session_state:
    st.session_state.page_nav = "Overview"

# Navigation requests triggered by buttons lower on the page are applied
# here, before the radio widget is instantiated. This avoids Streamlit's
# "cannot be modified after the widget ... is instantiated" error.
if "pending_page" not in st.session_state:
    st.session_state.pending_page = None

if st.session_state.pending_page in NAV:
    st.session_state.page_nav = st.session_state.pending_page
    st.session_state.pending_page = None

if st.session_state.page_nav not in NAV:
    st.session_state.page_nav = "Overview"

if st.button("💧  AquaFusion", key="home_button", width="stretch"):
    st.session_state.page_nav = "Overview"
    st.rerun()

page = st.radio(
    "Primary navigation",
    NAV,
    index=NAV.index(st.session_state.page_nav),
    horizontal=True,
    key="page_nav",
    label_visibility="collapsed",
)

st.session_state.page = page

# ============================================================
# PAGE HEADER
# ============================================================

page_descriptions = {
    "Overview": "A live operating picture of pump health and maintenance activity.",
    "New Inspection": "Capture a pump recording and turn raw sound into evidence.",
    "AI Diagnosis": "Review acoustic evidence, confidence, severity, and next action.",
    "Agent Workflow": "Watch the maintenance agent move from observation to verification.",
    "Repair Dispatch": "Convert a validated anomaly into a field-ready service case.",
    "Verification": "Use a second acoustic recording to confirm whether the repair worked.",
    "Pump History": "Trace inspections, maintenance events, and pump state over time.",
    "Audit & Safety": "Review the safety gates and consequential actions taken by the system.",
}

st.markdown(
    f"""
    <div class="aqua-hero">
        <div class="aqua-kicker">✦ AQUAFUSION OPERATIONS CORE</div>
        <div class="aqua-hero-title">{page}</div>
        <div class="aqua-hero-copy">{page_descriptions[page]}</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================
# ============================================================
# NAVIGATION HELPER
# ============================================================

def go_to(target):
    # Do not change the page_nav widget state after the radio has been
    # instantiated. Store the destination and apply it at the top of the
    # next Streamlit run instead.
    st.session_state.pending_page = target if target in NAV else "Overview"
    st.rerun()




def add_audit(action, detail, severity="INFO"):
    event = {
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "action": action,
        "detail": detail,
        "severity": severity,
    }
    st.session_state.audit.append(event)


def stage_badge(label, status):
    cls = {
        "COMPLETE": "js-chip-green",
        "ACTIVE": "js-chip-blue",
        "WAITING": "js-chip-yellow",
        "ESCALATED": "js-chip-red",
    }.get(status, "js-chip-blue")
    return f'<span class="js-chip {cls}">{label.upper()} · {status}</span>'


def svg_signal(mode="live"):
    heights = [18, 38, 26, 55, 32, 68, 44, 25, 61, 36, 52, 30, 48, 22, 57]
    bars = []
    for i, height in enumerate(heights):
        x = 90 + i * 25
        delay = (i % 8) * 0.09
        bars.append(
            f'<rect x="{x}" y="{72-height}" width="5" height="{height}" rx="3" '
            f'style="animation: signalBar 1.1s ease-in-out {delay}s infinite; transform-origin:center" />'
        )
    label = "LIVE SIGNAL" if mode == "live" else "ACOUSTIC SAMPLE"
    return f"""
    <div class="aqua-panel" style="margin-bottom:18px">
        <div class="aqua-panel-head">
            <div><div class="aqua-panel-label">SIGNAL MONITOR</div><div class="aqua-panel-title">{label}</div></div>
            <div class="aqua-live">ACTIVE</div>
        </div>
        <div class="aqua-panel-body" style="padding:8px 16px 16px">
            <svg viewBox="0 0 520 110" width="100%" height="120" preserveAspectRatio="none">
                <defs>
                    <linearGradient id="sig" x1="0" x2="1">
                        <stop offset="0%" stop-color="#31e7ff" stop-opacity=".18"/>
                        <stop offset="50%" stop-color="#31e7ff" stop-opacity="1"/>
                        <stop offset="100%" stop-color="#956dff" stop-opacity=".35"/>
                    </linearGradient>
                </defs>
                <path d="M0 72 H520" stroke="rgba(120,170,210,.10)" stroke-width="1"/>
                <path d="M0 40 H520" stroke="rgba(120,170,210,.06)" stroke-width="1"/>
                <g fill="url(#sig)">{''.join(bars)}</g>
                <rect x="0" y="0" width="3" height="100" fill="#8ef8ff" opacity=".9">
                    <animate attributeName="x" from="0" to="517" dur="2.6s" repeatCount="indefinite"/>
                </rect>
            </svg>
        </div>
    </div>
    <style>
    @keyframes signalBar {{ 0%,100% {{ transform:scaleY(.42); opacity:.45 }} 50% {{ transform:scaleY(1); opacity:1 }} }}
    </style>
    """


def svg_trend():
    pts = [(0, 88), (50, 76), (95, 82), (140, 58), (185, 63), (230, 44), (275, 50), (320, 36), (365, 40), (410, 27), (455, 31), (500, 18)]
    path = "M " + " L ".join(f"{x} {y}" for x, y in pts)
    return f"""
    <div class="aqua-panel">
        <div class="aqua-panel-head">
            <div><div class="aqua-panel-label">PUMP HEALTH TREND</div><div class="aqua-panel-title">30-day acoustic stability</div></div>
            <div class="js-chip js-chip-blue">LIVE MODEL VIEW</div>
        </div>
        <div class="aqua-panel-body">
            <svg viewBox="0 0 520 125" width="100%" height="150" preserveAspectRatio="none">
                <defs>
                    <linearGradient id="area" x1="0" x2="0" y1="0" y2="1">
                        <stop offset="0%" stop-color="#31e7ff" stop-opacity=".22"/>
                        <stop offset="100%" stop-color="#31e7ff" stop-opacity="0"/>
                    </linearGradient>
                </defs>
                <path d="M0 110 H520" stroke="rgba(120,170,210,.08)"/>
                <path d="M0 75 H520" stroke="rgba(120,170,210,.06)"/>
                <path d="M0 40 H520" stroke="rgba(120,170,210,.06)"/>
                <path d="{path} L520 125 L0 125 Z" fill="url(#area)"/>
                <path d="{path}" fill="none" stroke="#31e7ff" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" stroke-dasharray="900" stroke-dashoffset="900">
                    <animate attributeName="stroke-dashoffset" from="900" to="0" dur="2.4s" fill="freeze"/>
                </path>
                <circle cx="500" cy="18" r="5" fill="#8ef8ff">
                    <animate attributeName="r" values="4;7;4" dur="1.6s" repeatCount="indefinite"/>
                    <animate attributeName="opacity" values="1;.4;1" dur="1.6s" repeatCount="indefinite"/>
                </circle>
            </svg>
        </div>
    </div>
    """


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Pumps Monitored", "24", "+3 this month")
    m2.metric("Healthy", "17", "71% of network")
    m3.metric("Needs Attention", "5", "2 high severity")
    m4.metric("Active Cases", "2", "1 awaiting verification")

    left, right = st.columns([1.5, 1], gap="large")

    with left:
        st.markdown(svg_trend(), unsafe_allow_html=True)

    with right:
        st.markdown(svg_signal(), unsafe_allow_html=True)

    st.divider()

    left2, right2 = st.columns([1.15, 1], gap="large")

    with left2:
        st.markdown(
            """
            <div class="aqua-panel">
                <div class="aqua-panel-head">
                    <div><div class="aqua-panel-label">NETWORK SNAPSHOT</div><div class="aqua-panel-title">Pump condition distribution</div></div>
                    <div class="js-chip js-chip-green">24 PUMPS</div>
                </div>
                <div class="aqua-panel-body">
                    <div style="margin-bottom:14px">
                        <div style="display:flex;justify-content:space-between;font-size:11px;color:#93a7ba;margin-bottom:6px"><span>Healthy</span><b style="color:#72f3c3">17 · 71%</b></div>
                        <div style="height:8px;border-radius:20px;background:#0b1422;overflow:hidden"><div style="width:71%;height:100%;background:linear-gradient(90deg,#39e3a1,#31e7ff);box-shadow:0 0 16px rgba(49,231,255,.3);animation:barIn 1.1s var(--ease)"></div></div>
                    </div>
                    <div style="margin-bottom:14px">
                        <div style="display:flex;justify-content:space-between;font-size:11px;color:#93a7ba;margin-bottom:6px"><span>Needs attention</span><b style="color:#ffd667">5 · 21%</b></div>
                        <div style="height:8px;border-radius:20px;background:#0b1422;overflow:hidden"><div style="width:21%;height:100%;background:linear-gradient(90deg,#ffd667,#ff9d5f);box-shadow:0 0 16px rgba(255,214,103,.2);animation:barIn 1.25s var(--ease)"></div></div>
                    </div>
                    <div>
                        <div style="display:flex;justify-content:space-between;font-size:11px;color:#93a7ba;margin-bottom:6px"><span>Under repair / escalated</span><b style="color:#ff7f97">2 · 8%</b></div>
                        <div style="height:8px;border-radius:20px;background:#0b1422;overflow:hidden"><div style="width:8%;height:100%;background:linear-gradient(90deg,#ff647f,#956dff);box-shadow:0 0 16px rgba(255,100,127,.2);animation:barIn 1.35s var(--ease)"></div></div>
                    </div>
                </div>
            </div>
            <style>@keyframes barIn{from{width:0}to{opacity:1}}</style>
            """,
            unsafe_allow_html=True,
        )

    with right2:
        st.markdown(
            """
            <div class="aqua-panel">
                <div class="aqua-panel-head">
                    <div><div class="aqua-panel-label">AGENT ACTIVITY</div><div class="aqua-panel-title">Autonomous maintenance loop</div></div>
                    <div class="aqua-live">RUNNING</div>
                </div>
                <div class="aqua-panel-body">
                    <div style="display:grid;gap:10px">
                        <div class="js-chip js-chip-green">OBSERVE · COMPLETE</div>
                        <div class="js-chip js-chip-green">UNDERSTAND · COMPLETE</div>
                        <div class="js-chip js-chip-blue">DIAGNOSE · ACTIVE</div>
                        <div class="js-chip js-chip-yellow">PLAN · QUEUED</div>
                        <div class="js-chip js-chip-yellow">ACT · WAITING</div>
                        <div class="js-chip js-chip-yellow">VERIFY · WAITING</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()
    st.subheader("Recent inspections")

    rows = [
        {"Pump": "JAL-014", "Location": "Nitte Village", "Result": "Healthy", "Severity": "Low", "Time": "Today"},
        {"Pump": "JAL-009", "Location": "Karkala", "Result": "Abnormal", "Severity": "High", "Time": "Today"},
        {"Pump": "JAL-021", "Location": "Belman", "Result": "Under Verification", "Severity": "Medium", "Time": "Yesterday"},
        {"Pump": "JAL-007", "Location": "Mangalore Rural", "Result": "Healthy", "Severity": "Low", "Time": "2 days ago"},
        {"Pump": "JAL-018", "Location": "Mulki", "Result": "Needs Attention", "Severity": "Medium", "Time": "3 days ago"},
    ]
    st.dataframe(rows, width="stretch", hide_index=True)

    st.divider()
    a1, a2, a3 = st.columns(3)
    with a1:
        if st.button("🎙️  Start inspection", width="stretch"):
            go_to("New Inspection")
    with a2:
        if st.button("🧠  Open AI diagnosis", width="stretch"):
            go_to("AI Diagnosis")
    with a3:
        if st.button("🛡️  Safety center", width="stretch"):
            go_to("Audit & Safety")



# ============================================================
# NEW INSPECTION
# ============================================================

elif page == "New Inspection":
    col1, col2 = st.columns([1, 1.3], gap="large")

    with col1:
        st.subheader("1 · Pump context")
        pump_id = st.selectbox("Pump", ["JAL-014", "JAL-009", "JAL-021", "JAL-007", "JAL-018"])
        location = st.text_input("Location", value="Nitte Village")
        observation = st.text_area(
            "Field observation",
            placeholder="Example: handle feels harder than usual; metallic sound noticed.",
            height=115,
        )

        st.subheader("2 · Acoustic recording")
        audio_mode = st.radio(
            "Audio Input Mode:",
            ["Upload Custom Audio File", "Select Benchmark Acoustic Sample"],
            horizontal=True
        )

        audio_input = None
        audio_name = "recording.wav"

        if audio_mode == "Select Benchmark Acoustic Sample":
            benchmark_samples = {
                "Normal Pump Rhythm (Clean Harmonic Baseline)": "samples/audio/normal_pump_sample.wav",
                "Seal Leak Anomaly (High Frequency Friction)": "samples/audio/seal_leak_anomaly.wav",
                "Bearing Friction Chatter (Mechanical Degradation)": "samples/audio/bearing_friction_high.wav",
                "Low Confidence Chatter (Acoustic Ambiguity)": "samples/audio/low_confidence_chatter.wav"
            }
            sample_choice = st.selectbox("Select Benchmark Audio Sample", list(benchmark_samples.keys()))
            sample_path = Path(benchmark_samples[sample_choice])
            if sample_path.exists():
                with open(sample_path, "rb") as f:
                    audio_input = f.read()
                audio_name = sample_path.name
                st.audio(audio_input, format="audio/wav")
                st.markdown(
                    f'<div class="js-chip js-chip-green">BENCHMARK READY · {audio_name}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.warning(f"Benchmark file not found: {sample_path}")
        else:
            audio_file = st.file_uploader(
                "Upload pump audio",
                type=["wav", "mp3", "m4a", "ogg", "flac"],
                label_visibility="visible",
            )
            if audio_file:
                audio_input = audio_file.getvalue()
                audio_name = audio_file.name
                st.audio(audio_file, format=audio_file.type)
                st.markdown(
                    f'<div class="js-chip js-chip-green">READY · {audio_name}</div>',
                    unsafe_allow_html=True,
                )

    with col2:
        st.markdown(svg_signal("sample"), unsafe_allow_html=True)
        model_st = get_model_status()
        badge_cls = "js-chip-green" if model_st == "TRAINED MODEL" else "js-chip-yellow"
        st.markdown(
            f'''
            <div class="aqua-panel">
                <div class="aqua-panel-head">
                    <div><div class="aqua-panel-label">PIPELINE</div><div class="aqua-panel-title">Inspection sequence</div></div>
                    <div class="{badge_cls}">{model_st}</div>
                </div>
                <div class="aqua-panel-body">
                    <div style="display:grid;gap:10px">
                        <div>◉ <b>Capture</b> <span style="color:#6f849a">· decode 16-bit acoustic waveform</span></div>
                        <div>◉ <b>Preprocess</b> <span style="color:#6f849a">· resample 22.05kHz, normalize</span></div>
                        <div>◉ <b>Extract</b> <span style="color:#6f849a">· 18 features (RMS, ZCR, Centroid, Rolloff, 13 MFCCs)</span></div>
                        <div>◉ <b>Diagnose</b> <span style="color:#6f849a">· standardized multi-feature anomaly scoring</span></div>
                        <div>◉ <b>Decide</b> <span style="color:#6f849a">· neuro-symbolic safety gate</span></div>
                    </div>
                </div>
            </div>
            ''',
            unsafe_allow_html=True,
        )

    st.divider()
    c1, c2 = st.columns([1.5, 1])
    with c1:
        if st.button("⚡  Analyze acoustic recording", width="stretch", type="primary", disabled=audio_input is None):
            try:
                result = analyze_audio(audio_input)
                st.session_state.inspection = {
                    "pump_id": pump_id,
                    "location": location,
                    "observation": observation,
                    "filename": audio_name,
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
                st.session_state.analysis = result
                add_audit("Inspection analyzed", f"{pump_id} ({audio_name}) -> {result['diagnosis']['diagnosis']} ({result['diagnosis']['confidence']*100:.0f}%)")
                go_to("AI Diagnosis")
            except AudioProcessingError as aerr:
                st.error(f"Audio validation error: {aerr}")
            except Exception as exc:
                st.error(f"Audio analysis failed: {exc}")
    with c2:
        st.markdown(
            f'<div class="{badge_cls}">{model_st} · Standardized multi-feature anomaly scoring</div>',
            unsafe_allow_html=True,
        )

    if st.session_state.analysis:
        st.success("Analysis available. Opening the AI Diagnosis module will show the evidence.")


# ============================================================
# AI DIAGNOSIS
# ============================================================

elif page == "AI Diagnosis":
    analysis = st.session_state.analysis
    inspection = st.session_state.inspection

    if not analysis or not inspection:
        st.warning("No inspection has been analyzed yet. Start a New Inspection first.")
        if st.button("Start a new inspection", width="stretch", type="primary"):
            go_to("New Inspection")
    else:
        result = analysis["diagnosis"]
        diagnosis = result["diagnosis"]
        confidence = result["confidence"]
        severity = result["severity"]
        anomaly_score = result.get("anomaly_score", 0.0)
        model_status = result.get("model_status", get_model_status())

        diag_cls = "js-chip-green" if diagnosis == "NORMAL" else "js-chip-red"
        model_cls = "js-chip-green" if model_status == "TRAINED MODEL" else "js-chip-yellow"
        
        st.markdown(
            f'''
            <div class="aqua-panel" style="margin-bottom:18px">
                <div class="aqua-panel-body" style="padding:24px">
                    <div style="display:flex;justify-content:space-between;gap:20px;align-items:center;flex-wrap:wrap">
                        <div>
                            <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px">
                                <span class="aqua-panel-label">AI DECISION · {inspection['pump_id']}</span>
                                <span class="{model_cls}">{model_status}</span>
                            </div>
                            <div style="font-family:'Space Grotesk';font-size:38px;font-weight:700;letter-spacing:-.055em">{diagnosis}</div>
                            <div style="margin-top:6px;color:#7890a5;font-size:12px">{inspection['location']} · {inspection['filename']}</div>
                        </div>
                        <div style="text-align:right">
                            <div class="{diag_cls}">{diagnosis} · {severity} SEVERITY</div>
                            <div style="font-size:30px;font-weight:700;font-family:'Space Grotesk';margin-top:9px">{confidence*100:.1f}%</div>
                            <div style="font-size:10px;color:#647b91;letter-spacing:1px">ANOMALY SCORE: <b>{anomaly_score:.3f}</b></div>
                        </div>
                    </div>
                </div>
            </div>
            ''',
            unsafe_allow_html=True,
        )

        f1, f2, f3, f4 = st.columns(4)
        f1.metric("Duration", f"{analysis['duration']:.2f} s")
        f2.metric("Sample Rate", f"{analysis['sample_rate']:,} Hz")
        f3.metric("RMS Energy", f"{analysis['features']['rms_energy']:.4f}")
        f4.metric("Anomaly Score", f"{anomaly_score:.3f}")

        left, right = st.columns([1.2, 1], gap="large")
        with left:
            st.markdown(svg_signal("sample"), unsafe_allow_html=True)
            st.subheader("Acoustic feature evidence")
            features = analysis["features"]
            evidence = [
                ("Zero Crossing Rate (ZCR)", features["zero_crossing_rate"], min(features["zero_crossing_rate"] / 0.50, 1.0)),
                ("Spectral Centroid (Hz)", features["spectral_centroid"], min(features["spectral_centroid"] / 7000.0, 1.0)),
                ("Spectral Bandwidth (Hz)", features["spectral_bandwidth"], min(features["spectral_bandwidth"] / 5000.0, 1.0)),
                ("Spectral Rolloff (Hz)", features["spectral_rolloff"], min(features["spectral_rolloff"] / 10000.0, 1.0)),
            ]
            for label, raw_val, norm_val in evidence:
                val_str = f"{raw_val:.3f}" if raw_val < 10 else f"{raw_val:.1f}"
                st.markdown(
                    f'''
                    <div style="margin-bottom:12px">
                        <div style="display:flex;justify-content:space-between;font-size:11px;color:#8498ad;margin-bottom:5px">
                            <span>{label}</span><b style="color:#dffaff">{val_str}</b>
                        </div>
                        <div style="height:7px;border-radius:20px;background:#0b1422;overflow:hidden">
                            <div style="height:100%;width:{min(norm_val*100, 100):.1f}%;background:linear-gradient(90deg,#31e7ff,#956dff);box-shadow:0 0 14px rgba(49,231,255,.2)"></div>
                        </div>
                    </div>
                    ''',
                    unsafe_allow_html=True,
                )

            st.markdown("#### Important MFCC Spectral Envelopes")
            m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
            m_col1.metric("MFCC 1 (Power)", f"{features['mfcc_1']:.1f}")
            m_col2.metric("MFCC 2 (Tilt)", f"{features['mfcc_2']:.1f}")
            m_col3.metric("MFCC 3 (Harmonics)", f"{features['mfcc_3']:.1f}")
            m_col4.metric("MFCC 4", f"{features['mfcc_4']:.1f}")
            m_col5.metric("MFCC 5", f"{features['mfcc_5']:.1f}")

        with right:
            st.subheader("Decision evidence & reasons")
            if result.get("reasons"):
                if diagnosis == "ABNORMAL":
                    st.warning("Acoustic anomaly factors detected:")
                else:
                    st.success("Healthy baseline acoustic factors:")
                for reason in result["reasons"]:
                    st.write(f"• {reason}")
            else:
                st.info("No abnormal acoustic indicators detected.")

            st.markdown(
                '''
                <div class="aqua-panel" style="margin-top:16px">
                    <div class="aqua-panel-body">
                        <div class="aqua-panel-label">NEURO-SYMBOLIC SAFETY GATE</div>
                        <div style="margin-top:8px;font-size:13px;font-weight:700">Autonomous repair planning remains conditional.</div>
                        <div style="margin-top:7px;color:#73889e;font-size:11px">
                            Deterministic backend enforces physical state machine rules. Low confidence (<0.40), illegal transitions, or verification failures automatically escalate to human review.
                        </div>
                    </div>
                </div>
                ''',
                unsafe_allow_html=True,
            )

        st.divider()
        x1, x2 = st.columns(2)
        with x1:
            if st.button("🤖  Open agent workflow", width="stretch", type="primary"):
                go_to("Agent Workflow")
        with x2:
            if st.button("🛡️  Review safety gates", width="stretch"):
                go_to("Audit & Safety")

# ============================================================
# AGENT WORKFLOW
# ============================================================

elif page == "Agent Workflow":
    analysis = st.session_state.analysis
    inspection = st.session_state.inspection

    st.subheader("Observe → understand → diagnose → plan → act → verify")

    steps = [
        ("OBSERVE", "Capture pump sound and field context", "COMPLETE" if analysis else "WAITING"),
        ("UNDERSTAND", "Extract acoustic evidence", "COMPLETE" if analysis else "WAITING"),
        ("DIAGNOSE", "Evaluate condition and confidence", "COMPLETE" if analysis else "WAITING"),
        ("PLAN", "Select repair path and parts", "ACTIVE" if analysis and not st.session_state.repair else "WAITING"),
        ("ACT", "Create a simulated field dispatch", "ACTIVE" if st.session_state.repair else "WAITING"),
        ("VERIFY", "Collect post-repair evidence", "ACTIVE" if st.session_state.repair and not st.session_state.verification else "WAITING"),
        ("CLOSE / ESCALATE", "Close the case or route to human review", "COMPLETE" if st.session_state.verification else "WAITING"),
    ]

    for i, (label, description, status) in enumerate(steps, start=1):
        st.markdown(
            f"""
            <div class="aqua-panel" style="margin-bottom:9px">
                <div style="display:grid;grid-template-columns:52px 1fr auto;gap:14px;align-items:center;padding:13px 16px">
                    <div style="width:38px;height:38px;border-radius:12px;display:grid;place-items:center;background:rgba(49,231,255,.05);border:1px solid rgba(49,231,255,.12);color:#93f6ff;font-family:'Space Grotesk';font-weight:700">{i}</div>
                    <div>
                        <div style="font-size:12px;font-weight:800;letter-spacing:.4px">{label}</div>
                        <div style="font-size:10px;color:#71859b;margin-top:3px">{description}</div>
                    </div>
                    <div>{stage_badge(label, status)}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if analysis and not st.session_state.repair:
        st.info("Diagnosis is available. The next agent action is to create a repair plan.")
        if st.button("🔧  Create repair plan", width="stretch", type="primary"):
            diagnosis = analysis["diagnosis"]
            pump_id = inspection["pump_id"]
            part = "Valve assembly" if diagnosis["diagnosis"] == "ABNORMAL" else "No part required"
            st.session_state.repair = {
                "pump_id": pump_id,
                "location": inspection["location"],
                "part": part,
                "status": "Planned",
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            add_audit("Repair planned", f"{pump_id} · {part}")
            st.rerun()


# ============================================================
# REPAIR DISPATCH
# ============================================================

elif page == "Repair Dispatch":
    repair = st.session_state.repair
    inspection = st.session_state.inspection

    if not repair:
        st.warning("No repair plan exists yet. Complete an AI diagnosis and create the plan first.")
        if st.button("Open agent workflow", width="stretch"):
            go_to("Agent Workflow")
    else:
        st.subheader("Field-ready service case")
        c1, c2, c3 = st.columns(3)
        c1.metric("Pump", repair["pump_id"])
        c2.metric("Location", repair["location"])
        c3.metric("Required part", repair["part"])

        st.markdown(
            f"""
            <div class="aqua-panel" style="margin-top:18px">
                <div class="aqua-panel-body">
                    <div class="aqua-panel-label">DISPATCH PACKET</div>
                    <div style="font-size:25px;font-family:'Space Grotesk';font-weight:700;margin-top:7px">Mechanic dispatch prepared</div>
                    <div style="margin-top:9px;color:#7c90a6;font-size:11px">Case references the acoustic finding, selected repair part, pump location, and post-repair verification requirement.</div>
                    <div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:14px">
                        <div class="js-chip js-chip-blue">SIMULATED DISPATCH</div>
                        <div class="js-chip js-chip-yellow">VERIFY AFTER REPAIR</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if repair["status"] != "Dispatched":
            if st.button("📡  Dispatch mechanic", width="stretch", type="primary"):
                repair["status"] = "Dispatched"
                add_audit("Dispatch created", f"{repair['pump_id']} mechanic case dispatched")
                st.success("Simulated mechanic dispatch created.")
                st.rerun()
        else:
            st.success("Dispatch is active. Record the post-repair audio next.")
            if st.button("🎧  Open verification", width="stretch", type="primary"):
                go_to("Verification")



# ============================================================
# VERIFICATION
# ============================================================

elif page == "Verification":
    repair = st.session_state.repair

    if not repair:
        st.warning("There is no active repair case yet. Complete an inspection and repair plan first.")
    else:
        st.subheader("Post-repair acoustic verification")
        st.caption("A repair remains open until a second recording provides evidence confirming normal operation.")

        v_mode = st.radio(
            "Verification Audio Source:",
            ["Select Benchmark Sample", "Upload Post-Repair Audio"],
            horizontal=True
        )

        v_audio = None
        v_name = "verification.wav"

        if v_mode == "Select Benchmark Sample":
            v_samples = {
                "Healthy Pump Restored (normal_pump_sample.wav)": "samples/audio/normal_pump_sample.wav",
                "Persistent Anomaly (seal_leak_anomaly.wav)": "samples/audio/seal_leak_anomaly.wav",
                "Uncertain Chatter (low_confidence_chatter.wav)": "samples/audio/low_confidence_chatter.wav"
            }
            v_choice = st.selectbox("Select Verification Benchmark Sample", list(v_samples.keys()))
            v_path = Path(v_samples[v_choice])
            if v_path.exists():
                with open(v_path, "rb") as f:
                    v_audio = f.read()
                v_name = v_path.name
                st.audio(v_audio, format="audio/wav")
        else:
            v_file = st.file_uploader(
                "Upload post-repair pump audio",
                type=["wav", "mp3", "m4a", "ogg", "flac"],
                key="verification_audio",
            )
            if v_file:
                v_audio = v_file.getvalue()
                v_name = v_file.name
                st.audio(v_file, format=v_file.type)

        if st.button("✅  Analyze verification recording", width="stretch", type="primary", disabled=v_audio is None):
            try:
                verification_result = analyze_audio(v_audio)
                v_diag = verification_result["diagnosis"]
                is_normal = v_diag["diagnosis"] == "NORMAL"
                has_confidence = v_diag["confidence"] >= 0.60
                
                # Deterministic safety rule: only mark verified if normal AND sufficient confidence
                verified = is_normal and has_confidence
                
                status_str = "Verified" if verified else ("Failed Verification (Anomaly Persistent)" if not is_normal else "Ambiguous (Low Confidence)")
                st.session_state.verification = {
                    "status": status_str,
                    "is_verified": verified,
                    "pump_id": repair["pump_id"],
                    "result": v_diag,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
                add_audit(
                    "Verification complete",
                    f"{repair['pump_id']} · {status_str} (Conf: {v_diag['confidence']*100:.0f}%, Score: {v_diag['anomaly_score']:.3f})",
                    "INFO" if verified else "WARNING",
                )
                st.rerun()
            except AudioProcessingError as aerr:
                st.error(f"Verification audio error: {aerr}")
            except Exception as exc:
                st.error(f"Verification analysis failed: {exc}")

        if st.session_state.verification:
            v = st.session_state.verification
            v_res = v["result"]
            if v.get("is_verified"):
                st.success(f"✅ Repair verified for {v['pump_id']}! Acoustic signature conforms to healthy standards. Case can be closed.")
            else:
                st.error(f"⚠️ Verification did not pass for {v['pump_id']} ({v['status']}). Repair did not restore baseline rhythm. Case must be reopened or escalated.")
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Verification Outcome", v_res["diagnosis"])
            c2.metric("Verification Confidence", f"{v_res['confidence']*100:.1f}%")
            c3.metric("Anomaly Score", f"{v_res['anomaly_score']:.3f}")

# ============================================================
# PUMP HISTORY
# ============================================================

elif page == "Pump History":
    pump = st.selectbox("Select pump", ["JAL-014", "JAL-009", "JAL-021", "JAL-007", "JAL-018"])
    st.subheader(f"History · {pump}")

    history = [
        {"Date": "Today", "Event": "Acoustic inspection", "Status": "Healthy", "Confidence": "60%"},
        {"Date": "12 Sep", "Event": "Routine inspection", "Status": "Healthy", "Confidence": "67%"},
        {"Date": "28 Aug", "Event": "Repair verified", "Status": "Verified", "Confidence": "82%"},
        {"Date": "28 Aug", "Event": "Valve replacement", "Status": "Completed", "Confidence": "—"},
    ]
    st.dataframe(history, width="stretch", hide_index=True)

    st.markdown(svg_trend(), unsafe_allow_html=True)


# ============================================================
# AUDIT & SAFETY
# ============================================================

elif page == "Audit & Safety":
    st.subheader("Safety gates")
    gates = [
        ("Low confidence", "Human review", "Required"),
        ("High severity", "Engineer escalation", "Required"),
        ("Conflicting evidence", "Request second recording", "Required"),
        ("Repair not verified", "Reopen / escalate", "Required"),
        ("Verified healthy", "Close case", "Allowed"),
    ]

    for condition, action, rule in gates:
        st.markdown(
            f"""
            <div class="aqua-panel" style="margin-bottom:9px">
                <div style="display:grid;grid-template-columns:1.2fr 1fr auto;gap:14px;align-items:center;padding:13px 16px">
                    <div style="font-size:12px;font-weight:700">{condition}</div>
                    <div style="font-size:10px;color:#7b90a6">→ {action}</div>
                    <div class="js-chip js-chip-blue">{rule.upper()}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()
    st.subheader("Audit trail")
    if st.session_state.audit:
        st.dataframe(st.session_state.audit[::-1], width="stretch", hide_index=True)
    else:
        st.info("No consequential actions recorded in this session yet.")


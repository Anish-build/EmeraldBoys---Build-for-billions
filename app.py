"""
Emerald Boys — AquaFusion Water Intelligence Operations Dashboard
Streamlit Frontend Integrated with Authoritative FastAPI Backend.
"""

import os
import io
import time
import requests
import streamlit as st
from datetime import datetime
from pathlib import Path

# ============================================================
# CONFIGURATION & BACKEND URL
# ============================================================

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(
    page_title="AquaFusion | Emerald Boys Operations",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS if available
css_path = Path(__file__).parent / "style.css"
if css_path.exists():
    try:
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception:
        pass


# ============================================================
# BACKEND API CLIENT
# ============================================================

def check_backend_health():
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=2.0)
        return r.status_code == 200, r.json() if r.status_code == 200 else {}
    except Exception as e:
        return False, {"error": str(e)}


def fetch_all_pumps():
    try:
        r = requests.get(f"{BACKEND_URL}/pumps", timeout=3.0)
        if r.status_code == 200:
            return r.json()
        return []
    except Exception:
        return []


def fetch_pump_state(pump_id: str):
    try:
        r = requests.get(f"{BACKEND_URL}/pumps/{pump_id}", timeout=3.0)
        if r.status_code == 200:
            return r.json()
        return None
    except Exception:
        return None


def fetch_pump_history(pump_id: str):
    try:
        r = requests.get(f"{BACKEND_URL}/pumps/{pump_id}/history", timeout=4.0)
        if r.status_code == 200:
            return r.json()
        return None
    except Exception:
        return None


def api_evaluate_audio(pump_id: str, audio_bytes: bytes, filename: str):
    files = {"file": (filename, audio_bytes, "audio/wav")}
    data = {"pump_id": pump_id}
    r = requests.post(f"{BACKEND_URL}/evaluate/audio", files=files, data=data, timeout=30.0)
    return r.status_code, r.json()


def api_start_repair(pump_id: str, notes: str, tech_id: str):
    payload = {"pump_id": pump_id, "notes": notes, "technician_id": tech_id}
    r = requests.post(f"{BACKEND_URL}/repair/start", json=payload, timeout=5.0)
    return r.status_code, r.json()


def api_complete_repair(pump_id: str, notes: str):
    payload = {"pump_id": pump_id, "notes": notes}
    r = requests.post(f"{BACKEND_URL}/repair/complete", json=payload, timeout=5.0)
    return r.status_code, r.json()


def api_verify_audio(pump_id: str, audio_bytes: bytes, filename: str):
    files = {"file": (filename, audio_bytes, "audio/wav")}
    data = {"pump_id": pump_id}
    r = requests.post(f"{BACKEND_URL}/verify/audio", files=files, data=data, timeout=30.0)
    return r.status_code, r.json()


def api_reset_diagnosis(pump_id: str):
    r = requests.post(f"{BACKEND_URL}/pumps/{pump_id}/reset-diagnosis", timeout=5.0)
    return r.status_code, r.json()


# ============================================================
# STATE COLORS & BADGES
# ============================================================

STATE_STYLES = {
    "HEALTHY": {"color": "#34d399", "bg": "rgba(52, 211, 153, 0.15)", "label": "HEALTHY", "icon": "🟢"},
    "DIAGNOSIS_PENDING": {"color": "#60a5fa", "bg": "rgba(96, 165, 250, 0.15)", "label": "DIAGNOSIS PENDING", "icon": "🔵"},
    "MAINTENANCE_REQUIRED": {"color": "#fb923c", "bg": "rgba(251, 146, 60, 0.15)", "label": "MAINTENANCE REQUIRED", "icon": "🟠"},
    "REPAIR_IN_PROGRESS": {"color": "#818cf8", "bg": "rgba(129, 140, 248, 0.15)", "label": "REPAIR IN PROGRESS", "icon": "🟣"},
    "VERIFICATION_PENDING": {"color": "#f472b6", "bg": "rgba(244, 114, 182, 0.15)", "label": "VERIFICATION PENDING", "icon": "🌸"},
    "ESCALATED": {"color": "#f87171", "bg": "rgba(248, 113, 113, 0.15)", "label": "ESCALATED / REVIEW", "icon": "🔴"},
}


def render_state_badge(state: str):
    info = STATE_STYLES.get(state, {"color": "#94a3b8", "bg": "rgba(148, 163, 184, 0.15)", "label": state, "icon": "⚪"})
    return f"""
    <span style="
        background-color: {info['bg']};
        color: {info['color']};
        border: 1px solid {info['color']};
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 13px;
        letter-spacing: 0.5px;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    ">
        {info['icon']} {info['label']}
    </span>
    """


# ============================================================
# SIDEBAR: PUMP SELECTION & BACKEND HEALTH
# ============================================================

backend_online, backend_data = check_backend_health()

with st.sidebar:
    st.markdown("### 💧 AquaFusion Operations")
    
    if backend_online:
        st.markdown(
            '<div style="color:#34d399;font-size:12px;font-weight:600;margin-bottom:12px">● Backend Service Online (Port 8000)</div>',
            unsafe_allow_html=True
        )
    else:
        st.error(f"Backend Offline at {BACKEND_URL}\nStart with: python -m uvicorn api:app --port 8000")

    st.divider()

    # Fetch pumps from backend
    registered_pumps = fetch_all_pumps()
    pump_ids = [p["pump_id"] for p in registered_pumps] if registered_pumps else ["PUMP-001", "PUMP-002", "PUMP-003"]

    selected_pump_id = st.selectbox("Select Target Handpump", pump_ids, index=0)
    
    # Custom pump ID option
    allow_custom = st.checkbox("Enter custom pump ID")
    if allow_custom:
        selected_pump_id = st.text_input("Custom Pump ID", value="PUMP-001").strip().upper()

    # Query authoritative state for selected pump
    pump_info = fetch_pump_state(selected_pump_id)
    current_state = pump_info["status"] if pump_info else "DIAGNOSIS_PENDING"

    st.markdown("#### Hardware Metadata")
    if pump_info:
        st.markdown(f"**Status:** {render_state_badge(current_state)}", unsafe_allow_html=True)
        st.markdown(f"**Location:** `{pump_info.get('location_info', 'N/A')}`")
        st.markdown(f"**Age:** `{pump_info.get('age_years', 'N/A')} years`")
        st.markdown(f"**Last Service:** `{pump_info.get('last_maintenance_date', 'N/A')}`")
        st.markdown(f"**Known Issues:** `{pump_info.get('known_issues', 'None')}`")
    else:
        st.info("Pump record not yet registered in SQLite.")

    st.divider()
    st.caption("Emerald Boys Architecture\nAudio → ML → Agent → Deterministic Validator → SQLite Database → Frontend")


# ============================================================
# MAIN APPLICATION INTERFACE
# ============================================================

st.markdown(
    f"""
    <div style="background:rgba(10, 18, 32, 0.85);border:1px solid rgba(110, 200, 255, 0.15);border-radius:12px;padding:20px;margin-bottom:20px">
        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px">
            <div>
                <span style="font-size:11px;color:#22d3ee;font-weight:700;letter-spacing:1px">OPERATIONAL DASHBOARD</span>
                <h1 style="margin:4px 0 0 0;font-size:28px;font-family:'Space Grotesk',sans-serif">Handpump: {selected_pump_id}</h1>
                <div style="font-size:13px;color:#94a3b8;margin-top:4px">Authoritative Database Lifecycle Control</div>
            </div>
            <div>
                {render_state_badge(current_state)}
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Workflow Stepper Progression Bar
step_labels = ["1. Acoustic Ingestion", "2. ML Inference", "3. Agent Reasoning", "4. Deterministic Safety", "5. Repair / Verify"]
step_active = {
    "DIAGNOSIS_PENDING": 0,
    "MAINTENANCE_REQUIRED": 3,
    "REPAIR_IN_PROGRESS": 4,
    "VERIFICATION_PENDING": 4,
    "HEALTHY": 4,
    "ESCALATED": 3
}.get(current_state, 0)

cols = st.columns(len(step_labels))
for idx, (c, lbl) in enumerate(zip(cols, step_labels)):
    with c:
        bg = "#22d3ee" if idx <= step_active else "#334155"
        txt_col = "#030712" if idx <= step_active else "#94a3b8"
        st.markdown(
            f"""
            <div style="background:{bg};color:{txt_col};padding:8px 10px;border-radius:8px;font-size:11px;font-weight:700;text-align:center">
                {lbl}
            </div>
            """,
            unsafe_allow_html=True
        )

st.write("")

# Navigation Tabs
tab_ops, tab_report, tab_history, tab_escalations = st.tabs([
    "⚡ Active Operation",
    "📋 Diagnostic Report",
    "📜 Database Audit History",
    "🚨 Escalation Center"
])


# ============================================================
# TAB 1: ACTIVE OPERATION (STATE-DRIVEN CONTROLS)
# ============================================================

with tab_ops:

    # --------------------------------------------------------
    # STATE: DIAGNOSIS_PENDING
    # --------------------------------------------------------
    if current_state == "DIAGNOSIS_PENDING":
        st.markdown("### Step 1: Acoustic Diagnosis Evaluation")
        st.info(f"Handpump **{selected_pump_id}** is awaiting acoustic telemetry evaluation.")

        c1, c2 = st.columns([1.2, 1])
        with c1:
            audio_source = st.radio(
                "Audio Input Source:",
                ["Use Pre-Recorded Field Acoustic Sample", "Upload Custom WAV Audio File"],
                horizontal=True
            )

            audio_bytes = None
            audio_name = "sample.wav"

            if audio_source == "Use Pre-Recorded Field Acoustic Sample":
                sample_options = {
                    "Seal Leak Anomaly (High Frequency Turbulence)": "samples/audio/seal_leak_anomaly.wav",
                    "Bearing Friction Chatter (Mechanical Degradation)": "samples/audio/bearing_friction_high.wav",
                    "Normal Pump Rhythm (Clean Harmonic Baseline)": "samples/audio/normal_pump_sample.wav",
                    "Low Confidence Chatter (Inconclusive Signal)": "samples/audio/low_confidence_chatter.wav",
                }
                sample_choice = st.selectbox("Select Benchmark Acoustic Sample", list(sample_options.keys()))
                sample_file_path = sample_options[sample_choice]
                audio_name = Path(sample_file_path).name

                if Path(sample_file_path).exists():
                    with open(sample_file_path, "rb") as f:
                        audio_bytes = f.read()
                    st.audio(audio_bytes, format="audio/wav")
                else:
                    st.warning(f"File not found: {sample_file_path}")

            else:
                uploaded_file = st.file_uploader("Upload Acoustic Recording", type=["wav", "mp3", "ogg", "flac"])
                if uploaded_file is not None:
                    audio_bytes = uploaded_file.getvalue()
                    audio_name = uploaded_file.name
                    st.audio(audio_bytes, format="audio/wav")

            if st.button("⚡ Run AI Acoustic Diagnosis Pipeline", type="primary", disabled=audio_bytes is None):
                with st.spinner("Processing audio, extracting features, and invoking Agent..."):
                    status_code, response_data = api_evaluate_audio(selected_pump_id, audio_bytes, audio_name)
                    if status_code == 200:
                        st.session_state["last_eval"] = response_data
                        st.success(f"Evaluation completed: State transitioned to {response_data['final_state']}")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(f"Evaluation error ({status_code}): {response_data.get('detail', response_data)}")

        with c2:
            st.markdown(
                """
                <div style="background:rgba(15, 23, 42, 0.8);border:1px solid rgba(110, 200, 255, 0.15);border-radius:10px;padding:16px">
                    <div style="font-size:12px;font-weight:700;color:#22d3ee;margin-bottom:8px">REAL SEQUENTIAL PIPELINE</div>
                    <ul style="font-size:12px;color:#cbd5e1;line-height:1.8;padding-left:18px;margin:0">
                        <li><b>1. Real Audio Waveform:</b> Decoded directly via SoundFile/Librosa.</li>
                        <li><b>2. Feature Extraction:</b> Computes RMS, ZCR, Spectral Centroid, Bandwidth, Rolloff, and 13 MFCCs.</li>
                        <li><b>3. ML Inference:</b> Generates MLPayload (Prediction, Probability Score).</li>
                        <li><b>4. AI Agent:</b> Queries hardware context and proposes next state.</li>
                        <li><b>5. Safety Kernel:</b> Enforces state machine validation.</li>
                        <li><b>6. SQLite:</b> Authoritative persistence of cases and transitions.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True
            )

    # --------------------------------------------------------
    # STATE: MAINTENANCE_REQUIRED
    # --------------------------------------------------------
    elif current_state == "MAINTENANCE_REQUIRED":
        st.markdown(
            """
            <div style="background:rgba(251, 146, 60, 0.12);border:1px solid #fb923c;border-radius:10px;padding:18px;margin-bottom:16px">
                <h3 style="color:#fb923c;margin:0 0 6px 0">⚠️ Maintenance Required</h3>
                <div style="font-size:13px;color:#fed7aa">Acoustic anomaly detected. A physical maintenance inspection is required before the pump can be returned to service.</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        c1, c2 = st.columns(2)
        with c1:
            tech_id = st.text_input("Technician Identifier", value="TECH-NBO-01")
            repair_notes = st.text_area("Initial Repair Assessment & Dispatched Parts", value="Dispatched mechanic to inspect pump cylinder seals and rod assembly.")
            
            if st.button("🔧 Start Repair Procedure", type="primary"):
                with st.spinner("Recording state transition in SQLite..."):
                    status_code, resp = api_start_repair(selected_pump_id, repair_notes, tech_id)
                    if status_code == 200:
                        st.success("Repair officially initiated! Pump status -> REPAIR_IN_PROGRESS")
                        time.sleep(0.8)
                        st.rerun()
                    else:
                        st.error(f"Failed to start repair: {resp.get('detail', resp)}")

        with c2:
            st.info("State Machine Rule: Only pumps in 'MAINTENANCE_REQUIRED' state can transition to 'REPAIR_IN_PROGRESS'. The backend enforces this strictly.")

    # --------------------------------------------------------
    # STATE: REPAIR_IN_PROGRESS
    # --------------------------------------------------------
    elif current_state == "REPAIR_IN_PROGRESS":
        st.markdown(
            """
            <div style="background:rgba(129, 140, 248, 0.12);border:1px solid #818cf8;border-radius:10px;padding:18px;margin-bottom:16px">
                <h3 style="color:#818cf8;margin:0 0 6px 0">🛠️ Physical Repair In Progress</h3>
                <div style="font-size:13px;color:#c7d2fe">Technician is performing component replacement. Once work is finished, mark complete to initiate acoustic verification.</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        completion_notes = st.text_area(
            "Technician Service Completion Notes",
            value="Replaced worn leather piston cups, tightened pump head bolts, and lubricated handle pivot."
        )

        if st.button("✅ Mark Repair Complete (Awaiting Verification)", type="primary"):
            with st.spinner("Updating database status to VERIFICATION_PENDING..."):
                status_code, resp = api_complete_repair(selected_pump_id, completion_notes)
                if status_code == 200:
                    st.success("Repair marked complete! Ready for post-repair acoustic verification.")
                    time.sleep(0.8)
                    st.rerun()
                else:
                    st.error(f"Error: {resp.get('detail', resp)}")

    # --------------------------------------------------------
    # STATE: VERIFICATION_PENDING
    # --------------------------------------------------------
    elif current_state == "VERIFICATION_PENDING":
        st.markdown(
            """
            <div style="background:rgba(244, 114, 182, 0.12);border:1px solid #f472b6;border-radius:10px;padding:18px;margin-bottom:16px">
                <h3 style="color:#f472b6;margin:0 0 6px 0">🎧 Post-Repair Acoustic Verification</h3>
                <div style="font-size:13px;color:#fbcfe8">Upload post-repair audio to verify the physical repair succeeded. The ML model must confirm normal pump rhythm before returning to HEALTHY.</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        v_source = st.radio("Verification Audio Source:", ["Use Benchmark Sample", "Upload Custom WAV"], horizontal=True)
        v_bytes = None
        v_name = "verification.wav"

        if v_source == "Use Benchmark Sample":
            v_samples = {
                "Healthy Rhythm Sample (Repaired Pump Restored)": "samples/audio/normal_pump_sample.wav",
                "Persistent Anomaly Sample (Repair Unsuccessful)": "samples/audio/seal_leak_anomaly.wav",
                "Ambiguous Chatter (Uncertainty Test)": "samples/audio/low_confidence_chatter.wav"
            }
            v_choice = st.selectbox("Select Verification Audio Sample", list(v_samples.keys()))
            v_path = v_samples[v_choice]
            v_name = Path(v_path).name
            if Path(v_path).exists():
                with open(v_path, "rb") as f:
                    v_bytes = f.read()
                st.audio(v_bytes, format="audio/wav")
        else:
            v_upload = st.file_uploader("Upload Post-Repair Audio Recording", type=["wav", "mp3", "ogg", "flac"])
            if v_upload:
                v_bytes = v_upload.getvalue()
                v_name = v_upload.name
                st.audio(v_bytes, format="audio/wav")

        if st.button("🔍 Run Post-Repair Verification Analysis", type="primary", disabled=v_bytes is None):
            with st.spinner("Processing verification audio through ML and Agent..."):
                status_code, v_resp = api_verify_audio(selected_pump_id, v_bytes, v_name)
                if status_code == 200:
                    st.session_state["last_eval"] = v_resp
                    st.success(f"Verification completed: Pump state is now {v_resp['final_state']}")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(f"Verification error: {v_resp.get('detail', v_resp)}")

    # --------------------------------------------------------
    # STATE: HEALTHY
    # --------------------------------------------------------
    elif current_state == "HEALTHY":
        st.markdown(
            """
            <div style="background:rgba(52, 211, 153, 0.12);border:1px solid #34d399;border-radius:10px;padding:24px;text-align:center;margin-bottom:20px">
                <h2 style="color:#34d399;margin:0 0 8px 0">🟢 Pump Is Operating Normally</h2>
                <div style="font-size:14px;color:#a7f3d0">All acoustic features and harmonic frequencies conform to healthy baseline standards.</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.info("To perform a new routine diagnostic test on this pump, initiate a new cycle below.")
        if st.button("🔄 Initiate New Diagnostic Evaluation Cycle", type="primary"):
            with st.spinner("Resetting pump to DIAGNOSIS_PENDING..."):
                status_code, resp = api_reset_diagnosis(selected_pump_id)
                if status_code == 200:
                    st.success("Pump transitioned to DIAGNOSIS_PENDING.")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error(f"Error resetting pump: {resp.get('detail', resp)}")

    # --------------------------------------------------------
    # STATE: ESCALATED
    # --------------------------------------------------------
    elif current_state == "ESCALATED":
        st.markdown(
            """
            <div style="background:rgba(248, 113, 113, 0.12);border:1px solid #f87171;border-radius:10px;padding:20px;margin-bottom:20px">
                <h3 style="color:#f87171;margin:0 0 6px 0">🚨 Case Escalated — Human Review Required</h3>
                <div style="font-size:13px;color:#fecaca">
                    Deterministic safety kernel triggered an escalation. Automatic state changes are halted until a senior engineer reviews the evidence.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Escalation Actions")
            if st.button("🛠️ Acknowledge Review & Return to Healthy", type="primary"):
                api_reset_diagnosis(selected_pump_id)
                st.rerun()
        with c2:
            st.markdown("#### Diagnostic Uncertainty Protocol")
            st.caption("Low confidence acoustic evidence (<0.40) or illegal agent proposal overrides trigger this safety state.")


# ============================================================
# TAB 2: STRUCTURED DIAGNOSTIC REPORT
# ============================================================

with tab_report:
    st.markdown("### Structured Diagnostic Report")
    
    last_eval = st.session_state.get("last_eval")
    history = fetch_pump_history(selected_pump_id)
    
    report_data = None
    if last_eval and last_eval.get("diagnostic_report"):
        report_data = last_eval.get("diagnostic_report")
        rep_pred = last_eval.get("ml_prediction")
        rep_conf = last_eval.get("ml_confidence")
        rep_state = last_eval.get("final_state")
        rep_over = last_eval.get("was_overridden", False)
    elif history and history.get("cases"):
        latest_case = history["cases"][0]
        rep_pred = latest_case.get("ml_prediction")
        rep_conf = latest_case.get("ml_confidence")
        rep_state = latest_case.get("final_state")
        rep_over = latest_case.get("was_overridden", False)

    if report_data:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("ML Prediction", rep_pred)
        m2.metric("ML Confidence", f"{rep_conf*100:.1f}%" if rep_conf else "N/A")
        m3.metric("Final Validated State", rep_state)
        m4.metric("Deterministic Override", "YES" if rep_over else "NO")

        if rep_over:
            st.warning("⚠️ **Deterministic Safety Override Applied**: The AI agent proposal was rejected by the backend state machine to protect physical infrastructure.")

        st.markdown(
            f"""
            <div style="display:grid;gap:12px;margin-top:14px">
                <div style="background:rgba(15, 23, 42, 0.7);padding:14px;border-radius:8px;border-left:4px solid #22d3ee">
                    <b style="color:#22d3ee">1. Acoustic Observation:</b><br>
                    <span style="color:#e2e8f0">{report_data.get('observation', 'N/A')}</span>
                </div>
                <div style="background:rgba(15, 23, 42, 0.7);padding:14px;border-radius:8px;border-left:4px solid #818cf8">
                    <b style="color:#818cf8">2. Mechanical Failure Mode Interpretation:</b><br>
                    <span style="color:#e2e8f0">{report_data.get('interpretation', 'N/A')}</span>
                </div>
                <div style="background:rgba(15, 23, 42, 0.7);padding:14px;border-radius:8px;border-left:4px solid #fbbf24">
                    <b style="color:#fbbf24">3. Recommended Physical Maintenance Checklist:</b><br>
                    <span style="color:#e2e8f0">{report_data.get('recommendation', 'N/A')}</span>
                </div>
                <div style="background:rgba(15, 23, 42, 0.7);padding:14px;border-radius:8px;border-left:4px solid #34d399">
                    <b style="color:#34d399">4. Upstream Confidence Assessment:</b><br>
                    <span style="color:#e2e8f0">{report_data.get('confidence_assessment', 'N/A')}</span>
                </div>
                <div style="background:rgba(15, 23, 42, 0.7);padding:14px;border-radius:8px;border-left:4px solid #f472b6">
                    <b style="color:#f472b6">5. State Machine Next Action:</b><br>
                    <span style="color:#e2e8f0">{report_data.get('next_action', 'N/A')}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.info("No active diagnostic report for this pump yet. Run an acoustic diagnosis from the 'Active Operation' tab.")


# ============================================================
# TAB 3: DATABASE AUDIT HISTORY
# ============================================================

with tab_history:
    st.markdown(f"### Persistent SQLite Audit Trail for {selected_pump_id}")
    
    history_data = fetch_pump_history(selected_pump_id)
    if history_data:
        st.markdown(f"**Total Diagnostic Evaluations:** `{len(history_data.get('cases', []))}` | **State Transitions:** `{len(history_data.get('transitions', []))}`")

        st.markdown("#### State Transition Log")
        transitions = history_data.get("transitions", [])
        if transitions:
            t_rows = []
            for t in transitions:
                t_rows.append({
                    "Timestamp": t.get("created_at", "N/A"),
                    "From State": t.get("from_state"),
                    "Proposed State": t.get("proposed_state"),
                    "Final State": t.get("final_state"),
                    "Action": t.get("action"),
                    "Overridden": "YES" if t.get("was_overridden") else "NO",
                    "Reason / Notes": t.get("reason", "")
                })
            st.dataframe(t_rows, use_container_width=True)
        else:
            st.info("No state transitions recorded yet.")

        st.markdown("#### Diagnostic Case Evaluations")
        cases = history_data.get("cases", [])
        if cases:
            c_rows = []
            for c in cases:
                c_rows.append({
                    "Case ID": c.get("case_id"),
                    "Timestamp": c.get("created_at", "N/A"),
                    "ML Prediction": c.get("ml_prediction"),
                    "Confidence": f"{c.get('ml_confidence', 0)*100:.1f}%",
                    "Flag": c.get("confidence_flag"),
                    "Proposed State": c.get("proposed_state"),
                    "Final State": c.get("final_state"),
                    "Explanation": c.get("explanation")
                })
            st.dataframe(c_rows, use_container_width=True)
        else:
            st.info("No diagnostic cases recorded yet.")

    else:
        st.warning("Failed to fetch historical records from SQLite backend.")


# ============================================================
# TAB 4: ESCALATION CENTER
# ============================================================

with tab_escalations:
    st.markdown("### Escalation Events & Webhook Dispatch Log")
    
    if history_data:
        escalations = history_data.get("escalation_events", [])
        if escalations:
            st.warning(f"Recorded {len(escalations)} escalation events for {selected_pump_id}:")
            for e in escalations:
                st.markdown(
                    f"""
                    <div style="background:rgba(239, 68, 68, 0.1);border:1px solid rgba(239, 68, 68, 0.4);padding:14px;border-radius:8px;margin-bottom:10px">
                        <div style="display:flex;justify-content:space-between;font-size:12px;font-weight:700">
                            <span style="color:#f87171">EVENT ID: {e.get('event_id')}</span>
                            <span style="color:#94a3b8">{e.get('timestamp', 'N/A')}</span>
                        </div>
                        <div style="margin-top:6px;font-size:13px;color:#fecaca"><b>Reason:</b> {e.get('reason')}</div>
                        <div style="margin-top:4px;font-size:11px;color:#94a3b8">Case ID: {e.get('case_id', 'N/A')} · Webhook Status: {e.get('webhook_status', 'DISABLED')}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.success("No active escalation events for this handpump.")
    else:
        st.info("Connect to the backend to view escalation events.")

# Emerald Boys — Rural Handpump AI Diagnostic & Maintenance System

A production-grade, local, neuro-symbolic diagnostic engine, FastAPI backend, and Streamlit frontend designed to evaluate rural handpump acoustic anomaly reports, maintain persistent operational history in SQLite, and guide technicians through deterministic lifecycle states with full safety verification.

---

## 1. System Architecture & Sequential Lifecycle

In rural communities across the developing world, handpumps represent the primary source of clean drinking water. Traditional maintenance is reactive, causing prolonged outages. The **Emerald Boys** system transforms acoustic diagnostic recordings into deterministic, actionable, and safe maintenance workflows:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Sequential System Workflow                        │
└─────────────────────────────────────────────────────────────────────────────┘

  [Technician / Field Sensor]
               │
               ▼
  ┌─────────────────────────┐
  │   Acoustic Audio (.wav) │
  └────────────┬────────────┘
               │
               ▼
  ┌─────────────────────────┐
  │   Audio Preprocessing   │ ◄─── audio_processing.py (librosa/soundfile)
  │ (RMS, ZCR, Centroid,    │      Validates length, sample rate, channels,
  │  Bandwidth, MFCCs)      │      and checks for corruption or silence.
  └────────────┬────────────┘
               │
               ▼
  ┌─────────────────────────┐
  │   ML Inference Adapter  │ ◄─── ml_adapter.py (MOCK ML prototype baseline)
  │ (Acoustic Feature Class)│      Outputs canonical MLPayload
  └────────────┬────────────┘
               │
               ▼
  ┌─────────────────────────┐
  │  Emerald Boys AI Agent  │ ◄─── agent.py (Safe LLM / Deterministic Prototype)
  │ (Reasoning & Proposal)  │      Queries SQLite tools: get_pump_context,
  └────────────┬────────────┘      get_pump_history. Proposes candidate state.
               │
               ▼
  ┌─────────────────────────┐
  │ Deterministic Validator │ ◄─── Deterministic Safety Kernel
  │ (Python Safety Kernel)  │      Authoritative state machine rules. Overrides
  └────────────┬────────────┘      illegal proposals to ESCALATED with audit log.
               │
               ▼
  ┌─────────────────────────┐
  │ SQLite Local Database   │ ◄─── database.py & repositories.py (data/emerald_boys.db)
  │ (Authoritative State)   │      Persists pumps, diagnostic_cases, state_transitions.
  └────────────┬────────────┘
               │
               ▼
  ┌─────────────────────────┐
  │  Streamlit Web Console  │ ◄─── app.py (Directly synchronizes with FastAPI / DB)
  │ (Technician Interface)  │      Enforces repair flow: Start -> Complete -> Verify
  └────────────┬────────────┘
               │
               ▼
  ┌─────────────────────────┐
  │ Post-Repair Verification│ ◄─── Follow-up acoustic check -> ML -> Agent -> Final State
  └─────────────────────────┘
```

### Core Architecture Invariants
1. **The Backend & Database are Authoritative:** The Streamlit frontend is a thin, reactive interface. It never mutates state locally; all actions invoke FastAPI endpoints backed by SQLite.
2. **The LLM Proposes, the Safety Kernel Decides:** The AI Agent is strictly a diagnostic reasoning and candidate proposal layer. It cannot bypass state machine transition rules.
3. **Deterministic Safety Override:** If an agent or client proposes an illegal transition (e.g., `HEALTHY -> MAINTENANCE_REQUIRED` without diagnosis, or jumping directly to `HEALTHY`), the backend safety kernel intercepts the action:
   - Preserves `proposed_state` for transparency and auditing.
   - Sets `final_state = "ESCALATED"`.
   - Sets `action = "ESCALATE"`, `needs_human_review = True`, and `was_overridden = True`.
   - Appends `[OVERRIDE]` to `audit_notes`.
4. **Preserved ML Confidence:** Upstream ML confidence scores cannot be altered, masked, or inflated by the agent or backend.
5. **Completely Local & Self-Contained:** Runs on standard Windows/Linux/macOS machines using SQLite and an offline deterministic agent mode when no API keys are configured.

---

## 2. Canonical State Machine

The system models the physical handpump maintenance lifecycle through 6 canonical states:

```
HEALTHY
  │
  ▼
DIAGNOSIS_PENDING
  ├───────────────────┬───────────────────┐
  ▼                   ▼                   ▼
HEALTHY      MAINTENANCE_REQUIRED     ESCALATED
                      │                   ▲
                      ▼                   │
              REPAIR_IN_PROGRESS          │
                      │                   │
                      ▼                   │
              VERIFICATION_PENDING ───────┘
                      ├──> HEALTHY
                      └──> MAINTENANCE_REQUIRED
```

### Transition Matrix

| Current State (`from_state`) | Permitted Next States (`to_state`) | Action Required |
| :--- | :--- | :--- |
| `HEALTHY` | `DIAGNOSIS_PENDING` | Anomaly reported or routine inspection |
| `DIAGNOSIS_PENDING` | `MAINTENANCE_REQUIRED`, `ESCALATED`, `HEALTHY` | Agent evaluation + deterministic validation |
| `MAINTENANCE_REQUIRED` | `REPAIR_IN_PROGRESS`, `ESCALATED` | Technician dispatches and starts repair |
| `REPAIR_IN_PROGRESS` | `VERIFICATION_PENDING`, `ESCALATED` | Technician marks mechanical work completed |
| `VERIFICATION_PENDING` | `HEALTHY`, `MAINTENANCE_REQUIRED`, `ESCALATED` | Post-repair acoustic verification test |
| `ESCALATED` | `HEALTHY`, `DIAGNOSIS_PENDING` | Human engineer manual review or re-inspection |

---

## 3. Audio Processing Pipeline (`audio_processing.py`)

A canonical audio extraction pipeline handles raw audio files uploaded through the web console or REST API:
* **Audio Loading:** Uses `soundfile` and `librosa` with automatic resampling to 22,050 Hz and mono downmixing.
* **Format Support:** Supports standard 16-bit PCM WAV, MP3, and FLAC files.
* **Validation & Error Handling:** Throws structured `AudioProcessingError` when:
  - File is empty or zero bytes.
  - File is corrupt or has invalid header markers.
  - Duration is below minimum length threshold (< 0.2s).
  - Signal is entirely pure silence.
* **Feature Extraction:**
  - Root Mean Square (RMS) energy.
  - Zero Crossing Rate (ZCR).
  - Spectral Centroid, Bandwidth, and Rolloff.
  - 13 Mel-Frequency Cepstral Coefficients (MFCCs).

---

## 4. ML Adapter & Engineering Transparency (`ml_adapter.py`)

> **Engineering Notice:**
> The current ML layer operates in **`MOCK ML`** adapter mode (`ADAPTER_MODE = "MOCK ML"`). It analyzes extracted acoustic features and deterministic profile signatures to produce canonical `MLPayload` objects. It serves as a transparent software engineering baseline and architectural contract, not a field-trained neural network.

### Acoustic Profiles:
* **Normal Operation (`normal` in filename / low spectral centroid):** `NORMAL`, confidence $0.88$ (`HIGH`).
* **High Bearing Friction (`bearing`, `friction` / high RMS + high centroid):** `ABNORMAL`, confidence $0.92$ (`HIGH`).
* **Seal Leak Anomaly (`seal`, `fault`, `leak`):** `ABNORMAL`, confidence $0.85$ (`HIGH`).
* **Chatter / Low Confidence (`low_confidence`, `chatter`):** `ABNORMAL`, confidence $0.35$ (`LOW` $\rightarrow$ triggers uncertainty escalation).

---

## 5. Neuro-Symbolic Agent (`agent.py`)

The agent supports two operational modes seamlessly:
1. **Live LLM Mode:** Configured with `OPENAI_API_KEY` (compatible with OpenAI or OpenRouter). Uses OpenAI Function Calling to dynamically inspect pump hardware specs and past maintenance history before proposing actions.
2. **Deterministic Prototype Mode:** Active by default when no API key is provided, or when `AGENT_MODE=deterministic`. Operates fully offline with zero external network dependencies, ensuring 100% deterministic and reproducible test suites and local demos.

### Sandboxed Database Tools (`tools.py`)
* `get_pump_context(pump_id)`: Fetches pump specifications, age, last maintenance date, and installation location from SQLite.
* `get_pump_history(pump_id)`: Retrieves historical diagnostic cases, past state transitions, and previous failure modes.

---

## 6. Local Database Schema (`database.py` & `repositories.py`)

Data is persisted locally in `data/emerald_boys.db` using SQLAlchemy 2.x:

1. **`pumps` Table:**
   - `pump_id` (Primary Key, e.g. `PUMP-001`)
   - `status`: Current lifecycle state
   - `location_info`, `age_years`, `last_maintenance_date`, `previous_failures`, `known_issues`
   - `created_at`, `updated_at`

2. **`diagnostic_cases` Table:**
   - `case_id` (Primary Key, e.g. `CASE-101`)
   - `pump_id`: Reference to pump
   - `ml_prediction`, `ml_confidence`, `confidence_flag`
   - `input_state`, `proposed_state`, `final_state`, `action`
   - `needs_human_review`, `was_overridden`, `explanation`, `audit_notes`
   - `created_at`, `updated_at`

3. **`state_transitions` Table:**
   - `id`: Auto-incrementing primary key
   - `case_id`, `pump_id`
   - `from_state`, `proposed_state`, `final_state`, `action`
   - `reason`, `was_overridden`, `created_at`

---

## 7. REST API Reference (`api.py`)

FastAPI backend with OpenAPI documentation available at `http://127.0.0.1:8000/docs`.

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Root service identification and version |
| `GET` | `/health` | Service health status |
| `GET` | `/pumps` | List all pumps and their current authoritative status |
| `GET` | `/pumps/{pump_id}` | Retrieve real-time status and context for a specific pump |
| `GET` | `/pumps/{pump_id}/history` | Retrieve full historical audit trail of cases and transitions |
| `POST` | `/evaluate` | Evaluate diagnostic case via JSON `MLPayload` |
| `POST` | `/evaluate/audio` | Upload raw audio file (`.wav`, `.mp3`) for end-to-end evaluation |
| `POST` | `/repair/start` | Transition pump from `MAINTENANCE_REQUIRED` to `REPAIR_IN_PROGRESS` |
| `POST` | `/repair/complete` | Transition pump from `REPAIR_IN_PROGRESS` to `VERIFICATION_PENDING` |
| `POST` | `/verify` | Post-repair verification via JSON payload |
| `POST` | `/verify/audio` | Post-repair verification via audio file upload |
| `POST` | `/pumps/{pump_id}/reset-diagnosis` | Set pump to `DIAGNOSIS_PENDING` for new diagnostic evaluation |

---

## 8. Streamlit Web Console (`app.py`)

The Streamlit technician console connects directly to the FastAPI backend:
* **Authoritative Dashboard:** Displays all pumps, their live hardware metadata, and current lifecycle badges.
* **Audio Diagnostic Suite:** Select preloaded audio samples or upload custom audio files. Visualizes audio waveforms and extracted acoustic features.
* **Interactive Technician Actions:**
  - **Start Repair:** Enabled only when the pump is in `MAINTENANCE_REQUIRED`.
  - **Complete Repair:** Enabled only when the pump is in `REPAIR_IN_PROGRESS`.
  - **Run Post-Repair Verification:** Enabled only when the pump is in `VERIFICATION_PENDING`.
* **Real-Time Audit Trail:** Inspect historical evaluations, transition timestamps, and safety override alerts.

---

## 9. Getting Started on Windows (VS Code)

### Prerequisites
* Windows 10/11
* Python 3.10+ (Tested on Python 3.13)
* VS Code with the Python extension installed

### Step 1: Set Up Virtual Environment
Open PowerShell in the project directory:
```powershell
# Create virtual environment (if not already created)
py -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Configure Environment
Copy `.env.example` to `.env`:
```powershell
Copy-Item .env.example .env
```
*(By default, `AGENT_MODE=deterministic` runs offline without needing an API key).*

### Step 3: Run with One Click in VS Code
The repository includes `.vscode/launch.json` and `.vscode/tasks.json`:
1. Press `F5` in VS Code and select **"FastAPI Backend"** to start the backend on `http://127.0.0.1:8000`.
2. Press `F5` and select **"Streamlit Frontend"** to launch the technician console in your browser at `http://localhost:8501`.
3. To run all tests, select **"Run All Tests (Pytest)"** from the debug dropdown or run task `Run All Pytest Tests` (`Ctrl+Shift+B`).

### Manual Terminal Commands
If running from separate PowerShell terminals:

**Terminal 1 — FastAPI Backend:**
```powershell
.\.venv\Scripts\Activate.ps1
uvicorn api:app --reload --port 8000
```

**Terminal 2 — Streamlit Frontend:**
```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

---

## 10. Running Automated Tests

The comprehensive test suite covers API endpoints, agent safety, database persistence, state transitions, audio processing, and the complete sequential repair lifecycle:

```powershell
# Run the entire test suite
.\.venv\Scripts\pytest.exe -v

# Run the end-to-end sequential workflow test specifically
.\.venv\Scripts\pytest.exe -v test_sequential_workflow.py
```

**Current Test Results:** `50 passed in ~2.7s` (100% pass rate).
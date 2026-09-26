# Emerald Boys — Rural Handpump AI Diagnostic & Maintenance Agent

A robust, local, neuro-symbolic diagnostic engine and FastAPI service designed to evaluate rural handpump acoustic anomaly reports, maintain persistent operational history in SQLite, and guide technicians through deterministic lifecycle states.

---

## 1. Project Purpose & Overview

In rural communities across the developing world, handpumps represent the primary source of clean drinking water. When a pump fails, communities face prolonged water crises. Traditional maintenance is reactive, relying on manual complaints days after a breakdown.

The **Emerald Boys** system transforms acoustic diagnostic data into actionable, safe maintenance workflows:

```text
               Audio Recording / Sensor Input
                             │
                             ▼
                 [ML Adapter (MOCK ML)]
                             │
                         MLPayload
                             │
                             ▼
             ┌───────────────────────────────┐
             │       Emerald Boys AI Agent   │
             │   (Reasoning & Proposal Layer)│
             └───────────────┬───────────────┘
                             │
                     Candidate Proposal
                             │
                             ▼
             ┌───────────────────────────────┐
             │ Deterministic State Machine   │
             │     (Python Safety Kernel)    │
             └───────────────┬───────────────┘
                             │
                   Validated Final State
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
     [Local SQLite Database]       [Escalation Event Engine]
  (Cases, Transitions, History)     (Local Log + Webhook)
```

### Core Architecture Invariant
* **AI Agent:** Interprets diagnostics, queries historical memory via controlled tools, and **proposes** the next logical maintenance action. The LLM is **never** the final source of truth.
* **Deterministic Backend:** Authoritatively validates every state transition. Illegal proposals are immediately rejected and overridden to `ESCALATED`.
* **Local Database:** Fully self-contained local SQLite database persists all evaluated cases, immutable state transitions, and current pump statuses.

---

## 2. Evolution: V1, V2, and V3

* **V1 — Core Neuro-Symbolic Agent:**
  - Pydantic V2 schemas (`MLPayload`, `LLMDecision`, `FinalAgentResponse`).
  - OpenAI / OpenRouter function calling interface with safe tool dispatcher.
  - Deterministic state machine transition matrix and confidence categorizer.
  - 6/6 unit tests passing.
* **V2 — FastAPI Service Layer:**
  - RESTful HTTP endpoints (`GET /`, `GET /health`, `POST /evaluate`).
  - Input validation, CORS configuration, and structured error isolation.
  - Interactive OpenAPI/Swagger documentation (`/docs`, `/redoc`).
  - 9/9 regression tests passing.
* **V3 — Complete Implementation (Persistent Memory & Full Workflow):**
  - **Step 1 (Local SQLite Foundation):** SQLAlchemy 2.x ORM models (`Pump`, `DiagnosticCase`, `StateTransition`) storing state locally in `data/emerald_boys.db`.
  - **Step 2 (FastAPI Database Integration):** Full end-to-end persistence in `POST /evaluate` ensuring validated states and override audits are saved.
  - **Step 3 (ML Boundary & Adapter):** `ml_adapter.py` providing `predict(audio_path)` with deterministic acoustic classification, clearly labeled as `MOCK ML`.
  - **Step 4 (Escalation Engine):** `events.py` generating structured escalation events upon validated `ESCALATED` states, supporting optional `ESCALATION_WEBHOOK_URL` with resilient failure handling.
  - **Step 5 (Historical Retrieval):** Controlled agent tool `get_pump_history` enabling the agent to reason over previous evaluations, past transitions, and failure patterns.
  - Full suite of 33 tests passing with 0 regressions.

---

## 3. Canonical State Machine

The system models the physical handpump maintenance lifecycle through 6 canonical states:

```text
HEALTHY
  ↓
DIAGNOSIS_PENDING
  ├───────────────┬───────────────────────────────┐
  ▼               ▼                               ▼
HEALTHY   MAINTENANCE_REQUIRED                ESCALATED
                  │                               ▲
                  ▼                               │
          REPAIR_IN_PROGRESS                      │
                  │                               │
                  ▼                               │
          VERIFICATION_PENDING ───────────────────┘
                  ├──> HEALTHY
                  └──> MAINTENANCE_REQUIRED
```

### Valid Transition Matrix

| Current State | Permitted Next States |
| :--- | :--- |
| `HEALTHY` | `DIAGNOSIS_PENDING` |
| `DIAGNOSIS_PENDING` | `MAINTENANCE_REQUIRED`, `ESCALATED`, `HEALTHY` |
| `MAINTENANCE_REQUIRED` | `REPAIR_IN_PROGRESS`, `ESCALATED` |
| `REPAIR_IN_PROGRESS` | `VERIFICATION_PENDING`, `ESCALATED` |
| `VERIFICATION_PENDING` | `HEALTHY`, `MAINTENANCE_REQUIRED`, `ESCALATED` |
| `ESCALATED` | `HEALTHY`, `DIAGNOSIS_PENDING` |

### Deterministic Safety Override Rule
If the agent proposes an illegal transition (e.g., attempting `HEALTHY` $\rightarrow$ `MAINTENANCE_REQUIRED` directly), the backend automatically:
1. Rejects the proposal.
2. Overrides the final state to `ESCALATED`.
3. Sets `action = "ESCALATE"` and `needs_human_review = True`.
4. Records `was_overridden = True` and appends `[OVERRIDE]` to `audit_notes`.

---

## 4. ML Adapter (`ml_adapter.py`)

The ML adapter forms a strict boundary between acoustic feature extraction and agent reasoning.

> **Transparent Engineering Notice:**
> The current adapter operates in **`MOCK ML`** mode for software prototyping and verification. It generates deterministic outputs conforming strictly to `MLPayload`. It does **not** claim to be a field-validated physical acoustic classifier.

### Clean Interface:
```python
from ml_adapter import predict

payload = predict("samples/audio/seal_leak_anomaly.wav", pump_id="PUMP-001")
```

### Deterministic Acoustic Profiles:
* **Normal Rhythm (`normal` in filename):** `NORMAL`, confidence $0.88$ (HIGH).
* **Seal Friction / Anomaly (`seal`, `friction`, `fault`):** `ABNORMAL`, confidence $0.92$ (HIGH).
* **Bearing Chatter (`bearing`):** `ABNORMAL`, confidence $0.78$ (HIGH/MEDIUM).
* **Valve Leak / Subtle Chatter (`leak`, `subtle`, `low_confidence`):** `ABNORMAL`, confidence $0.35$ (LOW).

---

## 5. Local Database Persistence (`database.py` & `repositories.py`)

All data is stored purely locally in SQLite at `data/emerald_boys.db` using SQLAlchemy 2.x. No cloud database or external server is required.

### Database Schema

1. **`pumps` Table:**
   - `id`: Primary key
   - `pump_id`: Unique identifier (e.g., `PUMP-001`)
   - `status`: Current lifecycle state
   - `location_info`: Geospatial/sector metadata
   - `created_at`, `updated_at`: Timestamps

2. **`diagnostic_cases` Table:**
   - `id`: Primary key
   - `case_id`: Unique diagnostic evaluation identifier (e.g., `CASE-101`)
   - `pump_id`: Foreign key referencing `pumps.pump_id`
   - `ml_prediction`, `ml_confidence`, `confidence_flag`
   - `input_state`, `proposed_state`, `final_state`, `action`
   - `needs_human_review`, `explanation`, `audit_notes`
   - `created_at`, `updated_at`

3. **`state_transitions` Table:**
   - `id`: Primary key
   - `case_id`: Associated evaluation case ID
   - `pump_id`: Foreign key referencing `pumps.pump_id`
   - `from_state`, `proposed_state`, `final_state`, `action`
   - `reason`: Explanation string
   - `was_overridden`: Boolean flag indicating if backend rejected agent proposal
   - `created_at`: Timestamp

---

## 6. Escalation System (`events.py`)

When the **final validated state** of an evaluation is `ESCALATED`, the system triggers an escalation event.

* **Trigger Condition:** Evaluated strictly against the Python-validated final state, **not** the raw LLM proposal.
* **Local Event Logging:** Events are formatted with unique `EVT-` IDs and logged locally.
* **Optional Webhook (`ESCALATION_WEBHOOK_URL`):**
  - If unset: Network calls are disabled; events are saved locally only.
  - If configured: Sends an HTTP POST with the JSON event payload (3.0s timeout).
  - **Fault Resilience:** Webhook network errors or timeouts are caught safely without breaking the diagnostic workflow or failing the API response.

---

## 7. Historical Context & Agent Memory (`tools.py`)

The agent accesses historical memory through the sandboxed `get_pump_history(pump_id)` tool:
* Safe retrieval via repository queries — the LLM never executes raw SQL.
* Returns total case counts, previous states, past anomaly occurrences, full case history, and escalation events.
* **Memory Invariant:** History provides context to explain recurring degradation patterns; it never supersedes the deterministic state machine rules.

---

## 8. API Reference

### Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Root service identification and version |
| `GET` | `/health` | Lightweight service health check |
| `POST` | `/evaluate` | Core evaluation endpoint: agent reasoning, validation, and SQLite persistence |
| `GET` | `/pumps/{pump_id}/history` | Historical cases, state transitions, and escalation logs for a pump |
| `GET` | `/docs` | Interactive Swagger UI documentation |
| `GET` | `/redoc` | Interactive ReDoc documentation |

### Sample Evaluation Request:
```bash
curl -X POST "http://127.0.0.1:8000/evaluate" \
  -H "Content-Type: application/json" \
  -d '{
    "case_id": "CASE-101",
    "pump_id": "PUMP-001",
    "current_state": "DIAGNOSIS_PENDING",
    "ml_prediction": "ABNORMAL",
    "ml_confidence": 0.92
  }'
```

---

## 9. Environment Variables

Create or configure `.env` (template in `.env.example`):

```ini
# OpenAI / OpenRouter LLM Configuration
OPENAI_API_KEY=YOUR_OPENROUTER_OR_OPENAI_KEY
OPENAI_BASE_URL=https://openrouter.ai/api/v1
OPENAI_MODEL=openai/gpt-4o-mini

# Deterministic Categorization Thresholds
ML_HIGH_CONFIDENCE_THRESHOLD=0.70
ML_LOW_CONFIDENCE_THRESHOLD=0.40

# Optional: Remote Escalation Webhook (Leave unset to disable external calls)
# ESCALATION_WEBHOOK_URL=https://webhook.site/your-webhook-endpoint

# Optional: CORS Origins
# CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
```

---

## 10. Installation & Running

### 1. Setup Virtual Environment
```powershell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Run the Full V3 Test Suite
```bash
pytest -q
```
*Current test suite: **33 passed** (6 unit tests, 9 API tests, 18 V3 tests).*

### 3. Run the V3 Laptop-Only Demonstration
```bash
python demo_v3.py
```
This demonstrates:
1. Ingesting prerecorded audio via `predict()`.
2. Producing validated `MLPayload` labeled `MOCK ML`.
3. Invoking the agent with dual tool access (`get_pump_context` & `get_pump_history`).
4. Enforcing deterministic state validation.
5. Persisting cases, transitions, and updated pump status in SQLite.
6. Generating structured escalation events upon failure/uncertainty.
7. Re-evaluating with historical case memory.

### 4. Start the FastAPI Server
```bash
python -m uvicorn api:app --reload --port 8000
```
Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) to test interactively.
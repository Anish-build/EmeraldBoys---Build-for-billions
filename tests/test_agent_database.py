from database.schema import create_tables
from database.queries import (
    create_inspection,
    save_diagnosis,
    create_repair,
    get_available_mechanics,
    log_event
)
from agent.decision import decide_action


create_tables()

pump_id = "P047"

# Simulated result coming from the ML/data teammate
prediction = "Abnormal"
confidence = 0.91
severity = "High"

# Save the inspection
inspection_id = create_inspection(
    pump_id,
    "demo_before_repair.wav"
)

print("Inspection created:", inspection_id)

# Save the ML diagnosis
diagnosis_id = save_diagnosis(
    inspection_id,
    prediction,
    confidence,
    severity
)

print("Diagnosis saved:", diagnosis_id)

# Ask the agent what to do
action = decide_action(
    prediction,
    confidence,
    severity
)

print("Agent decision:", action)

# If repair is needed, assign an available mechanic
if action == "DISPATCH_MECHANIC":

    mechanics = get_available_mechanics()

    if mechanics:

        mechanic = mechanics[0]
        mechanic_id = mechanic[0]

        repair_id = create_repair(
            pump_id,
            inspection_id,
            mechanic_id,
            "Inspect and repair pump mechanism",
            severity
        )

        print("Repair created:", repair_id)

        log_event(
            pump_id,
            inspection_id,
            "REPAIR_DISPATCHED",
            "Assigned mechanic",
            "High severity abnormal diagnosis"
        )

        print("Agent decision logged.")

    else:
        print("No mechanics available.")
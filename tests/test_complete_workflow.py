from database import create_tables
from database.queries import (
    create_pump,
    create_inspection,
    save_diagnosis,
    create_mechanic,
    create_repair,
    update_repair_status,
    save_verification,
    log_event,
    get_pump_status
)
from agent.decision import decide_action


# 1. Make sure database exists
create_tables()

pump_id = "P100"

# 2. Create pump
try:
    create_pump(
        pump_id,
        "Demo Village",
        "Hand Pump",
        "Village Committee"
    )
except Exception:
    # Pump may already exist from an earlier test
    pass


# 3. Create inspection
inspection_id = create_inspection(
    pump_id,
    "demo_audio.wav"
)

# 4. Simulated ML result
prediction = "Abnormal"
confidence = 0.91
severity = "High"


# 5. Save diagnosis
diagnosis_id = save_diagnosis(
    inspection_id,
    prediction,
    confidence,
    severity
)

# 6. Agent decision
action = decide_action(
    prediction,
    confidence,
    severity
)

print("Agent decision:", action)


# 7. Create mechanic
mechanic_id = create_mechanic(
    "Demo Mechanic",
    "Hand Pump Repair",
    "Demo Village"
)

# 8. Create repair
repair_id = create_repair(
    pump_id,
    inspection_id,
    mechanic_id,
    "Inspect and repair pump mechanism",
    severity
)

print("Repair created:", repair_id)


# 9. Complete repair
update_repair_status(
    repair_id,
    "Completed"
)

print("Repair completed.")


# 10. Save verification
verification_id = save_verification(
    repair_id,
    "Abnormal",
    "Normal",
    0.91,
    0.96,
    "Verified"
)

print("Verification saved:", verification_id)


# 11. Log event
log_event(
    pump_id,
    inspection_id,
    "REPAIR_VERIFIED",
    "Repair verified successfully",
    "Post-repair audio classified as Normal"
)

print("Audit event logged.")


# 12. Get final pump status
status = get_pump_status(pump_id)

print("\nFinal Pump Status:")
print("--------------------------------")
print(status)
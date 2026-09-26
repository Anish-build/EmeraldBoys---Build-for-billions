from database.schema import create_tables
from database.queries import log_event


# Make sure the database tables exist
create_tables()


# Log an agent decision
log_id = log_event(
    pump_id="P047",
    inspection_id="I-4d3930fe",
    event_type="REPAIR_DISPATCHED",
    action="Assigned mechanic",
    reason="High severity abnormal diagnosis"
)

print("Audit event created:")
print(log_id)
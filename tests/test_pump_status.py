from database.queries import get_pump_status


pump_id = "P047"

status = get_pump_status(pump_id)

print("Current Pump Status:")
print("--------------------------------")
print(status)
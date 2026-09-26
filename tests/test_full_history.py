from database.queries import get_pump_history


pump_id = "P047"

history = get_pump_history(pump_id)

print("Complete Pump History:")
print("--------------------------------")

for record in history:
    print(record)
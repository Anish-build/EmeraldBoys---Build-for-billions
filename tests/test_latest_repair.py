from database.queries import get_latest_repair


pump_id = "P047"

repair = get_latest_repair(pump_id)

print("Latest Repair:")
print("--------------------------------")
print(repair)   
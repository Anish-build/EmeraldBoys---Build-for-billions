from database.queries import get_latest_diagnosis


pump_id = "P047"

diagnosis = get_latest_diagnosis(pump_id)

print("Latest Diagnosis:")
print("--------------------------------")
print(diagnosis)
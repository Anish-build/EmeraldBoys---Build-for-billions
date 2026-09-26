from database.queries import get_latest_verification


pump_id = "P047"

verification = get_latest_verification(pump_id)

print("Latest Verification:")
print("--------------------------------")
print(verification)
from agent.decision import decide_action


print("Test 1:")
print(decide_action("Abnormal", 0.91, "High"))

print("\nTest 2:")
print(decide_action("Normal", 0.95, "Low"))

print("\nTest 3:")
print(decide_action("Abnormal", 0.45, "High"))

print("\nTest 4:")
print(decide_action("Abnormal", 0.85, "Medium"))
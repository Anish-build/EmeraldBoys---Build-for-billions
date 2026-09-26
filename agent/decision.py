def decide_action(prediction, confidence, severity):

    # Rule 1: Low confidence means the AI is not sufficiently certain
    if confidence < 0.60:
        return "HUMAN_REVIEW"

    # Rule 2: Normal pump does not need repair
    if prediction == "Normal":
        return "NO_REPAIR_NEEDED"

    # Rule 3: High-severity abnormal pump
    if prediction == "Abnormal" and severity == "High":
        return "DISPATCH_MECHANIC"

    # Rule 4: Medium/low severity abnormal pump
    if prediction == "Abnormal":
        return "HUMAN_REVIEW"

    # Safety fallback
    return "HUMAN_REVIEW"
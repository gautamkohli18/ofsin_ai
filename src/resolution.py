import pandas as pd

def suggest_resolution(row):
    category = row["predicted_category"]
    if category == "DUPLICATE_CHARGE":
        return "Auto-refund", "Duplicate charge → refund recommended"
    elif category == "FAILED_TRANSACTION":
        return "Escalate to bank", "Bank verification required"
    elif category == "FRAUD":
        return "Mark as potential fraud", "Suspicious activity flagged"
    elif category == "REFUND_PENDING":
        return "Manual review", "Refund delay requires confirmation"
    else:
        return "Ask for more info", "Unclear case, need clarification"

def generate_resolutions(df: pd.DataFrame):
    df[["suggested_action", "justification"]] = df.apply(
        suggest_resolution, axis=1, result_type="expand"
    )
    return df

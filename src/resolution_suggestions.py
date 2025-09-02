import pandas as pd

def suggest_resolutions(classified_df):
    results = []
    for _, row in classified_df.iterrows():
        dispute_id = row["dispute_id"]
        category = row["predicted_category"]

        # --- Mapping logic ---
        if category == "DUPLICATE_CHARGE":
            action = "Auto-refund"
            justification = "Duplicate transaction detected, refund automatically."
        elif category == "FAILED_TRANSACTION":
            action = "Manual review"
            justification = "Investigate failed payment and retry if necessary."
        elif category == "FRAUD":
            action = "Mark as potential fraud"
            justification = "Flag for fraud investigation and escalation."
        elif category == "REFUND_PENDING":
            action = "Escalate to bank"
            justification = "Refund still pending, escalate for resolution."
        else:
            action = "Ask for more info"
            justification = "Dispute unclear, request more details."

        results.append({
            "dispute_id": dispute_id,
            "suggested_action": action,
            "justification": justification
        })

    resolutions_df = pd.DataFrame(results)
    resolutions_df.to_csv("resolutions.csv", index=False)
    return resolutions_df

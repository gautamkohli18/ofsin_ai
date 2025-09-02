import pandas as pd

def classify_disputes(disputes_df, transactions_df):
    results = []
    for _, row in disputes_df.iterrows():
        dispute_id = row["dispute_id"]
        txn_id = row["txn_id"]
        status = row.get("status", "").lower()
        desc = str(row.get("description", "")).lower()

        if "duplicate" in desc:
            category = "DUPLICATE_CHARGE"
        elif "fail" in desc or "error" in desc or status == "failed":
            category = "FAILED_TRANSACTION"
        elif "fraud" in desc or "unauthorized" in desc:
            category = "FRAUD"
        elif "refund" in desc or "pending" in desc:
            category = "REFUND_PENDING"
        else:
            category = "OTHERS"

        confidence = 0.85 if category != "OTHERS" else 0.6
        explanation = f"Classified as {category} based on keywords/status match."

        results.append({
            "dispute_id": dispute_id,
            "predicted_category": category,
            "confidence": confidence,
            "explanation": explanation
        })

    classified_df = pd.DataFrame(results)
    classified_df.to_csv("classified_disputes.csv", index=False)
    return classified_df

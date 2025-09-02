import pandas as pd

def rule_based_classifier(description: str):
    desc = str(description).lower()
    if "duplicate" in desc or "charged twice" in desc:
        return "DUPLICATE_CHARGE", 0.9, "Duplicate charge detected"
    elif "failed" in desc or "declined" in desc:
        return "FAILED_TRANSACTION", 0.85, "Failed transaction keywords"
    elif "fraud" in desc or "unauthorized" in desc:
        return "FRAUD", 0.95, "Fraudulent/unauthorized keywords"
    elif "refund" in desc or "pending" in desc:
        return "REFUND_PENDING", 0.8, "Refund-related keywords"
    else:
        return "OTHERS", 0.5, "No strong keyword match"

def classify_disputes(df: pd.DataFrame):
    df[["predicted_category", "confidence", "explanation"]] = df["description"].apply(
        lambda x: pd.Series(rule_based_classifier(x))
    )
    return df

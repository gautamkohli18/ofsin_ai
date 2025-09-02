import pandas as pd

def suggest_resolution(df: pd.DataFrame) -> pd.DataFrame:
    suggestions = []
    for cat in df["category"]:
        if cat == "Fraud":
            suggestions.append("Escalate to fraud department")
        elif cat == "Payment Issue":
            suggestions.append("Check payment gateway logs")
        elif cat == "Account Issue":
            suggestions.append("Reset account credentials")
        else:
            suggestions.append("Manual review required")
    df["suggested_resolution"] = suggestions
    return df

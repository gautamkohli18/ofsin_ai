import pandas as pd

def classify_disputes(df: pd.DataFrame) -> pd.DataFrame:
    categories = []
    for text in df["description"]:
        text_lower = str(text).lower()
        if "fraud" in text_lower:
            categories.append("Fraud")
        elif "payment" in text_lower:
            categories.append("Payment Issue")
        elif "account" in text_lower:
            categories.append("Account Issue")
        else:
            categories.append("Other")
    df["category"] = categories
    return df


from fuzzywuzzy import fuzz
import pandas as pd

def detect_potential_duplicates(transactions_df: pd.DataFrame):
    duplicates = []
    for i in range(len(transactions_df)):
        for j in range(i+1, len(transactions_df)):
            if fuzz.ratio(str(transactions_df.iloc[i]["amount"]), str(transactions_df.iloc[j]["amount"])) > 90:
                t1 = pd.to_datetime(transactions_df.iloc[i]["timestamp"])
                t2 = pd.to_datetime(transactions_df.iloc[j]["timestamp"])
                if abs((t1 - t2).total_seconds()) < 300:
                    duplicates.append((transactions_df.iloc[i]["transaction_id"], transactions_df.iloc[j]["transaction_id"]))
    return duplicates

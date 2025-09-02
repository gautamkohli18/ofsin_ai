from fuzzywuzzy import fuzz

def detect_potential_duplicates(transactions_df):
    duplicates = []
    for i in range(len(transactions_df)):
        for j in range(i + 1, len(transactions_df)):
            desc1 = str(transactions_df.iloc[i]["description"])
            desc2 = str(transactions_df.iloc[j]["description"])
            if fuzz.ratio(desc1, desc2) > 90:
                duplicates.append((
                    transactions_df.iloc[i].get("transaction_id", i),
                    transactions_df.iloc[j].get("transaction_id", j)
                ))
    return duplicates

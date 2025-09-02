from fuzzywuzzy import fuzz

def detect_potential_duplicates(transactions_df):
    duplicates = []
    cols = transactions_df.columns

    # Choose which field to compare
    if "description" in cols:
        field = "description"
    elif "merchant" in cols:
        field = "merchant"
    elif "details" in cols:
        field = "details"
    else:
        return []  # no comparable field

    for i in range(len(transactions_df)):
        for j in range(i + 1, len(transactions_df)):
            val1 = str(transactions_df.iloc[i][field])
            val2 = str(transactions_df.iloc[j][field])
            if fuzz.ratio(val1, val2) > 90:
                duplicates.append((
                    transactions_df.iloc[i].get("transaction_id", i),
                    transactions_df.iloc[j].get("transaction_id", j),
                    field,
                    val1,
                    val2
                ))
    return duplicates

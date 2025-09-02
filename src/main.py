import pandas as pd
from classifier import classify_disputes
from resolution import generate_resolutions
from query_agent import create_dispute_agent, ask_query
from fuzzy_duplicates import detect_potential_duplicates

disputes_df = pd.read_csv("data/disputes.csv")
transactions_df = pd.read_csv("data/transactions.csv")

classified_df = classify_disputes(disputes_df)
classified_df.to_csv("outputs/classified_disputes.csv", index=False)

resolved_df = generate_resolutions(classified_df)
resolved_df.to_csv("outputs/resolutions.csv", index=False)

agent = create_dispute_agent(resolved_df)
print(ask_query(agent, "How many duplicate charges today?"))
print(ask_query(agent, "List unresolved fraud disputes"))
print(ask_query(agent, "Break down disputes by type"))

potential_duplicates = detect_potential_duplicates(transactions_df)
print("Potential duplicate transactions:", potential_duplicates)

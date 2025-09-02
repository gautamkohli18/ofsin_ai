import streamlit as st
import pandas as pd
from classify_disputes import classify_disputes
from resolution_suggestions import suggest_resolution
from fuzzy_duplicates import detect_potential_duplicates
from query_agent import create_dispute_agent, ask_query

st.set_page_config(page_title="AI-Powered Dispute Assistant", layout="wide")

st.title("🤖 AI-Powered Dispute Assistant")
disputes_df = pd.read_csv("data/disputes.csv")
transactions_df = pd.read_csv("data/transactions.csv")
classified_df = classify_disputes(disputes_df)
classified_df.to_csv("outputs/classified_disputes.csv", index=False)
resolved_df = suggest_resolution(classified_df)
resolved_df.to_csv("outputs/resolutions.csv", index=False)
duplicates = detect_potential_duplicates(transactions_df)

st.subheader("📂 Classified Disputes")
st.dataframe(classified_df)

st.subheader("🛠 Suggested Resolutions")
st.dataframe(resolved_df)

st.subheader("🔍 Potential Duplicate Transactions")
if duplicates:
    st.write("Found possible duplicates based on text similarity:")
    st.dataframe(pd.DataFrame(duplicates, columns=["id_1", "id_2", "field", "value_1", "value_2"]))
else:
    st.write("No duplicates found.")


st.subheader("🔎 Ask a Question about Disputes")
query = st.text_input("Enter your query")

if query:
    agent = create_dispute_agent(resolved_df)
    answer = ask_query(agent, query)
    st.write("**Answer:**", answer)

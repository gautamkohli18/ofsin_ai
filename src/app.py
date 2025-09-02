import streamlit as st
import pandas as pd
from classifier import classify_disputes
from resolution import generate_resolutions
from query_agent import create_dispute_agent, ask_query
from fuzzy_duplicates import detect_potential_duplicates

st.set_page_config(page_title="AI Dispute Assistant", layout="wide")
st.title("🤖 AI-Powered Dispute Assistant")

disputes_df = pd.read_csv("data/disputes.csv")
transactions_df = pd.read_csv("data/transactions.csv")

classified_df = classify_disputes(disputes_df)
resolved_df = generate_resolutions(classified_df)

st.subheader("📂 Classified Disputes")
st.dataframe(resolved_df)

classified_df.to_csv("outputs/classified_disputes.csv", index=False)
resolved_df.to_csv("outputs/resolutions.csv", index=False)

st.subheader("🔍 Ask a Question about Disputes")
query = st.text_input("Enter your query")
if query:
    agent = create_dispute_agent(resolved_df)
    answer = ask_query(agent, query)
    st.success(answer)

st.subheader("⚡ Potential Duplicate Transactions")
duplicates = detect_potential_duplicates(transactions_df)
st.write(duplicates if duplicates else "No duplicates found")

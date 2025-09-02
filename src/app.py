import streamlit as st
import pandas as pd
from classify_disputes import classify_disputes
from resolution_suggestions import suggest_resolutions
from fuzzy_duplicates import detect_potential_duplicates
from query_agent import create_dispute_agent, ask_query

st.title("Osfin AI - Detect Disputes")
disputes_df = pd.read_csv("data/disputes.csv")
transactions_df = pd.read_csv("data/transactions.csv")
classified_df = classify_disputes(disputes_df, transactions_df)
st.subheader("📌 Classified Disputes")
st.dataframe(classified_df.head())
resolutions_df = suggest_resolutions(classified_df)
st.subheader("🛠 Suggested Resolutions")
st.dataframe(resolutions_df.head())
agent = create_dispute_agent(classified_df)
st.subheader("🔍 Ask a Question about Disputes")
query = st.text_input("Enter your query")
if query:
    try:
        answer = ask_query(agent, query)
        st.success(answer)
    except Exception as e:
        st.error(f"Error: {e}")


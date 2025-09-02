import streamlit as st
import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_community.llms import Ollama
import re

# ---- Create Agent ----
def create_dispute_agent(df: pd.DataFrame):
    llm = Ollama(model="mistral")  # ✅ lightweight, works under 4GB
    agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        allow_dangerous_code=True,
        handle_parsing_errors=True,
    )
    return agent, llm

# ---- Robust Ask Function ----
def ask_query(agent, llm, query, df: pd.DataFrame = None):
    query_str = str(query) if query is not None else ""
    dispute_ids = re.findall(r'\bD0\d+\b', query_str.upper())

    if dispute_ids and df is not None:
        responses = []
        for dispute_id in dispute_ids:
            row = df[df['dispute_id'].str.upper() == dispute_id]
            if row.empty:
                responses.append(f"No details found for dispute ID {dispute_id}.")
                continue

            dispute = row.iloc[0]
            prompt = f"""
You are a dispute resolution assistant. Summarize this dispute clearly:

Dispute ID: {dispute['dispute_id']}
Customer ID: {dispute['customer_id']}
Transaction ID: {dispute['txn_id']}
Description: {dispute['description']}
Transaction Type: {dispute['txn_type']} via {dispute['channel']}
Amount: {dispute['amount']}
Category: {dispute['category']}
Suggested Resolution: {dispute['suggested_resolution']}

Write a clear and helpful explanation for the support team.
"""
            try:
                content = llm.invoke(prompt).content
                responses.append(f"--- Explanation for {dispute_id} ---\n{content}")
            except Exception as e:
                responses.append(f"⚠️ Error explaining dispute {dispute_id}: {e}")

        return "\n\n".join(responses)
    else:
        try:
            return agent.run(query_str)
        except Exception as e:
            return f"⚠️ Error while processing query: {e}"

# ---- Example DataFrame ----
data = [
    {
        "dispute_id": "D001",
        "customer_id": "C1001",
        "txn_id": "T9001",
        "description": "Refund requested for double charge",
        "txn_type": "Payment",
        "channel": "Online",
        "amount": 150.0,
        "category": "Billing",
        "suggested_resolution": "Refund the duplicate charge"
    },
    {
        "dispute_id": "D003",
        "customer_id": "C1003",
        "txn_id": "T9003",
        "description": "Late delivery of order",
        "txn_type": "Order",
        "channel": "Retail",
        "amount": 250.0,
        "category": "Logistics",
        "suggested_resolution": "Offer 10% discount and expedited delivery"
    }
]
df = pd.DataFrame(data)

# ---- Streamlit UI ----
st.title("🔎 Ask a Question about Disputes")

query = st.text_input("Enter your query")

if st.button("Submit") and query:
    # ✅ Create agent and LLM
    agent, llm = create_dispute_agent(df)

    # ✅ Call the robust ask_query
    answer = ask_query(agent, llm, query, df=df)

    st.subheader("Answer:")
    st.write(answer)


import pandas as pd
import re
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_community.llms import Ollama

# ---- Create Agent ----
def create_dispute_agent(df: pd.DataFrame):
    """
    Create a LangChain agent for a dispute DataFrame using Ollama LLM.
    Returns both the agent and the LLM instance.
    """
    llm = Ollama(model="mistral")  # ✅ lightweight, works under 4GB

    agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        allow_dangerous_code=True,
        handle_parsing_errors=True,  # prevents crashes
    )

    return agent, llm  # ✅ return both agent and llm so we can use llm separately


# ---- Robust Hybrid Ask Function ----
def ask_query(agent, llm, query, df: pd.DataFrame = None):
    """
    Handles both free-form queries and specific dispute ID queries.
    Features:
        - Detects dispute IDs of any format like D001, D012, D099, etc.
        - Fuzzy: accepts lowercase, trailing punctuation, and words around ID.
        - Supports multiple dispute IDs in a single query.
        - Safe fallback to agent reasoning if no dispute ID is found.
    """
    # Ensure query is a string
    query_str = str(query) if query is not None else ""

    # 🔍 Fuzzy pattern: detect D0xxx anywhere, case-insensitive
    dispute_ids = re.findall(r'\bD0\d+\b', query_str.upper())

    if dispute_ids and df is not None:
        responses = []
        for dispute_id in dispute_ids:
            row = df[df['dispute_id'].str.upper() == dispute_id]
            if row.empty:
                responses.append(f"No details found for dispute ID {dispute_id}.")
                continue

            dispute = row.iloc[0]

            # Prompt the LLM to write a human-friendly explanation
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
                # Add a header to make it easy to identify which dispute this is
                responses.append(f"--- Explanation for {dispute_id} ---\n{content}")
            except Exception as e:
                responses.append(f"⚠️ Error explaining dispute {dispute_id}: {e}")

        # Combine all dispute explanations
        return "\n\n".join(responses)

    else:
        # Fallback: general DataFrame reasoning mode
        try:
            return agent.run(query_str)
        except Exception as e:
            return f"⚠️ Error while processing query: {e}"


# ---- Example Usage ----
if __name__ == "__main__":
    # Example DataFrame
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
            "dispute_id": "D002",
            "customer_id": "C1002",
            "txn_id": "T9002",
            "description": "Incorrect item delivered",
            "txn_type": "Order",
            "channel": "Retail",
            "amount": 200.0,
            "category": "Logistics",
            "suggested_resolution": "Replace the item"
        }
    ]
    df = pd.DataFrame(data)

    # Create agent
    agent, llm = create_dispute_agent(df)

    # Example queries
    queries = [
        "What is the suggested resolution for dispute D001?",
        "Explain D002 and D003 for me",
        "Give me a summary of all billing disputes"
    ]

    for q in queries:
        print(f"\nQuery: {q}")
        print(ask_query(agent, llm, q, df))

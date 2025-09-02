import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_community.llms import Ollama


# ---- Create Agent ----
def create_dispute_agent(df: pd.DataFrame):
    llm = Ollama(model="mistral")  # ✅ lightweight, works under 4GB

    agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        allow_dangerous_code=True,
        handle_parsing_errors=True,  # prevents crashes
    )

    return agent, llm  # ✅ return both agent and llm so we can use llm separately


# ---- Hybrid Ask Function ----
def ask_query(agent, llm, query: str, df: pd.DataFrame = None):
    # 🔍 Check if query contains a dispute ID like D001
    if "D00" in query.upper() and df is not None:
        try:
            # Extract dispute details directly from df
            dispute_id = query.split()[-1].strip(":")
            row = df[df['dispute_id'] == dispute_id]

            if row.empty:
                return f"No details found for dispute ID {dispute_id}."

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

            return llm.invoke(prompt).content  # ✅ use LLM directly

        except Exception as e:
            return f"⚠️ Error while explaining dispute: {e}"

    else:
        # Dataframe reasoning mode
        try:
            return agent.run(query)
        except Exception as e:
            return f"⚠️ Error while processing query: {e}"

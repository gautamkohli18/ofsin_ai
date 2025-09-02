import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_community.llms import Ollama  # swap with KrutrimLLM if needed

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
    return agent


# ---- Hybrid Ask Function ----
def ask_query(agent, query: str):
    # 🔍 Check if query contains a dispute ID like D001, D002 etc.
    if "D00" in query.upper():
        # Prompting mode
        prompt = f"""
        You are a dispute resolution assistant. A user has asked: "{query}".
        Look at the provided dataframe of disputes and explain in detail:
        - The dispute description
        - Transaction type and channel
        - Amount and customer involved
        - Suggested resolution
        Provide a clear and human-friendly explanation.
        """
        return agent.llm(prompt)
    else:
        # Dataframe reasoning mode
        try:
            return agent.run(query)
        except Exception as e:
            return f"⚠️ Error while processing query: {e}"

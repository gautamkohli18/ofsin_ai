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
def ask_query(agent, query: str, df: pd.DataFrame = None):
    """
    Hybrid approach:
    1. If query asks about a specific dispute_id → fetch row & explain
    2. Otherwise → let agent handle it
    """

    if "dispute" in query.lower() and "d" in query.lower():
        try:
            # Extract dispute id from query
            words = query.upper().split()
            dispute_id = next((w for w in words if w.startswith("D")), None)

            if dispute_id and df is not None:
                row = df[df["dispute_id"] == dispute_id]
                if not row.empty:
                    row_dict = row.iloc[0].to_dict()
                    explanation = f"""
### 📝 Dispute Explanation

- **Dispute ID:** {row_dict['dispute_id']}
- **Customer ID:** {row_dict['customer_id']}
- **Transaction ID:** {row_dict['txn_id']}
- **Description:** {row_dict['description']}
- **Transaction Type:** {row_dict['txn_type']}
- **Channel:** {row_dict['channel']}
- **Amount:** {row_dict['amount']}
- **Date:** {row_dict['created_at']}
- **Category:** {row_dict['category']}
- **Suggested Resolution:** {row_dict['suggested_resolution']}

📌 Based on this data, the dispute falls under **{row_dict['category']}**.  
The recommended resolution is: **{row_dict['suggested_resolution']}**.
"""
                    return explanation
                else:
                    return f"No details found for dispute ID {dispute_id}."
        except Exception as e:
            return f"⚠️ Could not explain dispute due to error: {e}"

    # General queries → delegate to agent
    try:
        return agent.run(query)
    except Exception as e:
        return f"⚠️ Agent error: {e}"

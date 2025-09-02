from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents import AgentExecutor
import pandas as pd

# ---- Krutrim LLM Wrapper ----
from langchain.llms.base import LLM
import requests

class KrutrimLLM(LLM):
    def __init__(self, api_key: str, model: str = "Gemma-3-27B-IT"):
        super().__init__()
        self.api_key = api_key
        self.model = model

    @property
    def _llm_type(self) -> str:
        return "krutrim"

    def _call(self, prompt: str, stop=None, run_manager=None) -> str:
        url = "https://cloud.olakrutrim.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 512,
            "stream": False
        }
        resp = requests.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


# ---- Dispute Agent ----
def create_dispute_agent(df: pd.DataFrame, api_key: str):
    llm = KrutrimLLM(api_key=api_key)

    agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        allow_dangerous_code=True
    )

    # Wrap in AgentExecutor for better error handling
    return AgentExecutor.from_agent_and_tools(
        agent.agent,
        agent.tools,
        verbose=True,
        handle_parsing_errors=True
    )


# ---- Ask Query with Fallback ----
def ask_query(agent, df: pd.DataFrame, query: str) -> str:
    try:
        return agent.run(query)
    except Exception as e:
        # ✅ Fallback mode for common queries
        if "type" in query.lower() and "dispute" in query.lower():
            return f"Types of disputes: {df['category'].unique().tolist()}"
        elif "failed" in query.lower():
            return f"Number of failed transactions: {df[df['status']=='failed'].shape[0]}"
        elif "duplicate" in query.lower():
            return "Duplicate detection is enabled. Please check the duplicates section."
        else:
            return f"⚠️ Agent failed. Error: {str(e)}"

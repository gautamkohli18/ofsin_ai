import requests
from langchain.llms.base import LLM
from langchain_experimental.agents import create_pandas_dataframe_agent
from typing import Any, List, Optional
from pydantic import Field

class KrutrimLLM(LLM):
    api_key: str = Field(..., description="Krutrim API Key")

    @property
    def _llm_type(self) -> str:
        return "krutrim"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "model": "Gemma-3-27B-IT",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 512,
            "stream": False
        }
        response = requests.post(
            "https://cloud.olakrutrim.com/v1/chat/completions",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

def create_dispute_agent(df, api_key: str):
    llm = KrutrimLLM(api_key=api_key)
    agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        allow_dangerous_code=True,
        handle_parsing_errors=True  # ✅ fixes parsing crash
    )
    return agent

def ask_query(agent, query: str):
    try:
        return agent.run(query)
    except Exception as e:
        return f"⚠️ Parsing error handled gracefully: {str(e)}"

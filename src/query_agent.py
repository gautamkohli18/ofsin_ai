from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_core.language_models import LLM
from typing import Any, List, Optional
import requests


class KrutrimLLM(LLM):
    api_url: str = "https://cloud.olakrutrim.com/v1/chat/completions"
    api_key: str = "9klDvsb_AT5IRCiiBP00Q"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "Gemma-3-27B-IT",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 512,
            "stream": False
        }
        resp = requests.post(self.api_url, headers=headers, json=payload)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    @property
    def _identifying_params(self) -> dict:
        return {"model": "Gemma-3-27B-IT"}

    @property
    def _llm_type(self) -> str:
        return "krutrim-llm"

    # ✅ Support LangChain v0.2 agents
    def bind(self, **kwargs):
        return self


def create_dispute_agent(df):
    llm = KrutrimLLM()
    agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        allow_dangerous_code=True
    )
    return agent


def ask_query(agent, query: str):
    return agent.run(query)

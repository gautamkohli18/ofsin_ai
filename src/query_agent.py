import os
import requests
from langchain_experimental.agents import create_pandas_dataframe_agent

KRUTRIM_API_KEY = os.getenv("KRUTRIM_API_KEY")
KRUTRIM_URL = "https://cloud.olakrutrim.com/v1/chat/completions"

class KrutrimLLM:
    def __init__(self, model="Gemma-3-27B-IT", max_tokens=512, temperature=0.2):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def run(self, prompt):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {KRUTRIM_API_KEY}"
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
            "stream": False
        }
        response = requests.post(KRUTRIM_URL, headers=headers, json=payload)
        if response.status_code != 200:
            raise Exception(f"Error from Krutrim API: {response.text}")
        data = response.json()
        return data["choices"][0]["message"]["content"]

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

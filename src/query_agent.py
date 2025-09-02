from langchain_community.llms import HuggingFaceHub
from langchain_experimental.agents import create_pandas_dataframe_agent
import os

def create_dispute_agent(df):
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-base",
        task="text2text-generation",
        model_kwargs={"temperature": 0.1, "max_length": 512}
    )
    agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True
    )
    return agent

def ask_query(agent, query: str):
    return agent.run(query)

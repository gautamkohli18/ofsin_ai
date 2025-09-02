from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_community.llms import HuggingFaceEndpoint

def create_dispute_agent(df):
    llm = HuggingFaceEndpoint(
        repo_id="google/flan-t5-base",       
        task="text2text-generation",        
        max_new_tokens=256,
        temperature=0.1
    )
    agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        allow_dangerous_code=True            
    )
    return agent

def ask_query(agent, query: str):
    return agent.run(query)

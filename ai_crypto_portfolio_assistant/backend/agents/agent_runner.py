# agent_runner.py
import os
from langchain.agents import initialize_agent
from langchain.llms import OpenAI
from langchain.agents import AgentType
from agent_tools import crypto_analysis_tool, coin_ohlc_tool, news_check_tool

# Choose an LLM. Here we use OpenAI (you need OPENAI_API_KEY in env).
llm = OpenAI(temperature=0)

tools = [crypto_analysis_tool, coin_ohlc_tool, news_check_tool]

# initialize a zero-shot agent with react-style tool use
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
)

def run_agent(prompt: str) -> str:
    """
    Run the agent on the incoming prompt and return the textual response.
    """
    return agent.run(prompt)

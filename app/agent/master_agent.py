import os
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

from app.agent.prompts import MASTER_SYSTEM_PROMPT
from app.agent.tools.unstructured_tool import retrieve_unstructured_context
from app.agent.tools.structured_tool import query_structured_data

master_llm = ChatOpenAI(
    model="gpt-5.2",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY"),
    verbose=True
)

tools = [retrieve_unstructured_context, query_structured_data]

def get_master_agent_executor():
    return create_react_agent(
        model=master_llm,
        tools=tools,
        prompt=MASTER_SYSTEM_PROMPT
    )

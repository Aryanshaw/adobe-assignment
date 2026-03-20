import os
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

from app.agent.prompts import MASTER_SYSTEM_PROMPT, get_document_summaries
from app.agent.tools.unstructured_tool import retrieve_unstructured_context
from app.agent.tools.structured_tool import query_structured_data

master_llm = ChatOpenAI(
    model="gpt-5.2",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY"),
    verbose=True
)

tools = [retrieve_unstructured_context, query_structured_data]


async def get_master_agent_executor():
    doc_summaries = await get_document_summaries()
    prompt = MASTER_SYSTEM_PROMPT.format(document_summaries=doc_summaries)
    return create_react_agent(
        model=master_llm,
        tools=tools,
        prompt=prompt
    )

import os
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from datetime import datetime

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
    current_date = datetime.now()
    current_date_str = current_date.strftime("%Y-%m-%d")
    current_date_year = current_date.year
    current_date_month = current_date.month
    
    current_time = f"date: {current_date_str} {current_date_year}-{current_date_month}"

    prompt = MASTER_SYSTEM_PROMPT.format(document_summaries=doc_summaries, current_time=current_time)
    return create_react_agent(
        model=master_llm,
        tools=tools,
        prompt=prompt
    )

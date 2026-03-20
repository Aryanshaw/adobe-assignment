from langchain_core.tools import tool
from app.agent.subagents.structured_agent import structured_agent_executor

@tool
async def query_structured_data(requirement_prompt: str) -> str:
    """
    Queries structured data tables (CSV/XLSX files) loaded into SQLite.
    Uses enriched schema descriptions to generate accurate SQL.

    Use when the question asks about:
    - Revenue figures, targets, variances
    - Department-level KPIs and performance metrics
    - Headcount numbers, budgets
    - Any ranking, comparison, or trend across numeric data
    - "Which is highest/lowest/best/worst"

    Args:
        requirement_prompt: Precise description of what data is needed,
                            including relevant columns, filters, ordering.

    Returns:
        Query results formatted as a readable summary with table name cited.
    """
    response = await structured_agent_executor.ainvoke({"messages": [("user", requirement_prompt)]})
    return response["messages"][-1].content

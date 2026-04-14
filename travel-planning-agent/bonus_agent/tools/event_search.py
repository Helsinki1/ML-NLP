"""Events / things-to-do search tool backed by Tavily web search."""

import json
from langchain_tavily import TavilySearch


def _get_tavily():
    return TavilySearch(max_results=5, topic="general")


def search_events(
    city: str,
    date: str = "",
    interests: str = "",
) -> str:
    """Search for events, attractions, and things to do in a city. Returns search results JSON or an error."""
    if not city:
        return "Error: city is required"

    query = f"things to do in {city}"
    if interests:
        query += f" {interests}"
    if date:
        query += f" on {date}"
    query += " recommendations"

    try:
        results = _get_tavily().invoke({"query": query})
    except Exception as e:
        return f"Error: search failed — {e}"

    return json.dumps({"query": query, "results": results})

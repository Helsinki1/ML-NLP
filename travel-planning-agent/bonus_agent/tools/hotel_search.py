"""Hotel search tool backed by Tavily web search."""

import json
from datetime import date, datetime
from langchain_tavily import TavilySearch


def _get_tavily():
    return TavilySearch(max_results=5, topic="general")


def search_hotels(
    city: str,
    checkin_date: str,
    checkout_date: str,
    guests: int = 1,
) -> str:
    """Search for available hotels in a city for given dates. Returns search results JSON or an error."""
    if not city:
        return "Error: city is required"
    if not checkin_date or not checkout_date:
        return "Error: both checkin_date and checkout_date are required (YYYY-MM-DD)"

    try:
        cin = datetime.strptime(checkin_date, "%Y-%m-%d").date()
    except ValueError:
        return "Error: checkin_date must be in YYYY-MM-DD format"
    try:
        cout = datetime.strptime(checkout_date, "%Y-%m-%d").date()
    except ValueError:
        return "Error: checkout_date must be in YYYY-MM-DD format"

    if cin < date.today():
        return "Error: checkin_date must be today or in the future"
    if cout <= cin:
        return "Error: checkout_date must be after checkin_date"
    if not (1 <= guests <= 20):
        return "Error: guests must be between 1 and 20"

    query = (
        f"hotels in {city} available {checkin_date} to {checkout_date}"
        f" for {guests} guest{'s' if guests > 1 else ''} prices and reviews"
    )

    try:
        results = _get_tavily().invoke({"query": query})
    except Exception as e:
        return f"Error: search failed — {e}"

    return json.dumps({"query": query, "results": results})

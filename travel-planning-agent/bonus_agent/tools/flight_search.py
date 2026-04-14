"""Flight search tool backed by Tavily web search."""

import json
from datetime import date, datetime
from langchain_tavily import TavilySearch


def _get_tavily():
    return TavilySearch(max_results=5, topic="general")


def search_flights(
    origin: str,
    destination: str,
    departure_date: str,
    return_date: str = "",
    passengers: int = 1,
) -> str:
    """Search for available flights between two cities on a given date. Returns search results JSON or an error."""
    if not origin or not destination:
        return "Error: both origin and destination are required"
    if origin.strip().lower() == destination.strip().lower():
        return "Error: origin and destination must be different"
    if not departure_date:
        return "Error: departure_date is required (YYYY-MM-DD)"

    try:
        dep = datetime.strptime(departure_date, "%Y-%m-%d").date()
    except ValueError:
        return "Error: departure_date must be in YYYY-MM-DD format"
    if dep < date.today():
        return "Error: departure_date must be today or in the future"

    if return_date:
        try:
            ret = datetime.strptime(return_date, "%Y-%m-%d").date()
        except ValueError:
            return "Error: return_date must be in YYYY-MM-DD format"
        if ret <= dep:
            return "Error: return_date must be after departure_date"

    if not (1 <= passengers <= 9):
        return "Error: passengers must be between 1 and 9"

    query = f"flights from {origin} to {destination} on {departure_date}"
    if return_date:
        query += f" returning {return_date}"
    if passengers > 1:
        query += f" for {passengers} passengers"
    query += " prices and availability"

    try:
        results = _get_tavily().invoke({"query": query})
    except Exception as e:
        return f"Error: search failed — {e}"

    return json.dumps({"query": query, "results": results})

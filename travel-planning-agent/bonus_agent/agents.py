"""Sub-agent definitions: one create_react_agent per specialist domain."""

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from bonus_agent.tools.auth import lookup_user_by_email, lookup_user_by_name
from bonus_agent.tools.flight_search import search_flights
from bonus_agent.tools.hotel_search import search_hotels
from bonus_agent.tools.event_search import search_events
from bonus_agent.tools.booking import create_booking, get_booking, cancel_booking
from bonus_agent.tools.escalation import transfer_to_human_agent


def build_agents(model_name: str = "gpt-4o-mini", temperature: float = 0.0):
    """Build and return all specialist sub-agents plus the escalation tool."""
    model = ChatOpenAI(model=model_name, temperature=temperature)

    auth_agent = create_react_agent(
        model=model,
        tools=[lookup_user_by_email, lookup_user_by_name],
        name="auth_agent",
        prompt=(
            "You are the authentication specialist for a travel planning service. "
            "Your ONLY job is to identify and authenticate the user. "
            "Ask for their email or full name, then look them up. "
            "Return the user profile once found. If you cannot find the user, "
            "say so clearly. Do NOT help with any travel-related requests — "
            "only handle authentication."
        ),
    )

    flight_agent = create_react_agent(
        model=model,
        tools=[search_flights],
        name="flight_agent",
        prompt=(
            "You are a flight search specialist. "
            "Help the user find flights by searching for routes between cities. "
            "Always ask for: origin city, destination city, departure date (YYYY-MM-DD), "
            "optional return date, and number of passengers. "
            "Present results clearly with prices when available. "
            "Do NOT book anything — only search and present options."
        ),
    )

    hotel_agent = create_react_agent(
        model=model,
        tools=[search_hotels],
        name="hotel_agent",
        prompt=(
            "You are a hotel search specialist. "
            "Help the user find hotels by searching for accommodations in a city. "
            "Always ask for: city, check-in date (YYYY-MM-DD), check-out date, "
            "and number of guests. "
            "Present results clearly with prices and ratings when available. "
            "Do NOT book anything — only search and present options."
        ),
    )

    event_agent = create_react_agent(
        model=model,
        tools=[search_events],
        name="event_agent",
        prompt=(
            "You are a local events and activities specialist. "
            "Help the user discover things to do, attractions, tours, "
            "restaurants, and events in a city. "
            "Ask for the city and optionally their interests or travel dates. "
            "Present results with descriptions and practical information."
        ),
    )

    booking_agent = create_react_agent(
        model=model,
        tools=[create_booking, get_booking, cancel_booking],
        name="booking_agent",
        prompt=(
            "You are the booking management specialist. "
            "You can create new bookings, retrieve existing bookings, or cancel bookings. "
            "Before creating or cancelling a booking, ALWAYS list the full details "
            "and ask the user for explicit confirmation (yes/no). "
            "For new bookings you need: user_id, booking_type (flight/hotel/event), "
            "and a details JSON string. "
            "Never create or cancel a booking without confirmation."
        ),
    )

    return {
        "auth_agent": auth_agent,
        "flight_agent": flight_agent,
        "hotel_agent": hotel_agent,
        "event_agent": event_agent,
        "booking_agent": booking_agent,
        "escalation_tool": transfer_to_human_agent,
    }

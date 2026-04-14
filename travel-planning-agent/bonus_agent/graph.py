"""Supervisor graph: assembles sub-agents into a routed multi-agent system."""

from langchain_openai import ChatOpenAI
from langgraph_supervisor import create_supervisor

from bonus_agent.agents import build_agents
from bonus_agent.tools.escalation import transfer_to_human_agent

SUPERVISOR_PROMPT = """\
You are the lead coordinator for a travel planning service. You manage a team \
of specialist agents and route user requests to the appropriate one.

## Your team
- **auth_agent**: Authenticates the user by email or name. MUST be called first \
before any other agent can act.
- **flight_agent**: Searches for flights between cities.
- **hotel_agent**: Searches for hotels in a city.
- **event_agent**: Finds things to do, attractions, and events in a city.
- **booking_agent**: Creates, retrieves, or cancels bookings.

## Rules
1. **Authentication first.** Before routing to any search or booking agent, \
the user MUST be authenticated via auth_agent. If they haven't been authenticated \
yet, route to auth_agent first.
2. **One specialist at a time.** Route each sub-task to exactly one specialist. \
You may call multiple specialists in sequence for a complex trip plan.
3. **Confirmation before mutations.** The booking_agent will ask for confirmation \
before creating or cancelling bookings — do not bypass this.
4. **Never fabricate information.** Do not invent flight prices, hotel rates, \
or availability. Only relay information returned by the specialist agents.
5. **Escalate when stuck.** If a request is outside the team's capabilities, \
or if you've been unable to resolve the issue after multiple attempts, use the \
transfer_to_human_agent tool.
6. **Be concise and helpful.** Summarize specialist results for the user in a \
clear, friendly manner.
"""


def build_graph(model_name: str = "gpt-4o-mini", temperature: float = 0.0):
    """Build and compile the supervisor graph. Returns the compiled LangGraph app."""
    agents = build_agents(model_name=model_name, temperature=temperature)

    supervisor = create_supervisor(
        agents=[
            agents["auth_agent"],
            agents["flight_agent"],
            agents["hotel_agent"],
            agents["event_agent"],
            agents["booking_agent"],
        ],
        model=ChatOpenAI(model=model_name, temperature=temperature),
        prompt=SUPERVISOR_PROMPT,
        tools=[transfer_to_human_agent],
    )

    app = supervisor.compile()
    return app


def run_conversation(app, user_message: str, history: list | None = None) -> dict:
    """Run a single turn through the supervisor graph.

    Args:
        app: The compiled LangGraph supervisor app.
        user_message: The latest user message.
        history: Previous message history (list of dicts with 'role' and 'content').

    Returns:
        dict with 'response' (str) and 'messages' (full message history).
    """
    messages = []
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    result = app.invoke(
        {"messages": messages},
        config={"recursion_limit": 60},
    )

    all_messages = result["messages"]
    assistant_response = all_messages[-1].content if all_messages else ""

    return {
        "response": assistant_response,
        "messages": all_messages,
    }

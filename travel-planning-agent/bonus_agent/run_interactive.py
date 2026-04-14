"""Interactive CLI for testing the travel planning agent.

Usage:
    python -m bonus_agent.run_interactive
    OPENAI_API_KEY=... python -m bonus_agent.run_interactive
"""

import os
import sys

from bonus_agent.graph import build_graph


def main():
    model_name = os.environ.get("TRAVEL_AGENT_MODEL", "gpt-4o-mini")
    print(f"Building travel agent graph (model={model_name})...")
    app = build_graph(model_name=model_name, temperature=0.0)
    print("Ready! Type your messages below. Type 'quit' or 'exit' to stop.\n")

    history: list = []
    max_turns = 30
    turn = 0

    while turn < max_turns:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        history.append({"role": "user", "content": user_input})

        try:
            result = app.invoke(
                {"messages": history},
                config={"recursion_limit": 60},
            )
            all_messages = result["messages"]
            assistant_text = all_messages[-1].content if all_messages else "(no response)"
        except Exception as e:
            assistant_text = f"Error: {e}"

        print(f"\nAgent: {assistant_text}\n")

        # Rebuild history from the graph's message objects for context continuity
        rebuilt = []
        for m in all_messages:
            role = getattr(m, "type", None)
            content = getattr(m, "content", "")
            if role == "human":
                rebuilt.append({"role": "user", "content": content})
            elif role == "ai" and content:
                rebuilt.append({"role": "assistant", "content": content})
        history = rebuilt

        if "transferring to a human agent" in assistant_text.lower():
            print("[Session ended — transferred to human agent]")
            break

        turn += 1

    if turn >= max_turns:
        print(f"\n[Session ended — reached {max_turns} turn limit]")


if __name__ == "__main__":
    main()

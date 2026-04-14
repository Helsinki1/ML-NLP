"""Human-agent escalation tool (fallback / circuit breaker)."""


def transfer_to_human_agent(reason: str) -> str:
    """Transfer the conversation to a human agent. Use this when the user's request is outside the agent's capabilities, or when the agent is unable to resolve the issue after multiple attempts. Returns a transfer confirmation message."""
    if not reason:
        return "Error: a reason for the transfer is required"
    return (
        f"Transferring to a human agent. Reason: {reason}. "
        "A human representative will be with you shortly. "
        "Thank you for your patience."
    )

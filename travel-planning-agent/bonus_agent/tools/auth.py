"""User authentication / lookup tools."""

import json
from bonus_agent.data import get_users


def lookup_user_by_email(email: str) -> str:
    """Look up a user by their email address. Returns user profile JSON or an error."""
    if not email or not isinstance(email, str):
        return "Error: email is required and must be a non-empty string"
    email = email.strip().lower()
    for user in get_users().values():
        if user["email"].lower() == email:
            return json.dumps(user)
    return "Error: user not found for the provided email"


def lookup_user_by_name(first_name: str, last_name: str) -> str:
    """Look up a user by first and last name (case-insensitive). Returns user profile JSON or an error."""
    if not first_name or not last_name:
        return "Error: both first_name and last_name are required"
    first_name = first_name.strip().lower()
    last_name = last_name.strip().lower()
    for user in get_users().values():
        if (user["first_name"].lower() == first_name
                and user["last_name"].lower() == last_name):
            return json.dumps(user)
    return "Error: user not found for the provided name"

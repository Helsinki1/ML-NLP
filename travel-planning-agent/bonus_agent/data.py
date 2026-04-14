"""Mock in-memory databases for users and bookings."""

import uuid
from typing import Any

USERS: dict[str, dict[str, Any]] = {
    "u_alice_001": {
        "user_id": "u_alice_001",
        "first_name": "Alice",
        "last_name": "Johnson",
        "email": "alice.johnson@email.com",
        "phone": "+1-555-0101",
        "address": "123 Maple St, New York, NY 10001",
        "preferences": {"seat": "window", "meal": "vegetarian"},
    },
    "u_bob_002": {
        "user_id": "u_bob_002",
        "first_name": "Bob",
        "last_name": "Smith",
        "email": "bob.smith@email.com",
        "phone": "+1-555-0102",
        "address": "456 Oak Ave, Los Angeles, CA 90001",
        "preferences": {"seat": "aisle", "meal": "standard"},
    },
    "u_carol_003": {
        "user_id": "u_carol_003",
        "first_name": "Carol",
        "last_name": "Williams",
        "email": "carol.w@email.com",
        "phone": "+1-555-0103",
        "address": "789 Pine Rd, Chicago, IL 60601",
        "preferences": {"seat": "window", "meal": "halal"},
    },
    "u_david_004": {
        "user_id": "u_david_004",
        "first_name": "David",
        "last_name": "Brown",
        "email": "david.brown@email.com",
        "phone": "+1-555-0104",
        "address": "321 Elm Blvd, Houston, TX 77001",
        "preferences": {"seat": "aisle", "meal": "kosher"},
    },
    "u_eve_005": {
        "user_id": "u_eve_005",
        "first_name": "Eve",
        "last_name": "Davis",
        "email": "eve.davis@email.com",
        "phone": "+1-555-0105",
        "address": "654 Cedar Ln, Miami, FL 33101",
        "preferences": {"seat": "middle", "meal": "vegan"},
    },
    "u_frank_006": {
        "user_id": "u_frank_006",
        "first_name": "Frank",
        "last_name": "Garcia",
        "email": "frank.garcia@email.com",
        "phone": "+1-555-0106",
        "address": "987 Birch Way, Seattle, WA 98101",
        "preferences": {"seat": "window", "meal": "standard"},
    },
}

BOOKINGS: dict[str, dict[str, Any]] = {
    "BK-000001": {
        "booking_id": "BK-000001",
        "user_id": "u_alice_001",
        "booking_type": "flight",
        "status": "confirmed",
        "details": {
            "origin": "New York",
            "destination": "London",
            "departure_date": "2026-06-15",
            "return_date": "2026-06-22",
            "passengers": 1,
        },
    },
    "BK-000002": {
        "booking_id": "BK-000002",
        "user_id": "u_bob_002",
        "booking_type": "hotel",
        "status": "confirmed",
        "details": {
            "city": "Paris",
            "checkin_date": "2026-07-01",
            "checkout_date": "2026-07-05",
            "guests": 2,
        },
    },
}

_booking_counter = len(BOOKINGS)


def generate_booking_id() -> str:
    global _booking_counter
    _booking_counter += 1
    return f"BK-{_booking_counter:06d}"


def get_users() -> dict[str, dict[str, Any]]:
    return USERS


def get_bookings() -> dict[str, dict[str, Any]]:
    return BOOKINGS

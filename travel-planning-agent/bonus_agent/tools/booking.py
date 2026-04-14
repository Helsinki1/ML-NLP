"""Booking management tools (mock in-memory store)."""

import json
from bonus_agent.data import get_users, get_bookings, generate_booking_id


def create_booking(user_id: str, booking_type: str, details: str) -> str:
    """Create a new booking for an authenticated user. booking_type must be 'flight', 'hotel', or 'event'. details is a JSON string describing what is being booked. Returns the new booking record or an error."""
    if not user_id:
        return "Error: user_id is required"
    if user_id not in get_users():
        return "Error: user not found — authenticate first"
    if booking_type not in ("flight", "hotel", "event"):
        return "Error: booking_type must be one of: flight, hotel, event"
    if not details:
        return "Error: details are required (JSON string)"

    try:
        parsed_details = json.loads(details) if isinstance(details, str) else details
    except json.JSONDecodeError:
        return "Error: details must be a valid JSON string"

    booking_id = generate_booking_id()
    record = {
        "booking_id": booking_id,
        "user_id": user_id,
        "booking_type": booking_type,
        "status": "confirmed",
        "details": parsed_details,
    }
    get_bookings()[booking_id] = record
    return json.dumps(record)


def get_booking(booking_id: str) -> str:
    """Retrieve a booking by its ID. Returns the booking record JSON or an error."""
    if not booking_id:
        return "Error: booking_id is required"
    bookings = get_bookings()
    if booking_id not in bookings:
        return "Error: booking not found"
    return json.dumps(bookings[booking_id])


def cancel_booking(booking_id: str, reason: str) -> str:
    """Cancel an existing booking. The booking must exist and be in 'confirmed' status. reason must be provided. Returns the updated booking record or an error."""
    if not booking_id:
        return "Error: booking_id is required"
    if not reason:
        return "Error: reason for cancellation is required"

    bookings = get_bookings()
    if booking_id not in bookings:
        return "Error: booking not found"

    booking = bookings[booking_id]
    if booking["status"] != "confirmed":
        return f"Error: booking is already {booking['status']} and cannot be cancelled"

    booking["status"] = "cancelled"
    booking["cancel_reason"] = reason
    return json.dumps(booking)

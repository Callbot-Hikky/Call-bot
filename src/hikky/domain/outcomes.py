from enum import StrEnum


class CallOutcome(StrEnum):
    RESERVATION_CREATED = "reservation_created"
    CALLBACK_REQUESTED = "callback_requested"
    TRANSFERRED = "transferred"
    FALLBACK_MESSAGE = "fallback_message"
    INTERRUPTED = "interrupted"
    UNKNOWN_RESTAURANT = "unknown_restaurant"
    TECHNICAL_ERROR = "technical_error"

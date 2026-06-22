from hikky.domain.outcomes import CallOutcome


def test_call_outcome_has_all_expected_values():
    expected = {
        "reservation_created",
        "callback_requested",
        "transferred",
        "fallback_message",
        "interrupted",
        "unknown_restaurant",
        "technical_error",
    }
    actual = {outcome.value for outcome in CallOutcome}
    assert actual == expected

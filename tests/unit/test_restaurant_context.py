from datetime import datetime, time

import pytest
from pydantic import ValidationError

from hikky.domain.restaurant_context import (
    OpeningHours,
    RestaurantContext,
    RestaurantRules,
)


def _sample_context(**overrides) -> RestaurantContext:
    defaults = dict(
        id="rest-1",
        name="Chez Test",
        greeting="Bonjour, Chez Test.",
        opening_hours=[
            OpeningHours(weekday=d, opens=time(19, 0), closes=time(23, 0))
            for d in range(7)
        ],
        total_capacity=40,
        rules=RestaurantRules(
            max_group_size=8,
            reservation_duration_minutes=90,
        ),
        transfer_number=None,
        fallback_message="Merci, un membre de l'équipe vous rappellera.",
    )
    defaults.update(overrides)
    return RestaurantContext(**defaults)


def test_restaurant_context_is_immutable():
    ctx = _sample_context()
    with pytest.raises(ValidationError):
        ctx.name = "autre"


def test_restaurant_context_rejects_negative_capacity():
    with pytest.raises(ValidationError):
        _sample_context(total_capacity=-1)


def test_restaurant_context_rejects_empty_opening_hours():
    with pytest.raises(ValidationError):
        _sample_context(opening_hours=[])


def test_attributes_default_to_empty_and_address_to_none():
    ctx = _sample_context()
    assert ctx.attributes == {}
    assert ctx.address is None


def test_attributes_are_kept_verbatim():
    ctx = _sample_context(attributes={"dietary": {"halal": True}})
    assert ctx.attributes["dietary"]["halal"] is True


def test_is_open_at_returns_true_inside_hours():
    ctx = _sample_context()
    assert ctx.is_open_at(datetime(2026, 6, 22, 20, 0)) is True


def test_is_open_at_returns_false_outside_hours():
    ctx = _sample_context()
    assert ctx.is_open_at(datetime(2026, 6, 22, 12, 0)) is False

from datetime import datetime

import pytest

from hikky.domain.reservation_intent import ReservationIntent


def test_new_intent_is_incomplete():
    intent = ReservationIntent()
    assert intent.is_complete() is False
    assert intent.missing_slots() == {"date_time", "party_size", "customer_name"}


def test_filling_all_slots_makes_intent_complete():
    intent = ReservationIntent()
    intent = intent.with_date_time(datetime(2026, 7, 1, 20, 0))
    intent = intent.with_party_size(4)
    intent = intent.with_customer_name("Dupont")
    assert intent.is_complete() is True
    assert intent.missing_slots() == set()


def test_intent_is_immutable_each_with_returns_new_instance():
    intent = ReservationIntent()
    new_intent = intent.with_party_size(4)
    assert intent.party_size is None
    assert new_intent.party_size == 4
    assert new_intent is not intent


def test_party_size_must_be_positive():
    intent = ReservationIntent()
    with pytest.raises(ValueError):
        intent.with_party_size(0)


def test_customer_name_must_not_be_blank():
    intent = ReservationIntent()
    with pytest.raises(ValueError):
        intent.with_customer_name("   ")

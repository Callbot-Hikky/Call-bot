from datetime import datetime

import pytest

from hikky.domain.reservation_intent import ReservationIntent


def test_new_intent_is_incomplete():
    intent = ReservationIntent()
    assert intent.is_complete() is False
    assert intent.missing_slots() == {"date", "time", "party_size", "customer_name"}


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


# ── Date et heure decouplees ────────────────────────────────────────────
#
# Bug constate en appel reel : sur « demain matin », l'extracteur refusait
# d'inventer une heure (correct) mais perdait aussi le jour. Le bot
# reclamait alors « la date exacte » que le client venait de donner,
# jusqu'au declenchement du fallback.

from datetime import date, time  # noqa: E402


def test_date_alone_is_recorded_without_a_time():
    i = ReservationIntent().with_date(date(2026, 7, 21))
    assert i.date == date(2026, 7, 21)
    assert i.date_time is None


def test_only_the_time_is_missing_when_the_date_is_known():
    i = ReservationIntent().with_date(date(2026, 7, 21))
    assert "time" in i.missing_slots()
    assert "date" not in i.missing_slots()


def test_date_and_time_combine_into_date_time():
    i = ReservationIntent().with_date(date(2026, 7, 21)).with_time(time(20, 30))
    assert i.date_time == datetime(2026, 7, 21, 20, 30)


def test_setting_date_time_directly_fills_both_parts():
    i = ReservationIntent().with_date_time(datetime(2026, 7, 21, 20, 30))
    assert i.date == date(2026, 7, 21)
    assert i.time == time(20, 30)
    assert not {"date", "time"} & i.missing_slots()


def test_intent_incomplete_while_the_time_is_unknown():
    i = (
        ReservationIntent()
        .with_date(date(2026, 7, 21))
        .with_party_size(4)
        .with_customer_name("Dupont")
    )
    assert i.is_complete() is False
    assert i.missing_slots() == {"time"}


def test_intent_complete_once_the_time_arrives():
    i = (
        ReservationIntent()
        .with_date(date(2026, 7, 21))
        .with_party_size(4)
        .with_customer_name("Dupont")
        .with_time(time(20, 0))
    )
    assert i.is_complete() is True

"""L'aiguilleur : le code decide, le modele reste pour l'intelligence.

Mesure qui motive ce module : Q4 et Q6 echouent IDENTIQUEMENT face a une
dizaine de contraintes de prompt — les deux reposent trois fois « a
quelle heure ? » apres avoir recu l'heure. Ce n'est pas la capacite du
modele qui est en cause mais la charge. Le code sait exactement ce qui
manque ; il n'a besoin de personne pour choisir la question suivante.

Le modele reste sollicite pour ce qu'il reussit : comprendre les phrases
(l'extracteur) et repondre aux questions hors script.
"""

from datetime import date, time

from hikky.domain.question_router import Action, route_turn
from hikky.domain.reservation_intent import ReservationIntent


def _complet():
    return (
        ReservationIntent()
        .with_date(date(2026, 7, 22))
        .with_time(time(12, 0))
        .with_party_size(2)
        .with_customer_name("Dupont")
    )


# ── Detection : question du client ou reponse ? ─────────────────────────


def test_client_question_is_routed_to_the_model():
    """« Vous etes ouverts dimanche ? » doit atteindre le LLM."""
    d = route_turn(ReservationIntent(), "Vous êtes ouverts le dimanche ?", False)
    assert d.action is Action.ANSWER_QUESTION


def test_asking_about_the_recorded_name_is_a_question():
    """Bug reel : « A quel nom ? » recevait « C'est a quel nom ? »."""
    d = route_turn(_complet(), "À quel nom ?", False)
    assert d.action is Action.ANSWER_QUESTION


def test_plain_answer_is_not_treated_as_a_question():
    d = route_turn(ReservationIntent(), "Deux personnes", False)
    assert d.action is not Action.ANSWER_QUESTION


def test_repeat_request_is_routed_to_the_model():
    d = route_turn(_complet(), "Tu peux répéter ?", False)
    assert d.action is Action.ANSWER_QUESTION


# ── Collecte pilotee par le code ────────────────────────────────────────


def test_missing_date_is_asked_first():
    d = route_turn(ReservationIntent(), "bonjour", False)
    assert d.action is Action.ASK_SLOT
    assert d.slot == "date"


def test_time_is_asked_once_the_day_is_known():
    intent = ReservationIntent().with_date(date(2026, 7, 22))
    d = route_turn(intent, "demain", False)
    assert d.slot == "time"


def test_party_size_then_name():
    intent = ReservationIntent().with_date(date(2026, 7, 22)).with_time(time(12))
    assert route_turn(intent, "midi", False).slot == "party_size"
    intent = intent.with_party_size(2)
    assert route_turn(intent, "deux", False).slot == "customer_name"


def test_a_known_slot_is_never_asked_again():
    """Le defaut central : le modele redemandait l'heure en la connaissant."""
    intent = _complet()
    d = route_turn(intent, "voila", False)
    assert d.action is not Action.ASK_SLOT


# ── Confirmation et reservation ─────────────────────────────────────────


def test_complete_intent_asks_for_confirmation():
    d = route_turn(_complet(), "Dupont", False)
    assert d.action is Action.CONFIRM


def test_agreement_while_awaiting_confirmation_books():
    d = route_turn(_complet(), "oui c'est ça", True)
    assert d.action is Action.BOOK


def test_refusal_while_awaiting_confirmation_reopens():
    d = route_turn(_complet(), "non, pas du tout", True)
    assert d.action is Action.CORRECT


def test_agreement_without_pending_confirmation_does_not_book():
    d = route_turn(_complet(), "oui", False)
    assert d.action is not Action.BOOK


def test_incomplete_intent_never_books():
    intent = ReservationIntent().with_date(date(2026, 7, 22))
    d = route_turn(intent, "oui je confirme", True)
    assert d.action is not Action.BOOK


# ── Formulations naturelles et variees ──────────────────────────────────


def test_slot_questions_are_natural_french():
    from hikky.domain.question_router import phrase_for_slot

    for slot in ("date", "time", "party_size", "customer_name"):
        q = phrase_for_slot(slot, set())
        assert q.endswith("?"), q
        assert len(q.split()) <= 12, q


def test_phrasing_varies_between_turns():
    """Une formulation figee sonne robotique au bout de deux appels."""
    from hikky.domain.question_router import phrase_for_slot

    vues = {phrase_for_slot("time", set()) for _ in range(30)}
    assert len(vues) > 1, "formulation unique"


def test_phrasing_avoids_recently_used_wording():
    from hikky.domain.question_router import phrase_for_slot

    premiere = phrase_for_slot("time", set())
    seconde = phrase_for_slot("time", {premiere})
    assert seconde != premiere

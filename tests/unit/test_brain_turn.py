"""Orchestration d'un tour pilote par le ConversationBrain.

Le Python ne decide plus quoi dire : il applique ce que le modele a
compris (slots), efface ce qu'il conteste (clear), et n'execute que les
actions demandees (action). La barriere de confirmation reste cote code
— une reservation ne doit jamais dependre du seul jugement du LLM.
"""

from datetime import date, datetime, time

from hikky.domain.brain_turn import run_brain_turn
from hikky.domain.conversation_brain import BrainDecision
from hikky.domain.outcomes import CallOutcome
from hikky.domain.reservation_intent import ReservationIntent
from hikky.domain.restaurant_context import (
    OpeningHours,
    RestaurantContext,
    RestaurantRules,
)


def _ctx():
    return RestaurantContext(
        id="r-1",
        name="Le Petit Sud",
        greeting="Bonjour",
        opening_hours=[
            OpeningHours(weekday=i, opens=time(19), closes=time(23)) for i in range(7)
        ],
        total_capacity=40,
        rules=RestaurantRules(max_group_size=10),
        transfer_number="+33100000000",
        fallback_message="Au revoir",
    )


class FakeBrain:
    def __init__(self, decisions) -> None:
        self._decisions = list(decisions)
        self.calls = []

    async def decide(self, *, user_text, intent, context, history):
        self.calls.append((user_text, intent, list(history)))
        return self._decisions.pop(0) if self._decisions else BrainDecision("ok")


class FakeSession:
    def __init__(self, intent=None, available=True) -> None:
        self.context = _ctx()
        self._intent = intent or ReservationIntent()
        self.booked = 0
        self.ended_with = None
        self.available = available
        self.availability_checks = []

    @property
    def intent(self):
        return self._intent

    def apply_slots(self, slots):
        i = self._intent
        if "date" in slots:
            i = i.with_date(slots["date"])
        if "time" in slots:
            i = i.with_time(slots["time"])
        if "party_size" in slots:
            i = i.with_party_size(slots["party_size"])
        if "customer_name" in slots:
            i = i.with_customer_name(slots["customer_name"])
        self._intent = i

    def clear_slots(self, names):
        for n in names:
            self._intent = self._intent.without(n)

    async def check_availability(self, when, size):
        self.availability_checks.append((when, size))
        return self.available

    async def book(self, phone):
        self.booked += 1
        return CallOutcome.RESERVATION_CREATED

    async def end_with(self, outcome):
        self.ended_with = outcome


async def _run(session, brain, text, spoken, history=None):
    return await run_brain_turn(
        session=session,
        brain=brain,
        user_text=text,
        customer_phone=None,
        history=history if history is not None else [],
        speak=lambda t: spoken.append(t) or _noop(),
    )


async def _noop():
    return None


# ── Application de ce que le modele a compris ───────────────────────────


async def test_reply_is_spoken():
    spoken = []
    await _run(FakeSession(), FakeBrain([BrainDecision("Pour combien ?")]), "x", spoken)
    assert spoken == ["Pour combien ?"]


async def test_slots_are_applied_to_the_intent():
    session = FakeSession()
    brain = FakeBrain([BrainDecision("ok", slots={"date": date(2026, 7, 22),
                                                  "party_size": 4})])
    await _run(session, brain, "demain pour 4", [])
    assert session.intent.date == date(2026, 7, 22)
    assert session.intent.party_size == 4


async def test_contested_slots_are_cleared():
    session = FakeSession(ReservationIntent().with_customer_name("Medica"))
    brain = FakeBrain([BrainDecision("Pardon, votre nom ?", clear=["customer_name"])])
    await _run(session, brain, "je n'ai jamais dit ça", [])
    assert session.intent.customer_name is None


async def test_history_grows_with_each_turn():
    session = FakeSession()
    brain = FakeBrain([BrainDecision("Bonjour !")])
    history = []
    await _run(session, brain, "salut", [], history=history)
    assert {"role": "user", "content": "salut"} in history
    # « Bonjour ! » seul laisserait le client sans savoir quoi dire : le
    # code y adjoint la question sur l'information manquante.
    dit = [m for m in history if m["role"] == "assistant"][0]["content"]
    assert dit.startswith("Bonjour !")
    assert "?" in dit


# ── Actions ─────────────────────────────────────────────────────────────


async def test_availability_is_checked_when_requested():
    session = FakeSession(
        ReservationIntent().with_date(date(2026, 7, 22)).with_time(time(20)),
    )
    brain = FakeBrain([BrainDecision("Je vérifie", action="check_availability",
                                     slots={"party_size": 4})])
    await _run(session, brain, "20h pour 4", [])
    assert session.availability_checks == [(datetime(2026, 7, 22, 20), 4)]


async def test_availability_is_not_checked_without_a_slot():
    session = FakeSession()
    brain = FakeBrain([BrainDecision("Je vérifie", action="check_availability")])
    await _run(session, brain, "x", [])
    assert session.availability_checks == []


async def test_booking_requires_a_complete_intent():
    session = FakeSession()
    brain = FakeBrain([BrainDecision("C'est noté", action="book")])
    decision = await _run(session, brain, "oui", [])
    assert session.booked == 0, "reserver sur une intention incomplete"
    assert decision.should_end is False


async def test_booking_happens_when_the_model_asks_and_the_intent_is_complete():
    session = FakeSession(
        ReservationIntent()
        .with_date(date(2026, 7, 22))
        .with_time(time(20))
        .with_party_size(4)
        .with_customer_name("Dupont")
    )
    brain = FakeBrain([BrainDecision("C'est noté !", action="book")])
    decision = await _run(session, brain, "oui c'est ça", [])
    assert session.booked == 1
    assert decision.should_end is True


async def test_transfer_ends_the_call():
    session = FakeSession()
    brain = FakeBrain([BrainDecision("Je vous passe quelqu'un", action="transfer")])
    decision = await _run(session, brain, "je veux un humain", [])
    assert decision.should_end is True
    assert session.ended_with == CallOutcome.TRANSFERRED


async def test_plain_turn_keeps_the_call_open():
    session = FakeSession()
    brain = FakeBrain([BrainDecision("Pour quand ?")])
    assert (await _run(session, brain, "bonjour", [])).should_end is False


# ── Extraction fiable en parallele du conversationnel ───────────────────
#
# Appel reel : le modele conversait correctement mais cessait d'emettre
# les slots a mesure que l'historique grandissait. Il annoncait « demain
# a 12:45 » alors que le systeme ne contenait rien, et aucune reservation
# n'aurait ete creee. L'extracteur dedie, lui, restait fiable : prompt
# focalise, pas d'historique, une seule tache.

class FakeExtractor:
    def __init__(self, slots_by_text=None) -> None:
        self._slots = slots_by_text or {}
        self.calls = []

    async def extract(self, user_text):
        self.calls.append(user_text)
        return dict(self._slots.get(user_text, {}))


async def test_extractor_slots_are_applied_even_when_the_brain_emits_none():
    session = FakeSession()
    brain = FakeBrain([BrainDecision("D'accord, demain à 12h45.")])  # aucun slot
    extractor = FakeExtractor({"demain 12h45": {"date": date(2026, 7, 22),
                                                "time": time(12, 45)}})
    await run_brain_turn(
        session=session, brain=brain, user_text="demain 12h45",
        customer_phone=None, history=[], extractor=extractor,
        speak=lambda t: _noop(),
    )
    assert session.intent.date == date(2026, 7, 22)
    assert session.intent.time == time(12, 45)


async def test_extractor_wins_over_the_brain():
    """L'extracteur est la source de verite. Le conversationnel a montre
    qu'il inventait des valeurs ; ses slots sont ignores."""
    session = FakeSession()
    brain = FakeBrain([BrainDecision("ok", slots={"party_size": 6})])
    extractor = FakeExtractor({"x": {"party_size": 2}})
    await run_brain_turn(
        session=session, brain=brain, user_text="x", customer_phone=None,
        history=[], extractor=extractor, speak=lambda t: _noop(),
    )
    assert session.intent.party_size == 2


async def test_extractor_is_optional():
    session = FakeSession()
    brain = FakeBrain([BrainDecision("ok", slots={"party_size": 3})])
    await run_brain_turn(
        session=session, brain=brain, user_text="x", customer_phone=None,
        history=[], speak=lambda t: _noop(),
    )
    assert session.intent.party_size == 3


async def test_brain_receives_the_intent_already_enriched_by_the_extractor():
    """Le conversationnel doit voir ce que l'extracteur vient de comprendre,
    sinon il redemande une information qu'il possede deja."""
    session = FakeSession()
    brain = FakeBrain([BrainDecision("ok")])
    extractor = FakeExtractor({"pour 4": {"party_size": 4}})
    await run_brain_turn(
        session=session, brain=brain, user_text="pour 4", customer_phone=None,
        history=[], extractor=extractor, speak=lambda t: _noop(),
    )
    _text, intent_seen, _hist = brain.calls[0]
    assert intent_seen.party_size == 4


async def test_brain_slots_are_ignored_when_an_extractor_is_present():
    """Le conversationnel a invente « party_size: 2 » sur une phrase qui
    ne mentionnait aucun nombre. L'etat appartient a l'extracteur seul.
    """
    session = FakeSession()
    brain = FakeBrain([BrainDecision("ok", slots={"party_size": 2})])
    extractor = FakeExtractor({"une table pour demain midi": {"date": date(2026, 7, 22)}})
    await run_brain_turn(
        session=session, brain=brain, user_text="une table pour demain midi",
        customer_phone=None, history=[], extractor=extractor,
        speak=lambda t: _noop(),
    )
    assert session.intent.party_size is None, "hallucination du conversationnel appliquee"
    assert session.intent.date == date(2026, 7, 22)


async def test_repeated_reply_is_replaced_by_a_targeted_question():
    """Le bot a repete « demain mercredi 22 juillet a midi ? » trois fois."""
    session = FakeSession()
    brain = FakeBrain([BrainDecision("demain à midi ?"), BrainDecision("demain à midi ?")])
    spoken = []
    history = []
    await _run(session, brain, "a", spoken, history=history)
    await _run(session, brain, "b", spoken, history=history)
    assert spoken[0] != spoken[1], f"phrase repetee: {spoken}"


async def test_targeted_question_asks_for_a_missing_slot():
    session = FakeSession()
    brain = FakeBrain([BrainDecision("même phrase"), BrainDecision("même phrase")])
    spoken = []
    history = []
    await _run(session, brain, "a", spoken, history=history)
    await _run(session, brain, "b", spoken, history=history)
    second = spoken[1].lower()
    assert any(w in second for w in ("personne", "jour", "heure", "nom"))


# ── L'ecriture en base ne depend pas du modele ──────────────────────────
#
# Appel reel : le client a dit « Oui, je vous la confirme », le modele a
# emis action="confirm" au lieu de "book", et RIEN n'a ete enregistre.
# Le bot a pourtant annonce « Votre reservation est confirmee ». Le
# client se serait presente sans table.

def _complete_session():
    return FakeSession(
        ReservationIntent()
        .with_date(date(2026, 7, 22))
        .with_time(time(12, 45))
        .with_party_size(2)
        .with_customer_name("Ryan")
    )


async def test_agreement_after_a_confirm_request_books_the_table():
    session = _complete_session()
    brain = FakeBrain([
        BrainDecision("Je récapitule… c'est bien cela ?", action="confirm"),
        BrainDecision("C'est enregistré !", action="confirm"),
    ])
    history = []
    await _run(session, brain, "au nom de Ryan", [], history=history)
    decision = await _run(session, brain, "Oui, je vous la confirme", [], history=history)
    assert session.booked == 1, "le client a accepte mais rien n'a ete enregistre"
    assert decision.should_end is True


async def test_agreement_without_a_pending_confirmation_books_nothing():
    session = _complete_session()
    brain = FakeBrain([BrainDecision("Bonjour !")])
    decision = await _run(session, brain, "oui", [], history=[])
    assert session.booked == 0
    assert decision.should_end is False


async def test_refusal_after_a_confirm_request_books_nothing():
    session = _complete_session()
    brain = FakeBrain([
        BrainDecision("C'est bien cela ?", action="confirm"),
        BrainDecision("D'accord, je corrige."),
    ])
    history = []
    await _run(session, brain, "au nom de Ryan", [], history=history)
    decision = await _run(session, brain, "non, pas du tout", [], history=history)
    assert session.booked == 0
    assert decision.should_end is False


async def test_agreement_on_an_incomplete_intent_books_nothing():
    session = FakeSession()
    brain = FakeBrain([
        BrainDecision("C'est bien cela ?", action="confirm"),
        BrainDecision("ok"),
    ])
    history = []
    await _run(session, brain, "x", [], history=history)
    decision = await _run(session, brain, "oui", [], history=history)
    assert session.booked == 0
    assert decision.should_end is False


async def test_confirmation_is_not_armed_on_an_incomplete_intent():
    """Appel reel : le bot a dit « c'est bien cela ? » alors qu'il
    ignorait le nombre de convives et le nom."""
    from hikky.domain.brain_turn import _confirmation_pending

    session = FakeSession()  # intention vide
    brain = FakeBrain([BrainDecision("c'est bien cela ?", action="confirm")])
    await _run(session, brain, "demain midi", [], history=[])
    assert _confirmation_pending(session) is False


async def test_confirmation_is_armed_when_everything_is_known():
    from hikky.domain.brain_turn import _confirmation_pending

    session = _complete_session()
    brain = FakeBrain([BrainDecision("c'est bien cela ?", action="confirm")])
    await _run(session, brain, "au nom de Ryan", [], history=[])
    assert _confirmation_pending(session) is True


# ── Le bot ne doit jamais annoncer une reservation inexistante ──────────
#
# Appel reel : le modele n'avait jamais demande le nom, a dit « nous vous
# reservons », puis « votre reservation est confirmee ». Le code avait
# correctement refuse d'ecrire en base — mais la phrase mensongere est
# quand meme partie. Le client raccroche en croyant avoir une table.

async def test_refused_booking_never_announces_a_confirmation():
    session = FakeSession(
        ReservationIntent()
        .with_date(date(2026, 7, 22))
        .with_time(time(12))
        .with_party_size(2)
    )  # nom manquant
    brain = FakeBrain([BrainDecision("Votre réservation est confirmée.", action="book")])
    spoken = []
    await _run(session, brain, "oui", spoken)
    assert session.booked == 0
    said = spoken[0].lower()
    assert "confirmée" not in said, f"annonce mensongere: {spoken[0]}"
    assert "nom" in said, f"devrait reclamer le nom manquant: {spoken[0]}"


async def test_refused_booking_asks_for_the_first_missing_slot():
    session = FakeSession()  # tout manque
    brain = FakeBrain([BrainDecision("C'est enregistré !", action="book")])
    spoken = []
    await _run(session, brain, "oui", spoken)
    assert "jour" in spoken[0].lower()


async def test_successful_booking_still_speaks_the_model_reply():
    session = _complete_session()
    brain = FakeBrain([BrainDecision("C'est noté, à demain !", action="book")])
    spoken = []
    await _run(session, brain, "oui", spoken)
    assert session.booked == 1
    assert spoken[0] == "C'est noté, à demain !"


async def test_successful_booking_is_logged():
    """L'operation la plus critique du systeme ne laissait aucune trace :
    impossible de verifier apres coup si une reservation existe."""
    import logging

    from hikky.domain import brain_turn as bt

    records = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    handler = _Capture()
    bt.logger.addHandler(handler)
    try:
        session = _complete_session()
        brain = FakeBrain([BrainDecision("C'est noté !", action="book")])
        await _run(session, brain, "oui", [])
    finally:
        bt.logger.removeHandler(handler)

    assert any("RESERVATION" in m.upper() for m in records), records


# ── Le recapitulatif de confirmation appartient au code ─────────────────
#
# Appel reel : « Puis-je confirmer votre reservation ? » sans dire QUOI,
# puis un recapitulatif qui OUBLIE le nom. Le client a du demander trois
# fois, puis a raccroche sans reservation. Le plafond de 15 mots a pousse
# le modele vers des messages qui ne disent rien.

async def test_confirmation_recap_is_generated_by_the_code():
    session = _complete_session()
    brain = FakeBrain([BrainDecision("Puis-je confirmer ?", action="confirm")])
    spoken = []
    await _run(session, brain, "au nom de Ryan", spoken)
    recap = spoken[0].lower()
    assert "ryan" in recap, f"le nom manque : {spoken[0]}"
    assert "22" in recap or "juillet" in recap, f"la date manque : {spoken[0]}"
    assert "2" in recap or "deux" in recap, f"le nombre manque : {spoken[0]}"


async def test_confirmation_recap_asks_for_agreement():
    session = _complete_session()
    brain = FakeBrain([BrainDecision("ok", action="confirm")])
    spoken = []
    await _run(session, brain, "x", spoken)
    assert "?" in spoken[0], f"pas de question : {spoken[0]}"


async def test_model_reply_is_kept_when_no_confirmation_is_asked():
    session = FakeSession()
    brain = FakeBrain([BrainDecision("C'est à quel nom ?")])
    spoken = []
    await _run(session, brain, "x", spoken)
    assert spoken[0] == "C'est à quel nom ?"


async def test_empty_acknowledgement_is_replaced_by_a_useful_question():
    """« Parfait. » ne fait rien avancer : le client ne sait pas quoi dire."""
    session = FakeSession()
    brain = FakeBrain([BrainDecision("Parfait.")])
    spoken = []
    await _run(session, brain, "deux personnes", spoken)
    assert "?" in spoken[0], f"reponse sans question : {spoken[0]}"

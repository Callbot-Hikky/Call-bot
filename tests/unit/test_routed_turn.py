"""Le tour complet : code aux commandes, modele pour l'intelligence."""

from datetime import date, time

from hikky.domain.outcomes import CallOutcome
from hikky.domain.reservation_intent import ReservationIntent
from hikky.domain.restaurant_context import (
    OpeningHours,
    RestaurantContext,
    RestaurantRules,
)
from hikky.domain.routed_turn import run_routed_turn


def _ctx():
    return RestaurantContext(
        id="r-1",
        name="Le Petit Sud",
        greeting="Bonjour",
        opening_hours=[
            OpeningHours(weekday=i, opens=time(9), closes=time(23)) for i in range(6)
        ],
        total_capacity=40,
        rules=RestaurantRules(max_group_size=10),
        transfer_number="+33100000000",
        fallback_message="Au revoir",
    )


class FakeSession:
    def __init__(self, intent=None) -> None:
        self.context = _ctx()
        self._intent = intent or ReservationIntent()
        self.booked = 0
        self.cleared = []

    @property
    def intent(self):
        return self._intent

    def apply_slots(self, slots):
        i = self._intent
        for k, v in slots.items():
            i = getattr(i, f"with_{k}")(v)
        self._intent = i

    def clear_slots(self, names):
        self.cleared.extend(names)
        for n in names:
            self._intent = self._intent.without(n)

    async def book(self, phone):
        self.booked += 1
        return CallOutcome.RESERVATION_CREATED

    async def end_with(self, outcome):
        self.ended = outcome


class FakeExtractor:
    def __init__(self, par_texte=None) -> None:
        self._m = par_texte or {}

    async def extract(self, texte):
        return dict(self._m.get(texte, {}))


class FakeAnswerer:
    """Le LLM, sollicite uniquement pour les questions hors script."""

    def __init__(self, reponse="Nous sommes ouverts tous les jours.") -> None:
        self.reponse = reponse
        self.appels = []

    async def answer(self, *, user_text, intent, context, history):
        self.appels.append(user_text)
        return self.reponse


async def _run(session, texte, dits, extractor=None, answerer=None, attente=False):
    return await run_routed_turn(
        session=session,
        user_text=texte,
        customer_phone=None,
        history=[],
        extractor=extractor or FakeExtractor(),
        answerer=answerer or FakeAnswerer(),
        awaiting_confirmation=attente,
        speak=lambda t: dits.append(t) or _noop(),
    )


async def _noop():
    return None


# ── L'intelligence est preservee ────────────────────────────────────────


async def test_client_question_reaches_the_model():
    a = FakeAnswerer("Nous sommes fermés le dimanche.")
    dits = []
    await _run(FakeSession(), "Vous êtes ouverts le dimanche ?", dits, answerer=a)
    assert a.appels == ["Vous êtes ouverts le dimanche ?"]
    # Appel d'information pur (aucun creneau engage) : on repond, point.
    # Observe en prod : « a chaque fin de reponse l'IA me casse la tete pour
    # reserver » — la relance systematique est supprimee.
    assert dits[0] == "Nous sommes fermés le dimanche."


async def test_a_question_during_a_booking_resumes_the_missing_slot():
    """Reservation en cours : apres la reponse, on reprend sur ce qui manque."""
    a = FakeAnswerer("Oui, nous avons une terrasse.")
    s = FakeSession(ReservationIntent().with_date(date(2026, 7, 22)).with_time(time(12)))
    dits = []
    await _run(s, "Vous avez une terrasse ?", dits, answerer=a)
    assert dits[0].startswith("Oui, nous avons une terrasse.")
    assert dits[0].endswith("?") and "personne" in dits[0].lower(), dits[0]


def test_no_relance_twice_in_a_row():
    """Si notre phrase precedente etait deja une relance, on repond seulement."""
    from hikky.domain.routed_turn import _ensure_progress
    intent = ReservationIntent().with_date(date(2026, 7, 22)).with_time(time(12))
    history = [
        {"role": "user", "content": "vous avez une terrasse"},
        {"role": "assistant", "content": "Oui. Vous serez combien de personnes ?"},
    ]
    assert _ensure_progress("Nous sommes au centre-ville.", intent, set(), history) == "Nous sommes au centre-ville."
    # ... mais sans relance precedente, on reprend bien.
    assert _ensure_progress("Nous sommes au centre-ville.", intent, set(), []).endswith("?")


async def test_the_model_is_not_called_on_an_ordinary_turn():
    """C'est la moitie de la latence : plus d'appel conversationnel."""
    a = FakeAnswerer()
    await _run(FakeSession(), "deux personnes", [], answerer=a)
    assert a.appels == [], "appel LLM inutile sur un tour nominal"


# ── La collecte est fiable ──────────────────────────────────────────────


async def test_extracted_slots_are_applied():
    s = FakeSession()
    e = FakeExtractor({"demain à midi": {"date": date(2026, 7, 22), "time": time(12)}})
    await _run(s, "demain à midi", [], extractor=e)
    assert s.intent.date == date(2026, 7, 22)
    assert s.intent.time == time(12)


async def test_a_known_slot_is_never_asked_again():
    s = FakeSession(ReservationIntent().with_date(date(2026, 7, 22)).with_time(time(12)))
    dits = []
    await _run(s, "voilà", dits)
    assert "heure" not in dits[0].lower(), dits[0]
    assert "jour" not in dits[0].lower(), dits[0]


async def test_the_next_missing_slot_is_asked():
    dits = []
    await _run(FakeSession(), "bonjour", dits)
    assert "jour" in dits[0].lower()


# ── Confirmation et reservation ─────────────────────────────────────────


def _complet():
    return (
        ReservationIntent()
        .with_date(date(2026, 7, 22))
        .with_time(time(12))
        .with_party_size(2)
        .with_customer_name("Dupont")
    )


async def test_complete_intent_produces_a_full_recap():
    dits = []
    await _run(FakeSession(_complet()), "Dupont", dits)
    r = dits[0].lower()
    assert "dupont" in r and "juillet" in r and "?" in r


async def test_agreement_books_and_ends():
    s = FakeSession(_complet())
    d = await _run(s, "oui c'est ça", [], attente=True)
    assert s.booked == 1
    assert d.should_end is True


async def test_refusal_reopens_without_booking():
    s = FakeSession(_complet())
    d = await _run(s, "non pas du tout", [], attente=True)
    assert s.booked == 0
    assert d.should_end is False


async def test_incomplete_intent_never_books():
    s = FakeSession(ReservationIntent().with_date(date(2026, 7, 22)))
    d = await _run(s, "oui je confirme", [], attente=True)
    assert s.booked == 0
    assert d.should_end is False


async def test_every_reply_ends_with_a_question_until_booking():
    """« Parfait. » laissait le client sans savoir quoi dire."""
    dits = []
    await _run(FakeSession(), "deux personnes", dits)
    assert dits[0].endswith("?")


async def test_group_over_cap_is_refused_clearly_without_booking():
    """Au-dela du plafond (max_group_size = 10), on refuse clairement et on ne
    propose PAS une autre heure — le probleme vient de la taille, pas du creneau.
    """
    dits: list[str] = []
    session = FakeSession()
    extractor = FakeExtractor({"on sera 12": {"party_size": 12}})
    outcome = await _run(session, "on sera 12", dits, extractor=extractor)
    assert session.booked == 0
    assert dits and "plus de 10 personnes" in dits[-1]
    assert "autre heure" not in dits[-1]
    assert outcome.should_end is False


# ── Appel du 2026-10-03 : « À quel nom ? » — « Royal. » — « Puis-je avoir votre nom ? »


class FakePhraseur:
    def __init__(self) -> None:
        self.appels = []

    async def formuler(self, intention):
        self.appels.append(intention.slot)
        return intention.repli


def _sans_nom():
    return FakeSession(
        ReservationIntent()
        .with_date(date(2026, 10, 4))
        .with_time(time(21))
        .with_party_size(4)
    )


async def test_un_mot_seul_en_reponse_a_la_demande_du_nom_est_le_nom():
    s = _sans_nom()
    dits = []
    await _run(s, "Royal.", dits)  # l'extracteur LLM n'a rien vu : c'est le code qui retient
    assert s.intent.customer_name == "Royal"
    # Tout est connu : le bot récapitule au lieu de redemander le nom.
    assert "c'est bien cela" in dits[-1].lower()
    assert "royal" in dits[-1].lower()


async def test_le_nom_court_est_nettoye_des_formules():
    s = _sans_nom()
    await _run(s, "au nom de Dupont", [])
    assert s.intent.customer_name == "Dupont"


async def test_une_reponse_courte_qui_n_est_pas_un_nom_n_est_pas_retenue():
    for texte in ("oui", "vingt-deux heures", "demain", "pardon ?", "4 personnes"):
        s = _sans_nom()
        await _run(s, texte, [])
        assert s.intent.customer_name is None, texte


async def test_le_nom_n_est_pas_devine_quand_un_autre_slot_manque():
    s = FakeSession(ReservationIntent().with_date(date(2026, 10, 4)))
    await _run(s, "Royal.", [])
    assert s.intent.customer_name is None


async def test_le_phraseur_n_est_pas_sollicite_sans_rien_a_accuser():
    """Rien d'extrait (phrase incomprise) : attendre 1,2 s le modèle pour retomber
    sur la phrase figée est du temps perdu — observé 2× sur un appel."""
    p = FakePhraseur()
    s = FakeSession(ReservationIntent().with_date(date(2026, 10, 4)))
    await run_routed_turn(
        session=s, user_text="Vingt-deux et neuf.", customer_phone=None, history=[],
        extractor=FakeExtractor(), answerer=FakeAnswerer(), awaiting_confirmation=False,
        speak=lambda t: _noop(), phraseur=p,
    )
    assert p.appels == []


async def test_le_phraseur_est_sollicite_quand_il_y_a_quelque_chose_a_accuser():
    p = FakePhraseur()
    s = FakeSession(ReservationIntent())
    e = FakeExtractor({"demain soir": {"date": date(2026, 10, 4)}})
    await run_routed_turn(
        session=s, user_text="demain soir", customer_phone=None, history=[],
        extractor=e, answerer=FakeAnswerer(), awaiting_confirmation=False,
        speak=lambda t: _noop(), phraseur=p,
    )
    assert p.appels == ["time"]

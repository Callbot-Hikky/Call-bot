"""Correctifs de dialogue : ne pas inventer l'heure, filtre STT, dispo avant
le nom, liste des créneaux, mention du SMS."""

from datetime import date, time

from hikky.adapters.voice.faster_whisper_stt import _is_hallucination
from hikky.domain.outcomes import CallOutcome
from hikky.domain.reservation_intent import ReservationIntent
from hikky.domain.restaurant_context import (
    OpeningHours,
    RestaurantContext,
    RestaurantRules,
)
from hikky.domain.routed_turn import (
    _demande_de_creneaux,
    _lister_creneaux,
    run_routed_turn,
)
from hikky.pipeline.llm_slot_extractor import _drop_invented_time, _resolve_weekday_date


# ── #1 : ne pas inventer l'heure depuis un moment vague ──────────────────


def test_ce_soir_seul_retire_l_heure_inventee():
    assert _drop_invented_time({"time": time(18, 0)}, "je réserve pour ce soir") == {}


def test_ce_soir_avec_heure_precise_garde_l_heure():
    out = _drop_invented_time({"time": time(20, 0)}, "ce soir à 20h")
    assert out == {"time": time(20, 0)}


def test_midi_est_une_vraie_heure_conservee():
    out = _drop_invented_time({"time": time(12, 0)}, "à midi")
    assert out == {"time": time(12, 0)}


def test_heure_explicite_sans_moment_vague_conservee():
    out = _drop_invented_time({"time": time(20, 0)}, "à vingt heures")
    assert out == {"time": time(20, 0)}


# ── #1bis : résolution déterministe des jours de la semaine ──────────────
#
# Constaté en réel : un lundi (2026-09-14), « jeudi » a été résolu par le LLM
# en 2026-09-18 (un vendredi) au lieu du 17. Les modèles calculent mal les
# dates ; on résout le jour nommé de façon déterministe, en code.

from datetime import datetime


def _lundi():
    return datetime(2026, 9, 14, 10, 0)  # lundi


def test_jeudi_donne_le_bon_jeudi_pas_le_vendredi():
    out = _resolve_weekday_date({"time": time(20, 0)}, "jeudi à 20h", _lundi())
    assert out["date"] == date(2026, 9, 17)  # jeudi, pas le 18


def test_mercredi_et_vendredi_sont_corrects():
    assert _resolve_weekday_date({}, "mercredi", _lundi())["date"] == date(2026, 9, 16)
    assert _resolve_weekday_date({}, "vendredi", _lundi())["date"] == date(2026, 9, 18)


def test_sans_jour_nomme_la_date_du_llm_est_conservee():
    # « demain » n'est pas un jour nommé : on ne touche pas à la date extraite.
    ext = {"date": date(2026, 9, 15)}
    assert _resolve_weekday_date(ext, "demain", _lundi()) == ext


def test_jour_nomme_corrige_une_date_llm_erronee():
    # Le LLM a mis le 18 (vendredi) pour « jeudi » : on écrase par le 17.
    out = _resolve_weekday_date({"date": date(2026, 9, 18)}, "pour jeudi", _lundi())
    assert out["date"] == date(2026, 9, 17)


# ── #2 : filtre anti-hallucination STT ───────────────────────────────────


def test_hallucinations_whisper_detectees():
    assert _is_hallucination("Sous-titrage MFP")
    assert _is_hallucination("Merci d'avoir regardé cette vidéo")
    assert _is_hallucination("   ")


def test_vraie_phrase_non_filtree():
    assert not _is_hallucination("Quatre personnes pour demain midi")


def test_boucle_de_repetition_est_filtree():
    """Sur l'écho de sa propre voix, Whisper boucle : « Bonne nuit. Bonne
    nuit. Bonne nuit… » ×12. Ces répétitions à très faible diversité sont des
    hallucinations, pas de la parole — on les jette."""
    assert _is_hallucination("Bonne nuit. " * 12)
    assert _is_hallucination("les les les les les les les les")


def test_repetition_courte_legitime_non_filtree():
    """« oui oui », « non non merci » restent de vraies réponses courtes."""
    assert not _is_hallucination("oui oui")
    assert not _is_hallucination("non non merci")
    assert not _is_hallucination("Je voudrais réserver une table pour ce soir")


class _Seg:
    def __init__(self, text):
        self.text = text


class _FakeWhisper:
    def transcribe(self, audio, **kwargs):
        return ([_Seg("Sous-titrage MFP"), _Seg("Bonjour")], None)


async def test_stream_filtre_hallucination_sans_crash():
    """Le chemin _stream (avec log) doit filtrer sans lever d'erreur."""
    from hikky.adapters.voice.faster_whisper_stt import FasterWhisperSTTAdapter

    adapter = FasterWhisperSTTAdapter(device="cpu")
    adapter._model = _FakeWhisper()  # injecte un modèle bidon

    async def _chunks():
        yield (b"\x00\x00") * 200

    out = [t async for t in await adapter.transcribe(_chunks())]
    assert out == ["Bonjour"]  # l'hallucination est jetée, pas de NameError


# ── #4 : détection + liste des créneaux ──────────────────────────────────


def test_detection_demande_creneaux():
    assert _demande_de_creneaux("quelles sont les heures disponibles ?")
    assert _demande_de_creneaux("vous avez de la place ce soir ?")
    assert not _demande_de_creneaux("je voudrais réserver")


def _contexte(max_group=15):
    return RestaurantContext(
        id="r-1",
        name="Le Petit Sud",
        greeting="Bonjour",
        opening_hours=[
            OpeningHours(weekday=i, opens=time(12), closes=time(14, 30))
            for i in range(7)
        ]
        + [OpeningHours(weekday=i, opens=time(19), closes=time(23)) for i in range(7)],
        total_capacity=40,
        rules=RestaurantRules(max_group_size=max_group),
        transfer_number=None,
        fallback_message="Au revoir",
    )


class _Intent:
    def __init__(self, jour=None):
        self.date = jour


def test_liste_creneaux_du_jour():
    class _S:
        context = _contexte()
        intent = _Intent(jour=date(2026, 9, 14))

    texte = _lister_creneaux(_S())
    assert "de midi à 14 heures 30" in texte
    assert "de 19 heures à 23 heures" in texte


def test_liste_creneaux_sans_jour_demande_le_jour():
    class _S:
        context = _contexte()
        intent = _Intent(jour=None)

    assert "quel jour" in _lister_creneaux(_S()).lower()


# ── #3 : vérification de dispo AVANT de demander le nom ───────────────────


class _Dispo:
    def __init__(self, available, reason=None, alternatives=None):
        self.available = available
        self.reason = reason
        self.alternatives = alternatives or []


class _SessionDispo:
    """Session avec date+heure+nombre déjà remplis, nom manquant."""

    def __init__(self, dispo):
        self.context = _contexte()
        self._intent = (
            ReservationIntent()
            .with_date(date(2026, 9, 14))
            .with_time(time(20, 0))
            .with_party_size(4)
        )
        self._dispo = dispo
        self._sig = None
        self.cleared: list[str] = []
        self.booked = 0

    @property
    def intent(self):
        return self._intent

    def apply_slots(self, slots):
        for k, v in slots.items():
            self._intent = getattr(self._intent, f"with_{k}")(v)
        self._sig = None

    def clear_slots(self, names):
        self.cleared.extend(names)
        for n in names:
            self._intent = self._intent.without(n)
        self._sig = None

    async def availability_detail(self, when, party):
        return self._dispo

    def availability_confirmed_for(self, when, party):
        return self._sig == (when, party)

    def mark_availability_ok(self, when, party):
        self._sig = (when, party)

    async def book(self, phone):
        self.booked += 1
        return CallOutcome.RESERVATION_CREATED


async def _noop():
    return None


class _Extractor:
    async def extract(self, texte):
        return {}


async def _run(session, texte, dits, attente=False):
    return await run_routed_turn(
        session=session,
        user_text=texte,
        customer_phone=None,
        history=[],
        extractor=_Extractor(),
        answerer=None,
        awaiting_confirmation=attente,
        speak=lambda t: dits.append(t) or _noop(),
    )


async def test_indispo_avant_nom_propose_autre_heure_et_vide_l_heure():
    session = _SessionDispo(_Dispo(available=False, reason="no_table"))
    dits: list[str] = []
    outcome = await _run(session, "voilà", dits)
    assert session.booked == 0
    assert "time" in session.cleared  # l'heure est effacée pour en redemander une
    assert dits and "complet" in dits[-1].lower()
    assert outcome.should_end is False


async def test_dispo_ok_laisse_demander_le_nom():
    session = _SessionDispo(_Dispo(available=True))
    dits: list[str] = []
    await _run(session, "voilà", dits)
    # Créneau libre → on ne vide pas l'heure, on demande le nom.
    assert "time" not in session.cleared
    assert dits and ("nom" in dits[-1].lower())


async def test_indispo_annoncee_avant_le_recap_meme_si_nom_deja_donne():
    """Nom donné tôt → tous les slots sont là → route CONFIRM. La dispo doit
    être vérifiée AVANT le récapitulatif, pas seulement au moment de réserver."""
    session = _SessionDispo(_Dispo(available=False, reason="no_table"))
    session.apply_slots({"customer_name": "Ayoub"})  # nom déjà présent → CONFIRM
    dits: list[str] = []
    outcome = await _run(session, "oui c'est ça", dits)
    assert session.booked == 0
    assert "time" in session.cleared
    assert dits and "complet" in dits[-1].lower()
    # Surtout : le message n'est PAS un récapitulatif.
    assert "récapitule" not in dits[-1].lower()
    assert outcome.should_end is False

"""Le moteur qui remplace le duo extracteur + generateur aveugles.

Un seul appel LLM par tour, qui recoit TOUT le contexte du restaurant et
peut demander des actions. Les deux appels precedents ne voyaient ni les
horaires, ni la capacite, et n'avaient aucun moyen d'agir : le bot ne
savait que reclamer les cases vides, et repetait la meme phrase des que
le client sortait du script.
"""

import json
from datetime import date, datetime, time

import pytest

from hikky.domain.conversation_brain import BrainDecision, ConversationBrain
from hikky.domain.reservation_intent import ReservationIntent
from hikky.domain.restaurant_context import (
    OpeningHours,
    RestaurantContext,
    RestaurantRules,
)

NOW = datetime(2026, 7, 21, 14, 30)


def _ctx() -> RestaurantContext:
    return RestaurantContext(
        id="r-1",
        name="Le Petit Sud",
        greeting="Bonjour",
        opening_hours=[
            OpeningHours(weekday=i, opens=time(19, 0), closes=time(23, 0))
            for i in range(5)
        ],
        total_capacity=40,
        rules=RestaurantRules(max_group_size=10),
        transfer_number="+33100000000",
        fallback_message="Au revoir",
    )


class FakeLLM:
    """Renvoie des reponses scriptees et memorise les prompts recus."""

    def __init__(self, payloads=None) -> None:
        self._payloads = list(payloads or [{}])
        self.prompts = []

    async def complete(self, messages):
        self.prompts.append(messages)
        payload = self._payloads.pop(0) if self._payloads else {}
        return payload if isinstance(payload, str) else json.dumps(payload)


def _brain(llm, ctx=None) -> ConversationBrain:
    return ConversationBrain(llm, clock=lambda: NOW)


async def _decide(llm, text="bonjour", intent=None, ctx=None, history=None):
    return await _brain(llm).decide(
        user_text=text,
        intent=intent or ReservationIntent(),
        context=ctx or _ctx(),
        history=history or [],
    )


# ── Le prompt doit tout contenir ────────────────────────────────────────


async def test_prompt_includes_opening_hours():
    llm = FakeLLM()
    await _decide(llm)
    system = llm.prompts[0][0]["content"]
    assert "19" in system and "23" in system, "horaires absents du prompt"


async def test_prompt_includes_capacity_and_group_limit():
    llm = FakeLLM()
    await _decide(llm)
    system = llm.prompts[0][0]["content"]
    assert "40" in system
    assert "10" in system


async def test_prompt_includes_current_date_and_time():
    llm = FakeLLM()
    await _decide(llm)
    system = llm.prompts[0][0]["content"]
    assert "21 juillet" in system
    assert "14:30" in system


async def test_prompt_includes_known_information():
    llm = FakeLLM()
    intent = ReservationIntent().with_party_size(4).with_customer_name("Dupont")
    await _decide(llm, intent=intent)
    system = llm.prompts[0][0]["content"]
    assert "4" in system and "Dupont" in system


async def test_history_is_replayed_to_the_model():
    llm = FakeLLM()
    history = [
        {"role": "user", "content": "je voudrais reserver"},
        {"role": "assistant", "content": "pour quand ?"},
    ]
    await _decide(llm, history=history)
    flat = " ".join(m["content"] for m in llm.prompts[0])
    assert "je voudrais reserver" in flat and "pour quand ?" in flat


async def test_identifiers_appear_only_in_the_json_schema():
    """Les noms techniques sont necessaires au schema de sortie, mais le
    modele ne doit jamais les prononcer.

    Appel reel : le bot a dit « Customer name, s'il vous plait » parce
    que le prompt listait les slots manquants par leur identifiant.
    """
    llm = FakeLLM()
    await _decide(llm)
    system = llm.prompts[0][0]["content"]
    avant_schema = system.split("RÉPONDS UNIQUEMENT")[0]
    for ident in ("customer_name", "party_size", "date_time"):
        assert ident not in avant_schema, f"{ident} expose hors du schema"
    assert "ne les prononce jamais" in system.lower()


# ── La sortie structuree ────────────────────────────────────────────────


async def test_reply_is_returned():
    llm = FakeLLM([{"reply": "Pour combien de personnes ?"}])
    decision = await _decide(llm)
    assert isinstance(decision, BrainDecision)
    assert decision.reply == "Pour combien de personnes ?"


async def test_slots_are_parsed_into_python_types():
    llm = FakeLLM([{"reply": "ok", "slots": {"date": "2026-07-22", "time": "20:30",
                                             "party_size": 4, "customer_name": "Dupont"}}])
    decision = await _decide(llm)
    assert decision.slots["date"] == date(2026, 7, 22)
    assert decision.slots["time"] == time(20, 30)
    assert decision.slots["party_size"] == 4
    assert decision.slots["customer_name"] == "Dupont"


async def test_vague_time_is_not_invented():
    llm = FakeLLM([{"reply": "À quelle heure ?", "slots": {"date": "2026-07-22"}}])
    decision = await _decide(llm, text="demain matin")
    assert "time" not in decision.slots


async def test_contested_slots_are_reported_for_clearing():
    llm = FakeLLM([{"reply": "Pardon, quel est votre nom ?", "clear": ["customer_name"]}])
    decision = await _decide(llm, text="je n'ai jamais dit ça")
    assert decision.clear == ["customer_name"]


async def test_action_defaults_to_none():
    llm = FakeLLM([{"reply": "ok"}])
    assert (await _decide(llm)).action == "none"


@pytest.mark.parametrize("action", ["check_availability", "confirm", "book", "transfer"])
async def test_known_actions_are_reported(action):
    llm = FakeLLM([{"reply": "ok", "action": action}])
    assert (await _decide(llm)).action == action


async def test_unknown_action_falls_back_to_none():
    llm = FakeLLM([{"reply": "ok", "action": "fais_le_cafe"}])
    assert (await _decide(llm)).action == "none"


# ── Robustesse ──────────────────────────────────────────────────────────


async def test_json_wrapped_in_prose_is_recovered():
    llm = FakeLLM(['Voici : {"reply": "Bonjour !"} — fin'])
    assert (await _decide(llm)).reply == "Bonjour !"


async def test_malformed_json_yields_a_safe_reply():
    llm = FakeLLM(["ceci n'est pas du json"])
    decision = await _decide(llm)
    assert decision.reply
    assert decision.slots == {}
    assert decision.action == "none"


async def test_llm_failure_yields_a_safe_reply():
    class Failing:
        async def complete(self, messages):
            raise RuntimeError("gpu perdu")

    decision = await _decide(Failing())
    assert decision.reply
    assert decision.action == "none"


async def test_bad_slot_values_are_dropped_not_crashing():
    llm = FakeLLM([{"reply": "ok", "slots": {"date": "pas-une-date",
                                             "party_size": -3,
                                             "time": "25:99"}}])
    decision = await _decide(llm)
    assert decision.slots == {}


# ── Restitution orale des dates ─────────────────────────────────────────
#
# Appel reel : le bot a dit « le 2026-07-22 » et « Ce mardi 2026-07-21
# est un mardi ». Le format ISO du prompt ressortait tel quel a l'oral.

async def test_prompt_states_the_date_in_french_not_iso():
    llm = FakeLLM()
    await _decide(llm)
    system = llm.prompts[0][0]["content"]
    assert "2026-07-21" not in system, "format ISO expose au modele"
    assert "21 juillet" in system


async def test_prompt_forbids_speaking_numeric_dates():
    llm = FakeLLM()
    await _decide(llm)
    system = llm.prompts[0][0]["content"].lower()
    assert "jamais" in system and ("chiffres" in system or "iso" in system)


async def test_known_information_is_stated_in_french():
    from datetime import date as _d

    llm = FakeLLM()
    await _decide(llm, intent=ReservationIntent().with_date(_d(2026, 7, 22)))
    system = llm.prompts[0][0]["content"]
    assert "22 juillet" in system
    assert "2026-07-22" not in system


async def test_prompt_forbids_confirming_before_everything_is_known():
    """Appel reel : le bot a dit « nous vous reservons » puis « votre
    reservation est confirmee » sans avoir jamais demande le nom."""
    llm = FakeLLM()
    await _decide(llm)
    system = llm.prompts[0][0]["content"].lower()
    assert "jamais trop tôt" in system or "ne confirme jamais" in system
    assert "nom du client avant de finaliser" in system


async def test_known_time_is_stated_in_words_not_digits():
    """Appel simule : le bot a dit « à 12:00 », que la synthese prononce
    « douze deux points zero zero »."""
    from datetime import time as _t

    llm = FakeLLM()
    await _decide(llm, intent=ReservationIntent().with_time(_t(12, 0)))
    system = llm.prompts[0][0]["content"]
    assert "12:00" not in system, "format numerique expose au modele"
    assert "midi" in system or "12 heures" in system


async def test_half_past_is_stated_in_words():
    from datetime import time as _t

    llm = FakeLLM()
    await _decide(llm, intent=ReservationIntent().with_time(_t(20, 30)))
    system = llm.prompts[0][0]["content"]
    assert "20:30" not in system
    assert "20 heures 30" in system


# ── Concision ───────────────────────────────────────────────────────────
#
# Mesure : « Parfait, la reservation au nom de Dupont pour mercredi 22
# juillet a 12 heures pour deux personnes est-elle correcte ? » = 11,1 s
# de parole. Le cout de synthese, de reechantillonnage et surtout de
# diffusion en temps reel est proportionnel a cette longueur. Le modele
# recite tout l'etat connu a chaque tour, alors que seule la question
# finale fait avancer l'echange.

async def test_prompt_forbids_repeating_the_known_state_every_turn():
    llm = FakeLLM()
    await _decide(llm)
    system = llm.prompts[0][0]["content"].lower()
    assert "ne répète pas" in system or "ne répete pas" in system
    assert "récapitul" in system


async def test_prompt_caps_the_reply_length():
    llm = FakeLLM()
    await _decide(llm)
    system = llm.prompts[0][0]["content"].lower()
    assert "15 mots" in system or "quinze mots" in system


async def test_prompt_shows_a_concise_example():
    llm = FakeLLM()
    await _decide(llm)
    system = llm.prompts[0][0]["content"]
    assert "Vous serez combien" in system or "À quelle heure" in system

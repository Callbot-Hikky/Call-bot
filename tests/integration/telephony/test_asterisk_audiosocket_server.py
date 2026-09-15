"""Tests d'intégration du serveur AudioSocket en mode smoke test.

On lance un vrai serveur asyncio sur `127.0.0.1:0` (port éphémère), on
s'y connecte comme si on était Asterisk, on envoie des paquets, on
vérifie que le serveur se comporte bien.

Ces tests ne nécessitent PAS de vraies dépendances IA — ils valident
uniquement le transport.
"""

from __future__ import annotations

import asyncio
from datetime import time
from uuid import UUID, uuid4

from hikky.adapters.telephony.asterisk_audiosocket_server import (
    AudioSocketServerConfig,
    AudioSocketServerDeps,
    _action_inactivite,
    handle_connection,
    start_server,
)
from hikky.adapters.telephony.audiosocket_protocol import (
    MessageType,
    encode_audio,
    encode_hangup,
)
from hikky.domain.restaurant_context import (
    OpeningHours,
    RestaurantContext,
    RestaurantRules,
)

# ── Fakes ──────────────────────────────────────────────────────────────────


class FakeRestaurantContextPort:
    def __init__(self, ctx: RestaurantContext) -> None:
        self._ctx = ctx
        self.calls: list[str] = []

    async def load(self, called_number: str) -> RestaurantContext:
        self.calls.append(called_number)
        return self._ctx


class FakeCallSession:
    """Session très minimaliste pour valider le cycle bind/unbind/end."""

    def __init__(self, call_id: str, ctx: RestaurantContext) -> None:
        self.call_id = call_id
        self.context = ctx
        self.begun = False
        self.ended_with: object | None = None

    async def begin(self) -> None:
        self.begun = True

    async def end_with(self, outcome: object) -> None:
        self.ended_with = outcome


def _make_ctx() -> RestaurantContext:
    return RestaurantContext(
        id="r-1",
        name="Le Petit Sud",
        greeting="Bonjour",
        opening_hours=[OpeningHours(weekday=0, opens=time(12, 0), closes=time(14, 0))],
        total_capacity=40,
        rules=RestaurantRules(),
        transfer_number=None,
        fallback_message="Réessayez plus tard.",
    )


def _make_deps() -> tuple[AudioSocketServerDeps, list[FakeCallSession]]:
    ctx = _make_ctx()
    sessions: list[FakeCallSession] = []

    def session_factory(call_id: str, ctx_arg: RestaurantContext) -> FakeCallSession:
        s = FakeCallSession(call_id, ctx_arg)
        sessions.append(s)
        return s

    port = FakeRestaurantContextPort(ctx)
    deps = AudioSocketServerDeps(
        session_factory=session_factory,  # type: ignore[arg-type]
        restaurant_context_port=port,  # type: ignore[arg-type]
    )
    return deps, sessions


# ── Relance sur inactivité ──────────────────────────────────────────────────
#
# Symptôme réel : après une question du bot, si la réponse du client n'est pas
# captée (réponse courte écartée, écho nettoyé par le VAD), aucun tour n'est
# produit et le bot reste muet indéfiniment — « au bout d'un moment je n'entends
# plus rien ». La boucle doit relancer puis clore, au lieu d'attendre un tour
# qui ne viendra jamais.


def _cfg_inactivite():
    # relances à 10, 20, 30 ; clôture à 40.
    return AudioSocketServerConfig(inactivite_relance_s=10, inactivite_max_relances=3)


def test_action_inactivite_ne_fait_rien_avant_le_premier_palier():
    assert _action_inactivite(5.0, relances_faites=0, config=_cfg_inactivite()) == "rien"


def test_action_inactivite_relance_a_chaque_palier():
    cfg = _cfg_inactivite()
    assert _action_inactivite(10.0, relances_faites=0, config=cfg) == "relance"
    assert _action_inactivite(20.0, relances_faites=1, config=cfg) == "relance"
    assert _action_inactivite(30.0, relances_faites=2, config=cfg) == "relance"


def test_action_inactivite_attend_le_palier_suivant_entre_deux_relances():
    cfg = _cfg_inactivite()
    # Déjà relancé une fois : rien tant qu'on n'atteint pas le 2e palier (20 s).
    assert _action_inactivite(15.0, relances_faites=1, config=cfg) == "rien"


def test_action_inactivite_ne_clot_qu_apres_toutes_les_relances():
    cfg = _cfg_inactivite()
    # Max de relances atteint mais pas encore le seuil de fin (40 s) → on patiente.
    assert _action_inactivite(35.0, relances_faites=3, config=cfg) == "rien"
    # Passé le dernier palier sans réponse → clôture.
    assert _action_inactivite(40.0, relances_faites=3, config=cfg) == "fin"


# ── Tests ───────────────────────────────────────────────────────────────────


async def test_server_starts_and_accepts_connection_with_uuid_handshake():
    deps, sessions = _make_deps()
    config = AudioSocketServerConfig(
        host="127.0.0.1", port=0, smoke_test=True, smoke_test_duration_s=1.0
    )
    server, _adapter = await start_server(deps, config)
    port = server.sockets[0].getsockname()[1]

    try:
        # Simule Asterisk : ouvre une connexion et envoie le paquet UUID
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        call_uuid = uuid4()
        packet = bytes([MessageType.UUID_ID, 0x00, 0x10]) + call_uuid.bytes
        writer.write(packet)
        await writer.drain()

        # Puis un chunk audio, puis hangup — le serveur doit terminer
        writer.write(encode_audio(b"\x00" * 320))
        await writer.drain()
        writer.write(encode_hangup())
        await writer.drain()

        # Attendre que le serveur ferme sa moitié
        await asyncio.wait_for(reader.read(), timeout=2.0)

        writer.close()
        await writer.wait_closed()
    finally:
        server.close()
        await server.wait_closed()

    # Vérifications post-mortem
    assert len(sessions) == 1
    session = sessions[0]
    assert session.begun is True
    assert session.call_id == str(call_uuid)
    assert session.ended_with is not None  # end_with a été appelé dans le finally


async def test_server_closes_connection_when_first_packet_is_not_uuid():
    deps, sessions = _make_deps()
    config = AudioSocketServerConfig(
        host="127.0.0.1", port=0, smoke_test=True, smoke_test_duration_s=1.0
    )
    server, _adapter = await start_server(deps, config)
    port = server.sockets[0].getsockname()[1]

    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        # Un paquet audio en 1er au lieu d'un UUID → le serveur doit fermer
        writer.write(encode_audio(b"\x00\x01"))
        await writer.drain()

        # Le serveur ferme sa moitié → read() renvoie b""
        data = await asyncio.wait_for(reader.read(), timeout=2.0)
        assert data == b""

        writer.close()
        await writer.wait_closed()
    finally:
        server.close()
        await server.wait_closed()

    # Pas de session créée, pas de restaurant chargé
    assert sessions == []


async def test_handle_connection_directly_with_streams():
    """Test unitaire de `handle_connection` — plus fin que via le serveur."""
    from hikky.adapters.telephony.asterisk_audiosocket_adapter import (
        AsteriskAudioSocketAdapter,
    )

    deps, sessions = _make_deps()
    config = AudioSocketServerConfig(smoke_test=True, smoke_test_duration_s=0.5)
    adapter = AsteriskAudioSocketAdapter()

    # Fabrique un couple reader/writer via un memory pipe
    call_uuid = UUID("11112222-3333-4444-5555-666677778888")
    packets = (
        bytes([MessageType.UUID_ID, 0x00, 0x10]) + call_uuid.bytes
        + encode_audio(b"\x00" * 320)
        + encode_hangup()
    )

    reader = asyncio.StreamReader()
    reader.feed_data(packets)
    reader.feed_eof()

    class _NullWriter:
        def get_extra_info(self, key):
            return "test-peer"

        def write(self, data):
            pass

        async def drain(self):
            pass

        def close(self):
            pass

        async def wait_closed(self):
            pass

    await handle_connection(
        reader, _NullWriter(), adapter=adapter, deps=deps, config=config  # type: ignore[arg-type]
    )

    assert len(sessions) == 1
    assert sessions[0].call_id == str(call_uuid)
    assert sessions[0].ended_with is not None

"""Faire parler le bot au téléphone.

Le texte part au service TTS ; la voix revient par morceaux (flux) ; chaque morceau
est converti en trames A-law de 20 ms et envoyé à Telnyx au rythme où il les joue.
Premier son ~0,65 s après la décision, mesuré sur chaque phrase dans les journaux.

Deux règles apprises en appel réel :
- Telnyx veut au plus un message `media` par seconde environ : on espace de 1,05 s.
- Quand le client coupe le bot, l'historique ne doit garder que ce qu'il a ENTENDU.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from typing import Any

from . import audio
from .call_state import CallState

log = logging.getLogger("telnyx.speaker")
conv = logging.getLogger("hikky.conversation")

MIN_GAP_BETWEEN_MEDIA_S = 1.05  # Telnyx bufferise et joue lui-même ; 1 message / s max
FRAME_DURATION_S = audio.FRAME_MS / 1000
SPOKEN_CHARS_PER_S = 14  # pour estimer ce que le client a entendu avant de couper


class TelnyxSpeaker:
    """Le haut-parleur d'un appel : un texte entre, de la voix sort vers Telnyx.

    Il ne crée ni le WebSocket ni le client HTTP : on les lui donne (injection de
    dépendances). En production ce sont les vrais ; en test, deux doublures qui
    enregistrent ce qui part, d'où 5 tests sans réseau ni GPU.

    Un `TelnyxSpeaker` par appel, créé dans `server.py` à l'ouverture du WebSocket.
    """

    def __init__(
        self,
        *,
        ws: Any,  # le WebSocket vers Telnyx : on y envoie les messages « media » et « clear »
        http: Any,  # le client HTTP (httpx) : on y appelle le service TTS
        state: CallState,  # le tableau blanc de l'appel : speaking, stop_requested, history…
        tts_url: str,  # TTS « phrase entière » (repli, et salutation mise en cache)
        tts_stream_url: str,  # TTS « en flux » : la voix arrive par morceaux
        streaming: bool,  # vrai = flux (TTS_STREAM=1) ; faux = phrase entière
    ) -> None:
        # Rien d'autre que ranger : aucun appel réseau à la construction.
        self._ws = ws
        self._http = http
        self._state = state
        self._tts_url = tts_url
        self._tts_stream_url = tts_stream_url
        self._streaming = streaming

    # ── bas niveau : des trames vers Telnyx ──────────────────────────────────
    async def send_frames(self, frames: list[bytes]) -> None:
        """Envoie des trames A-law en UN message, puis attend qu'elles soient jouées."""
        if not frames:
            return
        state = self._state
        state.speaking = True
        try:
            loop = asyncio.get_event_loop()
            gap = MIN_GAP_BETWEEN_MEDIA_S - (loop.time() - state.last_send_ts)
            if gap > 0:
                await asyncio.sleep(gap)
            if state.stop_requested:
                return
            payload = base64.b64encode(b"".join(frames)).decode()
            await self._ws.send_text(json.dumps({"event": "media", "media": {"payload": payload}}))
            state.last_send_ts = loop.time()
            # On attend que Telnyx ait joué ces trames — ou qu'une interruption arrive.
            await self._wait_played(len(frames) * FRAME_DURATION_S)
        finally:
            state.speaking = False

    async def _wait_played(self, seconds: float) -> None:
        """Attend la durée de l'audio envoyé, réveillée plus tôt par `interrupt()`."""
        try:
            await asyncio.wait_for(self._state.stop_event.wait(), timeout=seconds)
        except TimeoutError:
            pass

    async def interrupt(self) -> None:
        """Barge-in : arrêter le flux en cours et vider la file audio côté Telnyx."""
        self._state.stop_requested = True
        self._state.stop_event.set()
        await self._ws.send_text(json.dumps({"event": "clear"}))

    # ── haut niveau : une phrase ─────────────────────────────────────────────
    async def say(self, text: str) -> None:
        if not text:
            return
        state = self._state
        # Nouvelle phrase = nouveau droit à la parole. Le drapeau d'interruption n'était
        # remis à zéro qu'au moment d'ENVOYER ; le flux le testait AVANT d'envoyer :
        # après la première interruption, le bot restait muet jusqu'à la fin de l'appel.
        state.stop_requested = False
        state.stop_event.clear()
        conv.info("BOT    : %s", text)
        if not self._streaming:
            pcm, rate = await self.synthesize(text)
            if not state.stop_requested:
                await self.send_frames(audio.pcm_to_telephony_frames(pcm, rate))
            return
        n_frames = await self._say_streaming(text)
        if state.stop_requested:
            self._truncate_history(text, n_frames)

    async def synthesize(self, text: str) -> tuple[bytes, int]:
        """Synthèse complète (non flux) : repli, et salutation mise en cache au démarrage."""
        r = await self._http.post(self._tts_url, json={"text": text})
        return r.content, int(r.headers.get("X-Sample-Rate", str(audio.TTS_RATE)))

    async def _say_streaming(self, text: str) -> int:
        """Lit le flux TTS (morceaux préfixés par leur taille) et l'envoie au fil de l'eau.
        Rend le nombre de trames envoyées."""
        state = self._state
        downsampler = audio.StreamingDownsampler(audio.TTS_RATE)
        sent_any = False
        n_frames = 0
        t0 = time.time()
        try:
            async with self._http.stream("POST", self._tts_stream_url, json={"text": text}) as resp:
                buf = bytearray()
                async for data in resp.aiter_bytes():
                    buf.extend(data)
                    for chunk in audio.pop_length_prefixed(buf):
                        if state.stop_requested:
                            return n_frames
                        frames = downsampler.feed(chunk)
                        if frames:
                            sent_any = True
                            n_frames += len(frames)
                            await self.send_frames(frames)
        except Exception as e:  # noqa: BLE001 — une panne du TTS ne coupe pas l'appel
            log.warning(
                "stream TTS échoué (%s)%s", e, "" if sent_any else " -> repli synthèse complète"
            )
            if not sent_any and not state.stop_requested:
                pcm, rate = await self.synthesize(text)
                if not state.stop_requested:
                    await self.send_frames(audio.pcm_to_telephony_frames(pcm, rate))
        finally:
            log.info(
                "stream fini: %d frames (%.1fs audio) en %.1fs%s",
                n_frames,
                n_frames * FRAME_DURATION_S,
                time.time() - t0,
                " COUPÉ (barge-in)" if state.stop_requested else "",
            )
        return n_frames

    def _truncate_history(self, text: str, n_frames: int) -> None:
        """Le client a coupé : l'historique ne garde que ce qu'il a entendu. Sinon le
        modèle croit avoir dit des choses jamais prononcées et fabrique les malentendus."""
        heard = int(n_frames * FRAME_DURATION_S * SPOKEN_CHARS_PER_S)
        history = self._state.history
        if (
            history
            and history[-1].get("role") == "assistant"
            and history[-1].get("content") == text
            and heard < len(text)
        ):
            history[-1]["content"] = text[:heard].rstrip() + "… (coupé par le client)"
            log.info("historique tronqué à %d caractères (barge-in)", heard)

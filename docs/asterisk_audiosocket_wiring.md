# Brancher Asterisk (FreePBX) sur le serveur AudioSocket Hikky

Ce guide explique comment configurer une VM FreePBX/Asterisk 22 pour router les appels d'une extension vers le serveur AudioSocket Python (`hikky.adapters.telephony.asterisk_audiosocket_server`).

**Prérequis :**
- FreePBX 17 / Asterisk 22 fonctionnels sur une VM Debian 12 (cf. `GUIDE_FREEPBX_COLLEGUE.md` à la racine du dépôt `hikky`).
- Softphone (Zoiper) enregistré sur l'extension 1000 avec l'écho `*43` validé.
- Python 3.11+ installé sur la machine qui hébergera le bot (celle avec les modèles IA — Whisper/LlamaCpp/Piper).

---

## 1. Vue d'ensemble

```
Softphone (ext 1000)
    │  compose "2000"
    ▼
Asterisk (VM FreePBX)
    │  Extension 2000 = dialplan AudioSocket()
    │  ouvre une connexion TCP → bot Hikky
    ▼
Bot Python (asterisk_audiosocket_server, port 6666)
    │  charge RestaurantContext + CallSession
    ▼
Pipeline STT → CallSession → TTS (adapters existants)
    ▲
    │  audio synthétisé
    └── renvoyé à Asterisk via l'AudioSocket
```

L'IP `192.168.1.146` dans les exemples ci-dessous = **l'IP de la machine qui héberge le bot Python**. Adapte à ta config (souvent c'est ton PC hôte quand la VM FreePBX tourne dans VirtualBox en bridged).

---

## 2. Ajouter l'extension 2000 dans le dialplan Asterisk

FreePBX charge automatiquement `/etc/asterisk/extensions_custom.conf`. On y ajoute une extension `2000` qui route vers l'application `AudioSocket()`.

En SSH root sur la VM FreePBX :

```bash
cat > /etc/asterisk/extensions_custom.conf << 'EOF'
[from-internal-custom]

; Extension 999 — démo statique (message pré-enregistré)
; (existant, à conserver si déjà en place)
exten => 999,1,NoOp(Bienvenue Petit Sud - message pre-enregistre)
 same => n,Answer()
 same => n,Wait(1)
 same => n,Playback(custom-welcome)
 same => n,Hangup()

; Extension 2000 — bot IA via AudioSocket
; ${CHANNEL(uuid)} est le UUID interne du canal, transmis au bot
; comme call_id dans le premier paquet AudioSocket (type 0x01).
exten => 2000,1,NoOp(Bot Hikky IA via AudioSocket)
 same => n,Answer()
 same => n,AudioSocket(${CHANNEL(uuid)},192.168.1.146:6666)
 same => n,Hangup()
EOF

chown asterisk:asterisk /etc/asterisk/extensions_custom.conf
asterisk -rx 'dialplan reload'
```

⚠️ **Remplace `192.168.1.146` par l'IP réelle de la machine qui fait tourner le bot Python.** Si tu es sur une VM et que le bot tourne sur ton PC hôte, c'est l'IP wifi de ton PC.

### Vérifier que le dialplan est bien chargé

```bash
asterisk -rx "dialplan show 2000@from-internal-custom"
```

Doit afficher les 3 lignes `NoOp`, `Answer`, `AudioSocket`.

### Vérifier que le module AudioSocket est chargé

```bash
asterisk -rx "module show like audiosocket"
```

Doit lister `app_audiosocket.so`, `chan_audiosocket.so`, `res_audiosocket.so` en état `Running`. Si non :

```bash
asterisk -rx "module load app_audiosocket.so"
```

Ce module est fourni par défaut dans Asterisk 22 avec FreePBX 17 — il devrait déjà être chargé.

---

## 3. Lancer le bot Python en mode "smoke test"

Sur la machine qui héberge le bot (ton PC hôte ou une machine GPU), en dehors de la VM :

```bash
# Variables d'environnement minimales pour valider le transport
# (pas besoin des adapters IA en smoke test)
export HIKKY_AUDIOSOCKET_ENABLED=1
export HIKKY_AUDIOSOCKET_SMOKE_TEST=1
export HIKKY_AUDIOSOCKET_PORT=6666

# Fake variables pour que build_app_dependencies ne râle pas
# (elles ne seront pas utilisées en smoke test — ce mode ne charge
# ni STT/LLM/TTS, mais l'app FastAPI démarre quand même).
# ⚠️ Pour tester UNIQUEMENT AudioSocket sans démarrer le bot IA complet,
# le plus simple est de lancer directement le serveur AudioSocket
# via un petit script — voir section 4.

# Puis démarre l'app :
uvicorn hikky.app.main:app --host 0.0.0.0 --port 8000
```

L'app FastAPI démarre sur le port 8000 **et** le serveur AudioSocket sur le port 6666. Tu verras dans les logs :

```
AudioSocket server listening addresses=[('0.0.0.0', 6666)] smoke_test=True
```

## 4. Script standalone pour tester juste AudioSocket (sans IA)

Pour un test isolé du transport sans démarrer FastAPI ni charger les modèles IA, crée `scripts/run_audiosocket_smoke.py` :

```python
"""Lance le serveur AudioSocket seul, en mode smoke test.

Utile pour valider que le protocole AudioSocket fonctionne bout-en-bout
entre Asterisk et le bot, avant même d'avoir configuré Whisper/LlamaCpp/Piper.

Usage :
    python scripts/run_audiosocket_smoke.py
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, time

from hikky.adapters.telephony.asterisk_audiosocket_server import (
    AudioSocketServerConfig,
    AudioSocketServerDeps,
    start_server,
)
from hikky.domain.call_session import CallSession
from hikky.domain.dialogue_engine import DialogueEngine
from hikky.domain.fallback_policy import FallbackPolicy
from hikky.domain.restaurant_context import (
    OpeningHours,
    RestaurantContext,
    RestaurantRules,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


class _FakeRestaurantCtxPort:
    async def load(self, called_number):
        return RestaurantContext(
            id="r-poc",
            name="Le Petit Sud (POC)",
            greeting="Bonjour, POC AudioSocket.",
            opening_hours=[
                OpeningHours(weekday=i, opens=time(9, 0), closes=time(23, 0))
                for i in range(7)
            ],
            total_capacity=40,
            rules=RestaurantRules(),
            transfer_number=None,
            fallback_message="Merci de rappeler plus tard.",
        )


class _NoopReservationPort:
    async def check_availability(self, *a, **k): return True
    async def create(self, *a, **k): return "res-noop"
    async def create_callback_request(self, *a, **k): return "cb-noop"


class _NoopCallLogPort:
    async def start(self, *a, **k): pass
    async def end(self, *a, **k): pass


class _NoopNotifPort:
    async def send_confirmation(self, *a, **k): pass


class _NoopLLM:
    async def complete(self, messages): return ""


def _session_factory(call_id: str, ctx: RestaurantContext) -> CallSession:
    return CallSession(
        call_id=call_id,
        context=ctx,
        dialogue_engine=DialogueEngine(_NoopLLM()),  # type: ignore[arg-type]
        fallback_policy=FallbackPolicy(),
        reservation_port=_NoopReservationPort(),  # type: ignore[arg-type]
        call_log=_NoopCallLogPort(),  # type: ignore[arg-type]
        notification=_NoopNotifPort(),  # type: ignore[arg-type]
        clock=datetime.now,
    )


async def main():
    deps = AudioSocketServerDeps(
        session_factory=_session_factory,
        restaurant_context_port=_FakeRestaurantCtxPort(),  # type: ignore[arg-type]
    )
    config = AudioSocketServerConfig(
        host="0.0.0.0",
        port=6666,
        smoke_test=True,
        smoke_test_duration_s=15.0,
    )
    server, _adapter = await start_server(deps, config)
    print("AudioSocket smoke test server on 0.0.0.0:6666 — Ctrl+C to stop.")
    try:
        await server.serve_forever()
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        server.close()
        await server.wait_closed()


if __name__ == "__main__":
    asyncio.run(main())
```

Lance :

```bash
cd Call-bot
python scripts/run_audiosocket_smoke.py
```

---

## 5. Le test end-to-end

Une fois le bot Python démarré :

1. Sur le softphone (Zoiper), compose **`2000`**.
2. Asterisk exécute `AudioSocket(<uuid>, 192.168.1.146:6666)`.
3. Asterisk ouvre une connexion TCP vers ton bot Python.
4. Le bot log :
   ```
   audiosocket connection opened peer=('192.168.1.99', XXXXX)
   call bound call_id=<UUID>
   smoke test: peer closed packets=... audio_bytes=...
   ```

Si tu vois ces logs → **le transport AudioSocket marche**. 🎉

Sinon, débug ci-dessous.

---

## 6. Débogages courants

### Aucune connexion au bot / Asterisk raccroche direct

- **Vérifie l'IP du bot** dans `extensions_custom.conf` : c'est bien accessible depuis la VM ? Fais un `ping <IP>` depuis la VM.
- **Firewall Windows** sur le port 6666 : accepte les connexions entrantes depuis le sous-réseau 192.168.1.0/24.
- Regarde les logs Asterisk en direct : `asterisk -rvvv` sur la VM. Compose 2000 depuis Zoiper et regarde ce qu'il dit.

### `Application 'AudioSocket' not found`

Le module n'est pas chargé. En SSH root VM :
```bash
asterisk -rx "module load app_audiosocket.so"
asterisk -rx "module load chan_audiosocket.so"
```

Si le module n'existe pas du tout, ton install FreePBX est incomplète : réinstalle via le script officiel.

### Le bot Python reçoit la connexion mais aucun paquet audio

- Vérifie que l'appel est bien décroché avec `Answer()` dans le dialplan **avant** `AudioSocket()`.
- Regarde `asterisk -rx "core show channels concise"` — le canal doit être en état `Up`.

### `no such file or directory: /var/lib/asterisk/sounds/en/custom-welcome`

Ça concerne l'extension 999 (démo statique), pas 2000. Si tu ne l'utilises pas, tu peux supprimer les 5 lignes correspondantes du dialplan sans risque.

---

## 7. Prochaines étapes (après le smoke test)

Une fois que le smoke test log bien les paquets reçus, on peut brancher la vraie boucle IA :

1. Compléter `_run_ai_loop` dans `asterisk_audiosocket_server.py` (VAD, streaming STT, dispatch dialogue, streaming TTS).
2. Configurer les variables d'env `HIKKY_WHISPER_MODEL`, `HIKKY_LLAMA_MODEL_PATH`, `HIKKY_PIPER_MODEL_PATH` et démarrer avec `HIKKY_AUDIOSOCKET_SMOKE_TEST=0`.
3. Tester une conversation complète : "Bonjour, je voudrais réserver pour 4 demain à 20h" → l'IA extrait les slots + confirme.

Ces étapes sont documentées dans le repo Call-bot (todo interne).

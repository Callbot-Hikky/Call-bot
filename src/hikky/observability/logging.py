"""Logging structuré JSON pour Hikky.

Chaque ligne de log est un objet JSON contenant au minimum `timestamp`,
`level`, `message`, et — quand un appel est en cours — `call_id` et
`restaurant_id` injectés via un `contextvar`.

Permet de tracer un appel précis dans les logs sans grep poétique.
"""

import contextvars
import json
import logging
from datetime import UTC, datetime
from typing import Any

_call_context: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "hikky_call_context", default=None
)


def set_call_context(
    *,
    call_id: str,
    restaurant_id: str | None = None,
    ingest_ref: str | None = None,
) -> None:
    """Fixe le contexte de l'appel courant.

    `ingest_ref` est une clé d'idempotence UNIQUE par appel, distincte du
    `call_id` : le dialplan Asterisk peut renvoyer un UUID AudioSocket figé
    (le même à chaque appel), ce qui faisait retomber toutes les
    réservations sur la même clé et le backend les rejetait en 409. En
    mintant `ingest_ref` à l'ouverture de connexion, l'ingestion reste
    unique même quand `call_id` se répète.
    """
    _call_context.set(
        {
            "call_id": call_id,
            "restaurant_id": restaurant_id,
            "ingest_ref": ingest_ref,
        }
    )


def clear_call_context() -> None:
    _call_context.set(None)


def get_call_context() -> dict[str, Any]:
    ctx = _call_context.get()
    return dict(ctx) if ctx else {}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        ctx = get_call_context()
        for key, value in ctx.items():
            if value is not None:
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        # Tout extra ajouté via logger.info("msg", extra={...}) passe ici
        for key, value in record.__dict__.items():
            if key in payload or key.startswith("_"):
                continue
            if key in (
                "args",
                "asctime",
                "created",
                "exc_info",
                "exc_text",
                "filename",
                "funcName",
                "levelname",
                "levelno",
                "lineno",
                "module",
                "msecs",
                "message",
                "msg",
                "name",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "stack_info",
                "thread",
                "threadName",
                "taskName",
            ):
                continue
            payload[key] = value
        return json.dumps(payload, default=str)


def configure_json_logging(level: int = logging.INFO) -> None:
    """Configure le root logger pour émettre du JSON. À appeler au démarrage."""
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

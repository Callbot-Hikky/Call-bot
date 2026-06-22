import io
import json
import logging

from hikky.observability.logging import (
    JsonFormatter,
    clear_call_context,
    configure_json_logging,
    set_call_context,
)


def _capture_handler() -> tuple[logging.Logger, io.StringIO]:
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger("hikky.test_logging")
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger, stream


def test_log_line_is_valid_json_with_message_and_level():
    logger, stream = _capture_handler()
    logger.info("hello world")
    payload = json.loads(stream.getvalue().strip())
    assert payload["message"] == "hello world"
    assert payload["level"] == "INFO"
    assert "timestamp" in payload


def test_call_context_is_injected_into_log_payload():
    logger, stream = _capture_handler()
    try:
        set_call_context(call_id="c-42", restaurant_id="r-1")
        logger.info("during call")
    finally:
        clear_call_context()
    payload = json.loads(stream.getvalue().strip())
    assert payload["call_id"] == "c-42"
    assert payload["restaurant_id"] == "r-1"


def test_call_context_cleared_no_longer_appears():
    logger, stream = _capture_handler()
    set_call_context(call_id="c-1")
    clear_call_context()
    logger.info("after clear")
    payload = json.loads(stream.getvalue().strip())
    assert "call_id" not in payload


def test_extra_fields_are_preserved():
    logger, stream = _capture_handler()
    logger.info("with extras", extra={"latency_ms": 123.4})
    payload = json.loads(stream.getvalue().strip())
    assert payload["latency_ms"] == 123.4


def test_configure_json_logging_replaces_root_handlers():
    configure_json_logging()
    root = logging.getLogger()
    assert len(root.handlers) == 1
    assert isinstance(root.handlers[0].formatter, JsonFormatter)

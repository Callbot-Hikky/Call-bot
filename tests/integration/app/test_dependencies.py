from datetime import time

from hikky.adapters.back.call_log_adapter import BackHttpCallLogAdapter
from hikky.adapters.back.notification_adapter import BackHttpNotificationAdapter
from hikky.adapters.back.reservation_adapter import BackHttpReservationAdapter
from hikky.adapters.back.restaurant_context_adapter import (
    BackHttpRestaurantContextAdapter,
)
from hikky.adapters.voice.faster_whisper_stt import FasterWhisperSTTAdapter
from hikky.adapters.voice.piper_tts import PiperTTSAdapter
from hikky.app.config import AppConfig
from hikky.app.dependencies import build_app_dependencies
from hikky.domain.call_session import CallSession
from hikky.domain.restaurant_context import (
    OpeningHours,
    RestaurantContext,
    RestaurantRules,
)
from hikky.pipeline.llm_slot_extractor import LLMSlotExtractor


def _config() -> AppConfig:
    return AppConfig(
        back_base_url="https://back.example",
        back_api_key="k",
        whisper_model="distil-large-v3",
        whisper_device="cuda",
        whisper_compute_type="int8",
        llama_model_path="/models/m.gguf",
        llama_n_ctx=4096,
        llama_n_gpu_layers=-1,
        piper_model_path="/voices/v.onnx",
        tts_sample_rate=22050,
    )


def _ctx() -> RestaurantContext:
    return RestaurantContext(
        id="r-1",
        name="Chez Test",
        greeting="Bonjour.",
        opening_hours=[
            OpeningHours(weekday=d, opens=time(19, 0), closes=time(23, 0))
            for d in range(7)
        ],
        total_capacity=40,
        rules=RestaurantRules(),
        transfer_number=None,
        fallback_message="…",
    )


def test_build_app_dependencies_assembles_real_adapters():
    deps = build_app_dependencies(_config())
    assert isinstance(deps.stt_adapter, FasterWhisperSTTAdapter)
    assert isinstance(deps.tts_adapter, PiperTTSAdapter)
    assert isinstance(deps.restaurant_context_port, BackHttpRestaurantContextAdapter)
    assert isinstance(deps.slot_extractor, LLMSlotExtractor)
    assert deps.tts_sample_rate == 22050


def test_session_factory_builds_call_session_wired_to_back_adapters():
    deps = build_app_dependencies(_config())
    session = deps.session_factory("c-123", _ctx())
    assert isinstance(session, CallSession)
    assert session.call_id == "c-123"
    # Le ReservationPort câblé est bien l'adapter Back HTTP, pas un fake
    # (on n'a pas d'accesseur public ; on vérifie via les autres ports)
    assert session.context.id == "r-1"
    # Sanity : la session est bien câblée à TOUS les Back adapters
    # (on lit l'état interne pour ce test d'assemblage uniquement)
    assert isinstance(session._reservation, BackHttpReservationAdapter)
    assert isinstance(session._log, BackHttpCallLogAdapter)
    assert isinstance(session._notif, BackHttpNotificationAdapter)

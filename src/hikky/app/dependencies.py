"""Assembleur des `AppDependencies` à partir de l'`AppConfig`.

Sépare la construction des dépendances (logique d'install) du runtime
de l'app (création de l'app FastAPI). C'est ce module qui sait qu'on
utilise `BackHttpAdapter` pour le Back, `FasterWhisperSTTAdapter` pour
le STT, etc. — tout le câblage concret vit ici.
"""

from __future__ import annotations

from datetime import datetime

from hikky.adapters.back.call_log_adapter import BackHttpCallLogAdapter
from hikky.adapters.back.http_client import BackHttpClient
from hikky.adapters.back.notification_adapter import BackHttpNotificationAdapter
from hikky.adapters.back.reservation_adapter import BackHttpReservationAdapter
from hikky.adapters.back.restaurant_context_adapter import (
    BackHttpRestaurantContextAdapter,
)
from hikky.adapters.voice.faster_whisper_stt import FasterWhisperSTTAdapter
from hikky.adapters.voice.llama_cpp_llm import LlamaCppLLMAdapter
from hikky.adapters.voice.piper_tts import PiperTTSAdapter
from hikky.app.config import AppConfig
from hikky.domain.call_session import CallSession
from hikky.domain.dialogue_engine import DialogueEngine
from hikky.domain.fallback_policy import FallbackPolicy
from hikky.domain.restaurant_context import RestaurantContext
from hikky.pipeline.llm_slot_extractor import LLMSlotExtractor


def build_app_dependencies(config: AppConfig):
    """Construit `AppDependencies` à partir de la config. À appeler une
    seule fois au démarrage de l'app — les adapters de modèles font
    leur lazy-load au premier vrai appel."""

    from hikky.app.main import AppDependencies

    back_client = BackHttpClient(
        base_url=config.back_base_url,
        api_key=config.back_api_key,
    )

    llm = LlamaCppLLMAdapter(
        model_path=config.llama_model_path,
        n_ctx=config.llama_n_ctx,
        n_gpu_layers=config.llama_n_gpu_layers,
    )

    reservation = BackHttpReservationAdapter(back_client)
    call_log = BackHttpCallLogAdapter(back_client)
    notification = BackHttpNotificationAdapter(back_client)
    restaurant_ctx = BackHttpRestaurantContextAdapter(back_client)
    fallback = FallbackPolicy()
    slot_extractor = LLMSlotExtractor(llm)

    def session_factory(call_sid: str, ctx: RestaurantContext) -> CallSession:
        return CallSession(
            call_id=call_sid,
            context=ctx,
            dialogue_engine=DialogueEngine(llm),
            fallback_policy=fallback,
            reservation_port=reservation,
            call_log=call_log,
            notification=notification,
            clock=datetime.now,
        )

    return AppDependencies(
        stt_adapter=FasterWhisperSTTAdapter(
            model_name=config.whisper_model,
            device=config.whisper_device,
            compute_type=config.whisper_compute_type,
        ),
        tts_adapter=PiperTTSAdapter(model_path=config.piper_model_path),
        restaurant_context_port=restaurant_ctx,
        session_factory=session_factory,
        slot_extractor=slot_extractor,
        tts_sample_rate=config.tts_sample_rate,
    )

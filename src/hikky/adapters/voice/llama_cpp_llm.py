"""Adapter LLM basé sur llama-cpp-python.

Installation : `pip install -e ".[voice]"` puis télécharger un modèle GGUF
(par exemple Mistral 7B Instruct Q4_K_M, ~4 Go). Sur GPU NVIDIA, compiler
llama-cpp-python avec `CMAKE_ARGS="-DGGML_CUDA=on"` pour utiliser les
couches GPU via `n_gpu_layers`.

Lazy import — la suite de tests tourne sans la lib installée.

L'inférence llama-cpp est synchrone et CPU/GPU-bound ; on l'enveloppe
dans `asyncio.to_thread` pour ne pas bloquer l'event loop pendant que
le modèle génère.
"""

import asyncio
from typing import Any

from hikky.exceptions import LLMOverloaded
from hikky.ports.language_model import LanguageModelPort

DEFAULT_MAX_TOKENS = 256
DEFAULT_TEMPERATURE = 0.4


class LlamaCppLLMAdapter(LanguageModelPort):
    def __init__(
        self,
        *,
        model_path: str,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,  # -1 = tout sur GPU si possible
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> None:
        self._model_path = model_path
        self._n_ctx = n_ctx
        self._n_gpu_layers = n_gpu_layers
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._model: Any | None = None

    def _load_model(self) -> Any:
        if self._model is None:
            from llama_cpp import Llama  # lazy

            self._model = Llama(
                model_path=self._model_path,
                n_ctx=self._n_ctx,
                n_gpu_layers=self._n_gpu_layers,
                verbose=False,
            )
        return self._model

    async def complete(self, messages: list[dict[str, str]]) -> str:
        model = self._load_model()
        try:
            response = await asyncio.to_thread(
                model.create_chat_completion,
                messages=messages,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            )
        except (RuntimeError, ValueError, OSError) as exc:
            raise LLMOverloaded(f"llama-cpp failed: {exc}") from exc

        try:
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMOverloaded(f"Unexpected llama-cpp response shape: {response!r}") from exc

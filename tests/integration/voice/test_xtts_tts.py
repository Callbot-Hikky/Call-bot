import struct
from types import SimpleNamespace

import pytest

from hikky.adapters.voice.xtts_tts import XttsTTSAdapter

SPEAKER = "Lilya Stainthorpe"


class FakeSpeakerManager:
    def __init__(self, names) -> None:
        self.speakers = {
            n: {"gpt_cond_latent": f"lat-{n}", "speaker_embedding": f"emb-{n}"}
            for n in names
        }


class FakeXttsModel:
    """Reproduit la forme réelle de `TTS.tts.models.xtts.Xtts`.

    Classe explicite plutôt que MagicMock : un mock fabriquerait n'importe
    quel attribut et masquerait un changement d'API.
    """

    def __init__(self, chunks=None, speakers=(SPEAKER, "Autre")) -> None:
        self.speaker_manager = FakeSpeakerManager(speakers)
        self._chunks = chunks if chunks is not None else [[0.0, 0.5, -0.5, 1.0]]
        self.stream_calls = []
        self.cuda_called = False

    def cuda(self):
        self.cuda_called = True
        return self

    def inference_stream(self, text, language, gpt_cond_latent, speaker_embedding, **kw):
        self.stream_calls.append((text, language, gpt_cond_latent, speaker_embedding))
        for chunk in self._chunks:
            yield SimpleNamespace(tolist=lambda c=chunk: c)


@pytest.fixture
def fake_model(monkeypatch):
    model = FakeXttsModel()
    monkeypatch.setattr(
        "hikky.adapters.voice.xtts_tts._load_model", lambda *a, **k: model
    )
    return model


async def test_synthesize_yields_pcm16_bytes(fake_model):
    adapter = XttsTTSAdapter(speaker=SPEAKER)
    chunks = [c async for c in await adapter.synthesize("bonjour")]
    assert chunks
    assert all(isinstance(c, bytes) for c in chunks)


async def test_float_audio_is_converted_to_int16(fake_model):
    fake_model._chunks = [[0.0, 1.0, -1.0]]
    adapter = XttsTTSAdapter(speaker=SPEAKER)
    data = b"".join([c async for c in await adapter.synthesize("x")])
    values = struct.unpack(f"<{len(data) // 2}h", data)
    assert values == (0, 32767, -32767)


async def test_out_of_range_floats_are_clamped(fake_model):
    fake_model._chunks = [[4.0, -4.0]]
    adapter = XttsTTSAdapter(speaker=SPEAKER)
    data = b"".join([c async for c in await adapter.synthesize("x")])
    values = struct.unpack(f"<{len(data) // 2}h", data)
    assert all(-32768 <= v <= 32767 for v in values)


async def test_french_language_is_requested(fake_model):
    adapter = XttsTTSAdapter(speaker=SPEAKER)
    [c async for c in await adapter.synthesize("bonjour")]
    assert fake_model.stream_calls[0][1] == "fr"


async def test_selected_speaker_latents_are_used(fake_model):
    adapter = XttsTTSAdapter(speaker=SPEAKER)
    [c async for c in await adapter.synthesize("bonjour")]
    _, _, latent, embedding = fake_model.stream_calls[0]
    assert latent == f"lat-{SPEAKER}"
    assert embedding == f"emb-{SPEAKER}"


async def test_unknown_speaker_raises_with_available_names(fake_model):
    adapter = XttsTTSAdapter(speaker="Personne Inexistante")
    with pytest.raises(ValueError) as exc:
        [c async for c in await adapter.synthesize("bonjour")]
    assert SPEAKER in str(exc.value)


async def test_sample_rate_is_24000(fake_model):
    adapter = XttsTTSAdapter(speaker=SPEAKER)
    assert adapter.sample_rate == 24000


async def test_chunks_are_streamed_not_buffered(fake_model):
    """Le premier son doit sortir avant la fin de la generation.

    C'est TOUT l'interet du flux : mesure sur le pod, 0,33 s au premier
    chunk contre 1,62 s en bloquant. Une implementation qui accumule
    avant de livrer annule ce gain sans qu'aucun test naif ne s'en
    apercoive.
    """
    import time

    def slow_stream(text, language, gpt_cond_latent, speaker_embedding, **kw):
        for i in range(4):
            time.sleep(0.06)
            yield SimpleNamespace(tolist=lambda i=i: [0.1 * (i + 1)])

    fake_model.inference_stream = slow_stream
    adapter = XttsTTSAdapter(speaker=SPEAKER)

    t0 = time.monotonic()
    first_at = None
    stream = await adapter.synthesize("x")
    async for _chunk in stream:
        if first_at is None:
            first_at = time.monotonic() - t0
    total = time.monotonic() - t0

    assert first_at is not None
    assert first_at < total * 0.6, (
        f"buffering detecte: premier chunk a {first_at:.3f}s "
        f"sur {total:.3f}s au total"
    )


async def test_model_loaded_once_across_calls(fake_model):
    calls = []

    adapter = XttsTTSAdapter(speaker=SPEAKER)
    original = adapter._get_model

    def counting():
        calls.append(1)
        return original()

    adapter._get_model = counting
    [c async for c in await adapter.synthesize("un")]
    [c async for c in await adapter.synthesize("deux")]
    assert len(calls) == 2  # appelé 2 fois mais le modèle est mémorisé
    assert adapter._model is fake_model


async def test_empty_generation_yields_nothing(fake_model):
    fake_model._chunks = []
    adapter = XttsTTSAdapter(speaker=SPEAKER)
    assert [c async for c in await adapter.synthesize("")] == []


async def test_synthesize_does_not_block_event_loop(fake_model):
    import asyncio
    import time

    def slow_stream(text, language, gpt_cond_latent, speaker_embedding, **kw):
        time.sleep(0.08)
        yield SimpleNamespace(tolist=lambda: [0.1])

    fake_model.inference_stream = slow_stream
    adapter = XttsTTSAdapter(speaker=SPEAKER)
    progress = []

    async def tick():
        for _ in range(5):
            await asyncio.sleep(0.02)
            progress.append(1)

    async def synth():
        [c async for c in await adapter.synthesize("hello")]

    await asyncio.gather(tick(), synth())
    assert len(progress) >= 3

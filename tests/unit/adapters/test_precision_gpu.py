"""La precision de calcul doit suivre le materiel.

`compute_type="int8"` etait servi a un adapter dont `device` vaut
"cuda". Or int8 est le reglage prevu pour le **CPU** : sur un GPU Ampere
(RTX 3090), les Tensor Cores sont concus pour le float16, et l'int8 y
est plus lent tout en degradant la transcription.

Le cout est double sur un callbot : chaque mot mal transcrit devient une
information mal comprise, et l'appelant doit se repeter.

Repli prevu : si la VRAM manque, `int8_float16` garde les couches non
quantisees en FP16 — le compromis recommande sur ce materiel. Mais tant
qu'il reste de la place, le float16 pur gagne sur les deux tableaux.
"""

from hikky.adapters.voice.faster_whisper_stt import (
    PRECISION_CPU,
    PRECISION_GPU,
    FasterWhisperSTTAdapter,
)


class TestPrecisionParDefaut:
    def test_le_gpu_utilise_le_float16(self):
        """int8 sur CUDA gaspille les Tensor Cores et degrade le texte."""
        adapter = FasterWhisperSTTAdapter(device="cuda")
        assert adapter.compute_type == PRECISION_GPU
        assert adapter.compute_type == "float16"

    def test_le_cpu_garde_l_int8(self):
        """Sans GPU, l'int8 reste le bon compromis."""
        adapter = FasterWhisperSTTAdapter(device="cpu")
        assert adapter.compute_type == PRECISION_CPU
        assert adapter.compute_type == "int8"

    def test_un_choix_explicite_prime(self):
        """Si la VRAM manque, on doit pouvoir imposer int8_float16."""
        adapter = FasterWhisperSTTAdapter(device="cuda", compute_type="int8_float16")
        assert adapter.compute_type == "int8_float16"


class TestModele:
    def test_le_modele_reste_multilingue(self):
        """`distil-large-v3` est anglophone : il traduisait le francais."""
        adapter = FasterWhisperSTTAdapter(device="cuda")
        assert "distil" not in adapter.model_name
        assert adapter.model_name == "large-v3"

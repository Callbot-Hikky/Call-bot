"""Flash attention : petit gain, mais gratuit et mesure.

Mesure sur le pod, Qwen 14B Q6_K, prompt du Phraseur, mediane sur 5 :

    flash_attn=False   428,2 ms
    flash_attn=True    414,1 ms

Soit 14 ms, environ 3 %. Modeste, et c'est instructif : le prompt est
court, donc le temps ne part pas dans l'attention mais dans la
generation des tokens, limitee par la bande passante memoire.

Le reglage reste activable par defaut — aucun inconvenient mesure, et il
reduit aussi l'empreinte du cache KV. Il doit rester debrayable : sur un
GPU sans support, llama.cpp retomberait sur un chemin plus lent.
"""

from hikky.adapters.voice.llama_cpp_llm import LlamaCppLLMAdapter


class _LlamaEspion:
    """Capture les arguments passes a llama.cpp."""

    dernier_appel: dict = {}

    def __init__(self, **kwargs):
        _LlamaEspion.dernier_appel = kwargs


def _adapter(monkeypatch, **kwargs) -> LlamaCppLLMAdapter:
    # `llama_cpp` est une dependance GPU, absente de la machine de dev :
    # on injecte un module factice plutot que de sauter le test, sinon
    # le contrat avec llama.cpp ne serait verifie nulle part.
    import sys
    import types

    faux = types.ModuleType("llama_cpp")
    faux.Llama = _LlamaEspion
    monkeypatch.setitem(sys.modules, "llama_cpp", faux)
    adapter = LlamaCppLLMAdapter(model_path="/modele.gguf", **kwargs)
    adapter._load_model()
    return adapter


def test_flash_attention_est_active_par_defaut(monkeypatch):
    _adapter(monkeypatch)
    assert _LlamaEspion.dernier_appel["flash_attn"] is True


def test_flash_attention_reste_debrayable(monkeypatch):
    """Sur un GPU sans support, il faut pouvoir revenir en arriere."""
    _adapter(monkeypatch, flash_attn=False)
    assert _LlamaEspion.dernier_appel["flash_attn"] is False


def test_les_couches_restent_toutes_sur_le_gpu(monkeypatch):
    """-1 = tout sur GPU. Une regression ici ferait tourner sur CPU."""
    _adapter(monkeypatch)
    assert _LlamaEspion.dernier_appel["n_gpu_layers"] == -1

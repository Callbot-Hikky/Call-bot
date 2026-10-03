"""Chemin TTS accelere (CUDA graphs) pour Qwen3-TTS-12Hz-1.7B-CustomVoice.

Remplace la double boucle autoregressive HF (talker + code_predictor.generate
imbrique, 15 pas/frame) par deux graphes CUDA rejoues :
  - PredictorGraph : les 15 sous-pas du sous-talker en UN seul replay (greedy),
  - TalkerGraph    : le pas talker (28 couches) en UN seul replay.
Le codebook 0 garde le sampling complet (top_k/top_p/temperature/rep_pen) en eager.
Mesure : RTF ~0.35 vs 1.86 (5.3x). Sortie audio validee a l'oreille.

Buffers de graphe partages -> NON thread-safe : proteger chaque appel par _LOCK.
Chemin optionnel : n'est active que si le serveur le demande (flag TTS_FAST).
"""
import threading
import numpy as np
import torch

import streaming_engine as se     # _sample_next_token
import optimized_talker as ot     # TalkerGraph, PredictorGraph

# Verrou global : un enonce a la fois (le callbot est sequentiel par appel).
LOCK = threading.Lock()


def _patch_lazy_init():
    """transformers 4.57.3 : StaticLayer.lazy_initialization prend 1 arg.
    optimized_talker.py appelle (dk, dk) -> crash. On remplace _init_cache_layers."""
    def _fixed_init(self):
        c = self.pred_model.config if hasattr(self, "pred_model") else self.model.config
        kv = getattr(c, "num_key_value_heads", c.num_attention_heads)
        hd = getattr(c, "head_dim", c.hidden_size // c.num_attention_heads)
        dk = torch.zeros(1, kv, 1, hd, dtype=self.dtype, device=self.device)
        for L in self.static_cache.layers:
            if not L.is_initialized:
                L.lazy_initialization(dk)
    ot.PredictorGraph._init_cache_layers = _fixed_init
    ot.TalkerGraph._init_cache_layers = _fixed_init


class FastTTS:
    def __init__(self, m, device="cuda", dtype=torch.bfloat16, max_seq_len=1024):
        self.m = m
        self.model = m.model
        self.talker = self.model.talker
        tcfg = self.model.config.talker_config
        pcfg = tcfg.code_predictor_config
        self.NG = tcfg.num_code_groups
        self.eos_ids = {tcfg.codec_eos_token_id, 2150, 2157}
        vocab = tcfg.vocab_size
        self.suppress = [i for i in range(vocab - 1024, vocab) if i not in self.eos_ids]
        self._eos_idx = torch.tensor(sorted(self.eos_ids), device=device)

        _patch_lazy_init()
        self.tg = ot.TalkerGraph(self.talker.model, tcfg, device=device,
                                 dtype=dtype, max_seq_len=max_seq_len)
        self.pg = ot.PredictorGraph(self.talker.code_predictor, pcfg, tcfg.hidden_size,
                                    device=device, dtype=dtype, do_sample=False)
        self.tg.capture(prefill_len=100, num_warmup=3)
        self.pg.capture(num_warmup=3)

    def _sample(self, logits, temperature, top_k, top_p, do_sample):
        """Tirage du code suivant, le signal de FIN restant toujours tirable.

        Emballement observé en appel réel (« Puis-je avoir votre nom ? » -> 12,8 s de
        babil, ~1 phrase sur 15, surtout les courtes) : le token de fin (EOS) a une
        probabilité réelle mais n'est souvent pas dans les 50 meilleurs codes ; le filtre
        top-k le mettait à -inf -> impossible à tirer -> le modèle DEVAIT continuer.
        Le sampler de référence de Qwen sauvegarde le logit de l'EOS avant le filtre et le
        réinjecte après (mesure publiée : 16 fins/200 tirages avec, 0/200 sans). Idem ici.
        """
        if not do_sample or temperature <= 0:
            lg = logits.clone(); lg[..., self.suppress] = float("-inf")
            return torch.argmax(lg, dim=-1)
        lg = logits.clone()
        lg[..., self.suppress] = float("-inf")
        lg = lg / temperature
        eos_logits = lg[..., self._eos_idx].clone()
        lg = se._top_k_top_p_filtering(lg, top_k=top_k, top_p=top_p)
        lg[..., self._eos_idx] = eos_logits
        probs = torch.softmax(lg, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    # Arrêt sur code FIGÉ. Analyse de 11 emballements réels (debug/babil.json) : les longs
    # (89 à 160 frames) sont des BOUCLES — le code du codebook 0 reste identique sur 60 à
    # 75 % des frames de la queue (10 frames consécutives identiques = 0,8 s d'un même son,
    # tenu à -21/-26 dB : une voyelle ou un souffle qui ne s'arrête plus). En parole normale
    # un code figé aussi longtemps n'est que du silence de fin : couper là est sans perte.
    BOUCLE_FRAMES = 12

    @staticmethod
    def _boucle(rp_hist, k=BOUCLE_FRAMES):
        return len(rp_hist) >= k and len(set(rp_hist[-k:])) == 1

    @staticmethod
    def frames_plafond(text, max_frames=160, par_caractere=2.5, marge=12, minimum=12):
        """Plafond de frames PROPORTIONNEL au texte, au lieu de 160 pour tout.

        Au téléphone, le français fait ~1,1 frame (80 ms) par caractère ; 2,5 frames par
        caractère + marge laisse plus de 2x la durée naturelle (parole lente, émotion
        comprises) tout en bornant le pire cas : « Puis-je avoir votre nom ? » (26 car.)
        -> 77 frames = 6,2 s au lieu de 12,8 s si le modèle ne s'arrête pas.
        """
        n = len((text or "").strip())
        return max(minimum, min(max_frames, int(par_caractere * n) + marge))

    @torch.inference_mode()
    def generate(self, text, speaker, instruct, max_frames=None, do_sample=True,
                 top_k=50, top_p=1.0, temperature=0.9, rep_pen=1.05):
        m = self.m; model = self.model; talker = self.talker
        tg = self.tg; pg = self.pg; NG = self.NG
        eos_ids = self.eos_ids; suppress = self.suppress
        if max_frames is None:
            max_frames = self.frames_plafond(text)

        input_ids = m._tokenize_texts([m._build_assistant_text(text)])
        instruct_ids = [m._tokenize_texts([m._build_instruct_text(instruct)])[0]]
        embeds, attn, trailing, tts_pad = model._build_talker_inputs(
            input_ids, instruct_ids, None, None, ["French"], [speaker],
            non_streaming_mode=True)
        out = talker.forward(inputs_embeds=embeds, attention_mask=attn, use_cache=True,
                             output_hidden_states=True, return_dict=True,
                             trailing_text_hidden=trailing, tts_pad_embed=tts_pad,
                             generation_step=None, past_hidden=None, past_key_values=None,
                             subtalker_dosample=False)
        seqlen = tg.prefill_kv(out.past_key_values)
        tg.set_generation_state(attention_mask=attn, rope_deltas=talker.rope_deltas)
        past_hidden = out.past_hidden.clone()
        gstep = out.generation_step
        logits = out.logits[:, -1, :]
        token = self._sample(logits, temperature, top_k, top_p, do_sample)
        cp_embed = talker.code_predictor.get_input_embeddings()
        codes = []; pos = seqlen; rp_hist = [int(token[0])]
        for _ in range(max_frames):
            if int(token[0]) in eos_ids:
                break
            last_id_hidden = talker.get_input_embeddings()(token.unsqueeze(1))
            pred_in = torch.cat((past_hidden, last_id_hidden), dim=1)
            codes1_15 = pg.run(pred_in)
            codec_ids = torch.cat((token.unsqueeze(1), codes1_15.unsqueeze(0)), dim=1)
            ch = [last_id_hidden] + [cp_embed[i](codes1_15[i:i + 1].unsqueeze(0)) for i in range(NG - 1)]
            emb = torch.cat(ch, dim=1).sum(1, keepdim=True)
            emb = emb + (trailing[:, gstep].unsqueeze(1) if gstep < trailing.shape[1] else tts_pad)
            hidden = tg.run(emb, pos).clone()
            pos += 1; gstep += 1
            logits = talker.codec_head(hidden)[:, -1, :]
            past_hidden = hidden
            codes.append(codec_ids)
            if rep_pen != 1.0:
                uniq = torch.tensor(list(set(rp_hist[-100:])), device=logits.device)
                lg = logits.clone(); v = lg[0, uniq]
                lg[0, uniq] = torch.where(v > 0, v / rep_pen, v * rep_pen); logits = lg
            token = self._sample(logits, temperature, top_k, top_p, do_sample)
            rp_hist.append(int(token[0]))
            if self._boucle(rp_hist):
                print(f"[tts] boucle detectee frame {len(codes)} (code {rp_hist[-1]} x{self.BOUCLE_FRAMES}) -> arret", flush=True)
                break
        all_codes = torch.cat(codes, dim=0)
        wavs, fs = model.speech_tokenizer.decode([{"audio_codes": all_codes}])
        return wavs[0], fs, all_codes.shape[0]

    @torch.inference_mode()
    def generate_stream(self, text, speaker, instruct, emit_frames=16, holdback=2,
                        max_frames=None, do_sample=True, top_k=50, top_p=1.0,
                        temperature=0.9, rep_pen=1.05):
        """Generateur : yield (pcm_float32_numpy, fs) par chunks de ~emit_frames
        frames (~1.08s a 12 Hz pour 13). Holdback : on retient les `holdback`
        dernieres frames a chaque emission (leurs samples changent quand on ajoute
        des codes) -> pas de clic de raccord entre chunks. Le dernier chunk emet tout."""
        m = self.m; model = self.model; talker = self.talker
        tg = self.tg; pg = self.pg; NG = self.NG
        eos_ids = self.eos_ids; suppress = self.suppress
        if max_frames is None:
            max_frames = self.frames_plafond(text)

        input_ids = m._tokenize_texts([m._build_assistant_text(text)])
        instruct_ids = [m._tokenize_texts([m._build_instruct_text(instruct)])[0]]
        embeds, attn, trailing, tts_pad = model._build_talker_inputs(
            input_ids, instruct_ids, None, None, ["French"], [speaker],
            non_streaming_mode=True)
        out = talker.forward(inputs_embeds=embeds, attention_mask=attn, use_cache=True,
                             output_hidden_states=True, return_dict=True,
                             trailing_text_hidden=trailing, tts_pad_embed=tts_pad,
                             generation_step=None, past_hidden=None, past_key_values=None,
                             subtalker_dosample=False)
        seqlen = tg.prefill_kv(out.past_key_values)
        tg.set_generation_state(attention_mask=attn, rope_deltas=talker.rope_deltas)
        past_hidden = out.past_hidden.clone()
        gstep = out.generation_step
        logits = out.logits[:, -1, :]
        token = self._sample(logits, temperature, top_k, top_p, do_sample)
        cp_embed = talker.code_predictor.get_input_embeddings()
        codes = []; pos = seqlen; rp_hist = [int(token[0])]
        emitted = 0            # samples deja emis
        last_emit_len = 0      # nb de frames au dernier emit

        def _decode_all():
            all_codes = torch.cat(codes, dim=0)
            wavs, fs = model.speech_tokenizer.decode([{"audio_codes": all_codes}])
            return np.asarray(wavs[0], dtype=np.float32), fs

        for _ in range(max_frames):
            if int(token[0]) in eos_ids:
                break
            last_id_hidden = talker.get_input_embeddings()(token.unsqueeze(1))
            pred_in = torch.cat((past_hidden, last_id_hidden), dim=1)
            codes1_15 = pg.run(pred_in)
            codec_ids = torch.cat((token.unsqueeze(1), codes1_15.unsqueeze(0)), dim=1)
            ch = [last_id_hidden] + [cp_embed[i](codes1_15[i:i + 1].unsqueeze(0)) for i in range(NG - 1)]
            emb = torch.cat(ch, dim=1).sum(1, keepdim=True)
            emb = emb + (trailing[:, gstep].unsqueeze(1) if gstep < trailing.shape[1] else tts_pad)
            hidden = tg.run(emb, pos).clone()
            pos += 1; gstep += 1
            logits = talker.codec_head(hidden)[:, -1, :]
            past_hidden = hidden
            codes.append(codec_ids)
            if rep_pen != 1.0:
                uniq = torch.tensor(list(set(rp_hist[-100:])), device=logits.device)
                lg = logits.clone(); v = lg[0, uniq]
                lg[0, uniq] = torch.where(v > 0, v / rep_pen, v * rep_pen); logits = lg
            token = self._sample(logits, temperature, top_k, top_p, do_sample)
            rp_hist.append(int(token[0]))
            if self._boucle(rp_hist):
                print(f"[tts] boucle detectee frame {len(codes)} (code {rp_hist[-1]} x{self.BOUCLE_FRAMES}) -> arret", flush=True)
                break
            if len(codes) - last_emit_len >= emit_frames and len(codes) > holdback:
                w, fs = _decode_all()
                spf = len(w) / len(codes)
                keep = int((len(codes) - holdback) * spf)
                if keep > emitted:
                    yield w[emitted:keep], fs
                    emitted = keep
                    last_emit_len = len(codes)
        if codes:
            w, fs = _decode_all()
            if len(w) > emitted:
                yield w[emitted:], fs

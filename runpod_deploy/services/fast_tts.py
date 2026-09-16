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

        _patch_lazy_init()
        self.tg = ot.TalkerGraph(self.talker.model, tcfg, device=device,
                                 dtype=dtype, max_seq_len=max_seq_len)
        self.pg = ot.PredictorGraph(self.talker.code_predictor, pcfg, tcfg.hidden_size,
                                    device=device, dtype=dtype, do_sample=False)
        self.tg.capture(prefill_len=100, num_warmup=3)
        self.pg.capture(num_warmup=3)

    @torch.inference_mode()
    def generate(self, text, speaker, instruct, max_frames=512, do_sample=True,
                 top_k=50, top_p=1.0, temperature=0.9, rep_pen=1.05):
        m = self.m; model = self.model; talker = self.talker
        tg = self.tg; pg = self.pg; NG = self.NG
        eos_ids = self.eos_ids; suppress = self.suppress

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
        token = (se._sample_next_token(logits, temperature, top_k, top_p, suppress)
                 if do_sample else torch.argmax(logits, -1))
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
            token = (se._sample_next_token(logits, temperature, top_k, top_p, suppress)
                     if do_sample else torch.argmax(logits, -1))
            rp_hist.append(int(token[0]))
        all_codes = torch.cat(codes, dim=0)
        wavs, fs = model.speech_tokenizer.decode([{"audio_codes": all_codes}])
        return wavs[0], fs, all_codes.shape[0]

    @torch.inference_mode()
    def generate_stream(self, text, speaker, instruct, emit_frames=16, holdback=2,
                        max_frames=512, do_sample=True, top_k=50, top_p=1.0,
                        temperature=0.9, rep_pen=1.05):
        """Generateur : yield (pcm_float32_numpy, fs) par chunks de ~emit_frames
        frames (~1.08s a 12 Hz pour 13). Holdback : on retient les `holdback`
        dernieres frames a chaque emission (leurs samples changent quand on ajoute
        des codes) -> pas de clic de raccord entre chunks. Le dernier chunk emet tout."""
        m = self.m; model = self.model; talker = self.talker
        tg = self.tg; pg = self.pg; NG = self.NG
        eos_ids = self.eos_ids; suppress = self.suppress

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
        token = (se._sample_next_token(logits, temperature, top_k, top_p, suppress)
                 if do_sample else torch.argmax(logits, -1))
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
            token = (se._sample_next_token(logits, temperature, top_k, top_p, suppress)
                     if do_sample else torch.argmax(logits, -1))
            rp_hist.append(int(token[0]))
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

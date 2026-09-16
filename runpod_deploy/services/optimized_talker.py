"""CUDA-graph accelerated talker + code_predictor for qwen-tts 0.1.1.
Monkeypatch, aucune modif de site-packages. Reutilise les MEMES poids en VRAM.
Portage de andimarafioti/faster-qwen3-tts (Apache-2.0) sur l'API installee
Qwen3TTSTalkerForConditionalGeneration (modeling_qwen3_tts.py)."""
import torch
from transformers import StaticCache


# ------------------- TalkerGraph : 1 pas talker (28 couches, hidden 2048) -----
class TalkerGraph:
    def __init__(self, talker_model, talker_config, device='cuda',
                 dtype=torch.bfloat16, max_seq_len=512):
        self.device = device
        di = torch.device(device).index
        self.device_index = di if di is not None else torch.cuda.current_device()
        self.dtype = dtype
        self.max_seq_len = max_seq_len
        self.hidden_size = talker_config.hidden_size
        self.num_layers = talker_config.num_hidden_layers
        self.model = talker_model  # = talker.model (Qwen3TTSTalkerModel)
        self.static_cache = StaticCache(config=talker_config, max_cache_len=max_seq_len)
        self.input_buf = torch.zeros(1, 1, self.hidden_size, dtype=dtype, device=device)
        self.output_buf = torch.zeros(1, 1, self.hidden_size, dtype=dtype, device=device)
        self.cache_position = torch.zeros(1, dtype=torch.long, device=device)
        self.position_ids = torch.zeros(3, 1, 1, dtype=torch.long, device=device)  # MRoPE 3D
        self.rope_deltas = torch.zeros(1, 1, dtype=torch.long, device=device)
        self.graph = None
        self.captured = False
        self.attn_mask = None
        self.attn_mask_table = None
        self._mask_key = None

    def _init_cache_layers(self):
        c = self.model.config
        kv = getattr(c, 'num_key_value_heads', c.num_attention_heads)
        hd = getattr(c, 'head_dim', c.hidden_size // c.num_attention_heads)
        dk = torch.zeros(1, kv, 1, hd, dtype=self.dtype, device=self.device)
        for layer in self.static_cache.layers:
            if not layer.is_initialized:
                layer.lazy_initialization(dk, dk)

    def _build_attention_masks(self, padding=None):
        L = self.max_seq_len
        self.attn_mask_table = [None] * L
        kp = torch.arange(L, device=self.device)
        mn = torch.finfo(self.dtype).min
        sw = getattr(self.model.config, 'sliding_window', None)
        pad = None if padding is None else padding.to(torch.bool)
        for i in range(L):
            allowed = kp <= i
            if sw is not None:
                allowed = allowed & (kp > i - sw)
            allowed = allowed.unsqueeze(0) if pad is None else allowed.unsqueeze(0) & pad
            full = torch.where(allowed,
                               torch.zeros((), dtype=self.dtype, device=self.device),
                               torch.full((), mn, dtype=self.dtype, device=self.device))
            self.attn_mask_table[i] = full.unsqueeze(1).unsqueeze(1)  # [1,1,1,L]
        if self.attn_mask is None:
            self.attn_mask = self.attn_mask_table[0].clone()
        else:
            self.attn_mask.copy_(self.attn_mask_table[0])

    def _set_mask(self, pos):
        self.attn_mask.copy_(self.attn_mask_table[pos])

    def _decode_step(self):
        out = self.model(inputs_embeds=self.input_buf, attention_mask=self.attn_mask,
                         past_key_values=self.static_cache, cache_position=self.cache_position,
                         position_ids=self.position_ids, use_cache=True)
        self.output_buf.copy_(out.last_hidden_state)

    @torch.inference_mode()
    def capture(self, prefill_len=100, num_warmup=3):
        self._init_cache_layers()
        self._build_attention_masks()
        self.cache_position[0] = prefill_len
        self._set_mask(prefill_len)
        for _ in range(num_warmup):
            self._decode_step()
        torch.cuda.synchronize()
        with torch.cuda.device(self.device_index):
            self.graph = torch.cuda.CUDAGraph()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                self._decode_step()
                torch.cuda.synchronize()
                with torch.cuda.graph(self.graph):
                    self._decode_step()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        self.captured = True

    def prefill_kv(self, past_key_values):
        """Copie la DynamicCache du prefill dans la StaticCache. Retourne seq_len."""
        self.static_cache.reset()
        seq_len = 0
        for li in range(self.num_layers):
            layer = past_key_values.layers[li]
            k, v = layer.keys, layer.values
            seq_len = k.shape[2]
            if seq_len > self.max_seq_len:
                raise RuntimeError("prefill %d > max_seq_len %d" % (seq_len, self.max_seq_len))
            cp = torch.arange(seq_len, device=self.device)
            self.static_cache.update(k, v, li, {"cache_position": cp})
        return seq_len

    def set_generation_state(self, attention_mask, rope_deltas=None):
        mask_key = None
        full = None
        if attention_mask is not None:
            padc = (attention_mask == 0).sum(dim=-1)
            mask_key = tuple(padc.tolist())
            full = torch.ones(attention_mask.shape[0], self.max_seq_len,
                              dtype=attention_mask.dtype, device=attention_mask.device)
            for b, p in enumerate(padc.tolist()):
                if p > 0:
                    full[b, :p] = 0
        if self.attn_mask_table is None or mask_key != self._mask_key:
            self._build_attention_masks(full)
            self._mask_key = mask_key
        if rope_deltas is None:
            self.rope_deltas.zero_()
        else:
            rd = rope_deltas.unsqueeze(1) if rope_deltas.dim() == 1 else rope_deltas
            self.rope_deltas.copy_(rd.to(self.rope_deltas.device, dtype=self.rope_deltas.dtype))

    @torch.inference_mode()
    def run(self, input_embeds, position):
        self.input_buf.copy_(input_embeds)
        self.cache_position[0] = position
        self._set_mask(position)
        delta = self.rope_deltas + self.cache_position[0]
        self.position_ids.copy_(delta.unsqueeze(0).expand(3, -1, -1))
        self.graph.replay()
        return self.output_buf  # buffer statique : clone si conserve


# ------------- PredictorGraph : 15 pas sub-talker (5 couches, 16 codebooks) ----
class PredictorGraph:
    def __init__(self, code_predictor, pred_config, talker_hidden_size, device='cuda',
                 dtype=torch.bfloat16, do_sample=True, top_k=50, top_p=1.0, temperature=0.9):
        self.device = device
        di = torch.device(device).index
        self.device_index = di if di is not None else torch.cuda.current_device()
        self.dtype = dtype
        self.num_layers = pred_config.num_hidden_layers
        self.hidden_size = pred_config.hidden_size
        self.num_code_groups = pred_config.num_code_groups
        self.num_codebooks = self.num_code_groups - 1   # 15
        self.max_seq = 2 + self.num_codebooks           # 17
        self.do_sample, self.top_k, self.top_p, self.temperature = do_sample, top_k, top_p, temperature
        cp = code_predictor
        self.small_to_mtp = cp.small_to_mtp_projection
        self.pred_model = cp.model
        self.lm_heads = cp.lm_head                  # ModuleList[15]
        self.codec_embeds = cp.model.codec_embedding    # ModuleList[15]
        self.has_sliding = "sliding_attention" in getattr(self.pred_model.config, "layer_types", [])
        self.static_cache = StaticCache(config=pred_config, max_cache_len=self.max_seq)
        self.prefill_cache_pos = torch.arange(2, device=device)
        self.decode_cache_positions = [torch.tensor([2 + i], device=device)
                                       for i in range(self.num_codebooks - 1)]
        self.input_buf = torch.zeros(1, 2, talker_hidden_size, dtype=dtype, device=device)
        self.output_tokens = torch.zeros(self.num_codebooks, dtype=torch.long, device=device)
        self.graph = None
        self.captured = False
        self.prefill_attn = None
        self.decode_attn = None

    def _sample(self, logits):  # logits [1, vocab] -> scalaire long
        if not self.do_sample:
            return torch.argmax(logits, dim=-1)
        x = logits / self.temperature
        if self.top_k > 0:
            v, _ = torch.topk(x, min(self.top_k, x.size(-1)))
            x = torch.where(x < v[..., -1:], torch.full_like(x, float("-inf")), x)
        probs = torch.softmax(x, dim=-1)
        return torch.multinomial(probs, 1).squeeze(-1)

    def _init_cache_layers(self):
        c = self.pred_model.config
        kv = getattr(c, 'num_key_value_heads', c.num_attention_heads)
        hd = getattr(c, 'head_dim', c.hidden_size // c.num_attention_heads)
        dk = torch.zeros(1, kv, 1, hd, dtype=self.dtype, device=self.device)
        for layer in self.static_cache.layers:
            if not layer.is_initialized:
                layer.lazy_initialization(dk, dk)

    def _mask(self, emb, cache_position):
        kp = torch.arange(self.max_seq, device=self.device)
        allowed = kp.unsqueeze(0) <= cache_position.unsqueeze(1)
        mn = torch.finfo(emb.dtype).min
        m = torch.where(allowed, torch.zeros((), dtype=emb.dtype, device=self.device),
                        torch.full((), mn, dtype=emb.dtype, device=self.device)).unsqueeze(0).unsqueeze(0)
        if self.has_sliding:
            w = self.pred_model.config.sliding_window
            inw = kp.unsqueeze(0) > (cache_position.unsqueeze(1) - w)
            sl = torch.where(allowed & inw, torch.zeros((), dtype=emb.dtype, device=self.device),
                             torch.full((), mn, dtype=emb.dtype, device=self.device)).unsqueeze(0).unsqueeze(0)
            return {"full_attention": m, "sliding_attention": sl}
        return {"full_attention": m}

    def _build_masks(self):
        dp = torch.zeros(1, 2, self.hidden_size, dtype=self.dtype, device=self.device)
        dd = torch.zeros(1, 1, self.hidden_size, dtype=self.dtype, device=self.device)
        self.prefill_attn = self._mask(dp, self.prefill_cache_pos)
        self.decode_attn = [self._mask(dd, p) for p in self.decode_cache_positions]

    def _full_loop(self):
        h = self.small_to_mtp(self.input_buf)
        out = self.pred_model(inputs_embeds=h, attention_mask=self.prefill_attn,
                              past_key_values=self.static_cache,
                              cache_position=self.prefill_cache_pos, use_cache=True)
        h = out.last_hidden_state
        logits = self.lm_heads[0](h[:, -1:, :])
        tok = self._sample(logits[:, 0, :])
        self.output_tokens[0] = tok[0]
        for cb in range(1, self.num_codebooks):
            emb = self.codec_embeds[cb - 1](tok.unsqueeze(0))
            emb = self.small_to_mtp(emb)
            out = self.pred_model(inputs_embeds=emb, attention_mask=self.decode_attn[cb - 1],
                                  past_key_values=self.static_cache,
                                  cache_position=self.decode_cache_positions[cb - 1], use_cache=True)
            h = out.last_hidden_state
            logits = self.lm_heads[cb](h[:, -1:, :])
            tok = self._sample(logits[:, 0, :])
            self.output_tokens[cb] = tok[0]
        return self.output_tokens

    @torch.inference_mode()
    def capture(self, num_warmup=3):
        self._init_cache_layers()
        self._build_masks()
        for _ in range(num_warmup):
            self.static_cache.reset()
            self._full_loop()
        torch.cuda.synchronize()
        with torch.cuda.device(self.device_index):
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                self.graph = torch.cuda.CUDAGraph()
                self.static_cache.reset()
                self._full_loop()
                torch.cuda.synchronize()
                self.static_cache.reset()
                with torch.cuda.graph(self.graph):
                    self._full_loop()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        self.captured = True

    @torch.inference_mode()
    def run(self, pred_input):   # [1,2,talker_hidden] -> [15] long
        self.input_buf.copy_(pred_input)
        self.static_cache.reset()
        self.graph.replay()
        return self.output_tokens.clone()

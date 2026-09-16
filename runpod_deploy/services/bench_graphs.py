import os, time, torch, numpy as np
os.environ.setdefault("HF_HOME", "/workspace/hf")
torch.backends.cuda.matmul.allow_tf32 = True
import sys; sys.path.insert(0, "/workspace/services")
from qwen_tts import Qwen3TTSModel
m = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa")
talker = m.model.talker
tcfg = m.model.config.talker_config
pcfg = tcfg.code_predictor_config
print("importing optimized_talker...")
import optimized_talker as ot

# Patch API mismatch: transformers 4.57.3 lazy_initialization(self, key_states) takes 1 arg
def _fixed_init_cache_layers(self):
    c = self.pred_model.config if hasattr(self,'pred_model') else self.model.config
    dtype = self.dtype; dev = self.device
    kv = getattr(c, 'num_key_value_heads', c.num_attention_heads)
    hd = getattr(c, 'head_dim', c.hidden_size // c.num_attention_heads)
    dk = torch.zeros(1, kv, 1, hd, dtype=dtype, device=dev)
    for layer in self.static_cache.layers:
        if not layer.is_initialized:
            layer.lazy_initialization(dk)
ot.PredictorGraph._init_cache_layers = _fixed_init_cache_layers
ot.TalkerGraph._init_cache_layers = _fixed_init_cache_layers

# ---- Predictor graph (captures 15 steps + sampling) ----
try:
    pg = ot.PredictorGraph(talker.code_predictor, pcfg, tcfg.hidden_size, device="cuda",
                           dtype=torch.bfloat16, do_sample=False)  # greedy => capturable & correct
    t=time.time(); pg.capture(num_warmup=3); print("PredictorGraph.capture OK %.2fs"%(time.time()-t))
    x = torch.randn(1,2,tcfg.hidden_size, dtype=torch.bfloat16, device="cuda")
    # time replay
    for _ in range(5): pg.run(x)
    torch.cuda.synchronize(); s=time.time(); N=200
    for _ in range(N): pg.run(x)
    torch.cuda.synchronize(); dtp=(time.time()-s)/N
    print("PredictorGraph replay (15 substeps, greedy) = %.3f ms/frame"%(dtp*1000))
except Exception as e:
    import traceback; traceback.print_exc(); dtp=None

# ---- Talker graph (1 step, 28 layers) ----
try:
    tg = ot.TalkerGraph(talker.model, tcfg, device="cuda", dtype=torch.bfloat16, max_seq_len=512)
    t=time.time(); tg.capture(prefill_len=100, num_warmup=3); print("TalkerGraph.capture OK %.2fs"%(time.time()-t))
    emb = torch.randn(1,1,tcfg.hidden_size, dtype=torch.bfloat16, device="cuda")
    tg.set_generation_state(attention_mask=None, rope_deltas=None)
    for _ in range(5): tg.run(emb, 100)
    torch.cuda.synchronize(); s=time.time(); N=200
    for i in range(N): tg.run(emb, 100+ (i%300))
    torch.cuda.synchronize(); dtt=(time.time()-s)/N
    print("TalkerGraph replay (1 step, 28 layers) = %.3f ms/frame"%(dtt*1000))
except Exception as e:
    import traceback; traceback.print_exc(); dtt=None

if dtp and dtt:
    per_frame = dtp+dtt
    print("=== PER-FRAME (graphs only) = %.3f ms ; est 69 frames = %.3f s ; +decode 0.03s"%(per_frame*1000, per_frame*69))
    audio=5.5; tot=per_frame*69+0.03
    print("=== EST RTF (graph ceiling, ignores glue/sampling) = %.3f"%(tot/audio))

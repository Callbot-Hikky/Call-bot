import os, time, numpy as np, torch
os.environ.setdefault("HF_HOME", "/workspace/hf")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
SPEAKER = "ono_anna"
INSTRUCT = "Ton chaleureux et professionnel d une hotesse de restaurant francais"
TEXT = "Deux personnes le mercredi 16 septembre a 20 heures, au nom de Cadet. C'est bien cela ?"

from qwen_tts import Qwen3TTSModel
t=time.time()
m = Qwen3TTSModel.from_pretrained(MODEL_ID, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa")
print("load %.1fs"%(time.time()-t), flush=True)

model = m.model
talker = model.talker
cp = talker.code_predictor

# counters
stats = {"talker_forward":0, "talker_gen_calls":0, "cp_gen_calls":0, "cp_forward":0,
         "t_talker_fwd":0.0, "t_cp_gen":0.0, "t_decode":0.0}

import functools
orig_talker_fwd = talker.forward.__func__
orig_cp_gen = cp.generate.__func__
orig_cp_fwd = cp.forward.__func__

@functools.wraps(orig_talker_fwd)
def timed_talker_fwd(self, *a, **k):
    stats["talker_forward"]+=1
    torch.cuda.synchronize(); s=time.time()
    r = orig_talker_fwd(self,*a,**k)
    torch.cuda.synchronize(); stats["t_talker_fwd"]+=time.time()-s
    return r

@functools.wraps(orig_cp_gen)
def timed_cp_gen(self, *a, **k):
    stats["cp_gen_calls"]+=1
    torch.cuda.synchronize(); s=time.time()
    r = orig_cp_gen(self,*a,**k)
    torch.cuda.synchronize(); stats["t_cp_gen"]+=time.time()-s
    return r

@functools.wraps(orig_cp_fwd)
def timed_cp_fwd(self, *a, **k):
    stats["cp_forward"]+=1
    return orig_cp_fwd(self,*a,**k)

import types
talker.forward = types.MethodType(timed_talker_fwd, talker)
cp.generate = types.MethodType(timed_cp_gen, cp)
cp.forward = types.MethodType(timed_cp_fwd, cp)

def run_once(tag):
    for k in stats: stats[k]=0 if isinstance(stats[k],int) else 0.0
    torch.cuda.synchronize(); t0=time.time()
    wavs, sr = m.generate_custom_voice(text=TEXT, language="French", speaker=SPEAKER, instruct=INSTRUCT)
    torch.cuda.synchronize(); dt=time.time()-t0
    wav=np.asarray(wavs[0],dtype=np.float32); dur=len(wav)/sr
    print("[%s] total=%.3fs audio=%.3fs sr=%d RTF=%.3f"%(tag,dt,dur,sr,dt/dur), flush=True)
    print("   talker_forward=%d (=frames+prefill) t_talker_fwd=%.3fs"%(stats["talker_forward"],stats["t_talker_fwd"]))
    print("   cp_gen_calls=%d cp_forward=%d t_cp_gen=%.3fs"%(stats["cp_gen_calls"],stats["cp_forward"],stats["t_cp_gen"]))
    print("   avg per talker frame: talker_fwd=%.2fms cp_gen=%.2fms"%(
        1000*stats["t_talker_fwd"]/max(stats["talker_forward"],1),
        1000*stats["t_cp_gen"]/max(stats["cp_gen_calls"],1)))
    return wav, sr

# time decode separately
def time_decode(codes_list):
    torch.cuda.synchronize(); s=time.time()
    for _ in range(3):
        wavs, fs = m.model.speech_tokenizer.decode([{"audio_codes": c} for c in codes_list])
    torch.cuda.synchronize(); return (time.time()-s)/3, wavs, fs

print("=== warmup ==="); run_once("warmup")
print("=== timed ==="); run_once("timed1"); run_once("timed2")

# breakdown talker.generate vs decode using low-level
input_ids = m._tokenize_texts([m._build_assistant_text(TEXT)])
instruct_ids=[m._tokenize_texts([m._build_instruct_text(INSTRUCT)])[0]]
gen_kwargs=m._merge_generate_kwargs()
torch.cuda.synchronize(); s=time.time()
codes_list,_ = model.generate(input_ids=input_ids, instruct_ids=instruct_ids, languages=["French"], speakers=[SPEAKER], non_streaming_mode=True, **gen_kwargs)
torch.cuda.synchronize(); t_gen=time.time()-s
nframes=codes_list[0].shape[0]
print("talker.generate ONLY = %.3fs, frames=%d, code_groups=%d"%(t_gen,nframes,codes_list[0].shape[1]))
td, wavs, fs = time_decode(codes_list)
print("decode (token2wav) = %.3fs for %d frames -> %.3fs audio, decode-RTF=%.3f"%(td,nframes,len(wavs[0])/fs, td/(len(wavs[0])/fs)))
print("SPLIT: talker_gen=%.1f%%  decode=%.1f%%"%(100*t_gen/(t_gen+td),100*td/(t_gen+td)))

import os, time, torch, numpy as np
os.environ.setdefault("HF_HOME", "/workspace/hf")
torch.backends.cuda.matmul.allow_tf32 = True
import sys; sys.path.insert(0, "/workspace/services")
from qwen_tts import Qwen3TTSModel
import streaming_engine as se   # for _build_talker_inputs + _sample_next_token
import optimized_talker as ot

TEXT = "Deux personnes le mercredi 16 septembre a 20 heures, au nom de Cadet. C'est bien cela ?"
SPEAKER="ono_anna"; INSTRUCT="Ton chaleureux et professionnel d une hotesse de restaurant francais"

m = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa")
model=m.model; talker=model.talker; tcfg=model.config.talker_config; pcfg=tcfg.code_predictor_config

# --- fix lazy_initialization API (transformers 4.57.3) ---
def _fixed_init(self):
    c = self.pred_model.config if hasattr(self,'pred_model') else self.model.config
    kv=getattr(c,'num_key_value_heads',c.num_attention_heads); hd=getattr(c,'head_dim',c.hidden_size//c.num_attention_heads)
    dk=torch.zeros(1,kv,1,hd,dtype=self.dtype,device=self.device)
    for L in self.static_cache.layers:
        if not L.is_initialized: L.lazy_initialization(dk)
ot.PredictorGraph._init_cache_layers=_fixed_init; ot.TalkerGraph._init_cache_layers=_fixed_init

MAXSEQ=1024
tg=ot.TalkerGraph(talker.model,tcfg,device="cuda",dtype=torch.bfloat16,max_seq_len=MAXSEQ)
pg=ot.PredictorGraph(talker.code_predictor,pcfg,tcfg.hidden_size,device="cuda",dtype=torch.bfloat16,do_sample=False)
tg.capture(prefill_len=100,num_warmup=3); pg.capture(num_warmup=3)
print("graphs captured")

eos_ids={tcfg.codec_eos_token_id,2150,2157}
vocab=tcfg.vocab_size
suppress=[i for i in range(vocab-1024,vocab) if i not in eos_ids]
NG=tcfg.num_code_groups  # 16

@torch.inference_mode()
def fast_generate(text, max_frames=512, do_sample=True, top_k=50, top_p=1.0, temperature=0.9, rep_pen=1.05):
    input_ids=m._tokenize_texts([m._build_assistant_text(text)])
    instruct_ids=[m._tokenize_texts([m._build_instruct_text(INSTRUCT)])[0]]
    embeds,attn,trailing,tts_pad=model._build_talker_inputs(input_ids,instruct_ids,None,None,["French"],[SPEAKER],non_streaming_mode=True)
    out=talker.forward(inputs_embeds=embeds,attention_mask=attn,use_cache=True,output_hidden_states=True,
                       return_dict=True,trailing_text_hidden=trailing,tts_pad_embed=tts_pad,
                       generation_step=None,past_hidden=None,past_key_values=None,subtalker_dosample=False)
    seqlen=tg.prefill_kv(out.past_key_values)
    tg.set_generation_state(attention_mask=attn,rope_deltas=talker.rope_deltas)
    past_hidden=out.past_hidden.clone()
    gstep=out.generation_step
    logits=out.logits[:,-1,:]
    token=se._sample_next_token(logits,temperature,top_k,top_p,suppress) if do_sample else torch.argmax(logits,-1)
    cp_embed=talker.code_predictor.get_input_embeddings()
    codes=[]; pos=seqlen
    rp_hist=[int(token[0])]
    for step in range(max_frames):
        if int(token[0]) in eos_ids: break
        last_id_hidden=talker.get_input_embeddings()(token.unsqueeze(1))  # [1,1,H]
        pred_in=torch.cat((past_hidden,last_id_hidden),dim=1)             # [1,2,H]
        codes1_15=pg.run(pred_in)                                         # [15]
        codec_ids=torch.cat((token.unsqueeze(1),codes1_15.unsqueeze(0)),dim=1)  # [1,16]
        ch=[last_id_hidden]+[cp_embed[i](codes1_15[i:i+1].unsqueeze(0)) for i in range(NG-1)]
        emb=torch.cat(ch,dim=1).sum(1,keepdim=True)
        emb = emb + (trailing[:,gstep].unsqueeze(1) if gstep<trailing.shape[1] else tts_pad)
        hidden=tg.run(emb,pos).clone()
        pos+=1; gstep+=1
        logits=talker.codec_head(hidden)[:,-1,:]
        past_hidden=hidden
        codes.append(codec_ids)
        # codebook0 sampling + light repetition penalty
        if rep_pen!=1.0:
            uniq=torch.tensor(list(set(rp_hist[-100:])),device=logits.device)
            lg=logits.clone(); v=lg[0,uniq]; lg[0,uniq]=torch.where(v>0,v/rep_pen,v*rep_pen); logits=lg
        token=se._sample_next_token(logits,temperature,top_k,top_p,suppress) if do_sample else torch.argmax(logits,-1)
        rp_hist.append(int(token[0]))
    all_codes=torch.cat(codes,dim=0)  # [T,16]
    wavs,fs=model.speech_tokenizer.decode([{"audio_codes":all_codes}])
    return wavs[0],fs,all_codes.shape[0]

# warmup + timed
for tag in ["warm","t1","t2"]:
    torch.cuda.synchronize(); s=time.time()
    wav,fs,nf=fast_generate(TEXT)
    torch.cuda.synchronize(); dt=time.time()-s
    dur=len(wav)/fs
    print("[%s] frames=%d total=%.3fs audio=%.3fs RTF=%.3f"%(tag,nf,dt,dur,dt/dur))

import soundfile as sf
sf.write("/workspace/services/poc_fast_out.wav", np.clip(wav,-1,1), fs)
print("wrote poc_fast_out.wav  (rms=%.4f, peak=%.3f)"%(np.sqrt((wav**2).mean()), np.abs(wav).max()))

"""Moteur streaming pour qwen-tts 0.1.1 (monkeypatch, aucune modif de site-packages).
Blocs verbatim du fork rekuenkdr/Qwen3-TTS-streaming (Apache-2.0), attaches a la
classe installee. use_optimized_decode=False -> chemin decode standard (zero compile)."""
import numpy as np
import torch
import torch.nn.functional as F
from typing import Callable, Optional, Generator
from transformers.cache_utils import Cache, DynamicCache
from qwen_tts.core.models import modeling_qwen3_tts as M


def _top_k_top_p_filtering(logits: torch.Tensor, top_k: int = 0, top_p: float = 1.0) -> torch.Tensor:
    """Apply top-k and top-p (nucleus) filtering to logits."""
    if top_k > 0:
        topk = torch.topk(logits, k=min(top_k, logits.size(-1)), dim=-1)
        min_keep = topk.values[..., -1, None]
        logits = torch.where(logits < min_keep, torch.full_like(logits, float("-inf")), logits)
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumprobs = torch.cumsum(probs, dim=-1)
        mask = cumprobs > top_p
        mask[..., 0] = False
        sorted_logits = torch.where(mask, torch.full_like(sorted_logits, float("-inf")), sorted_logits)
        inv_idx = torch.argsort(sorted_idx, dim=-1)
        logits = torch.gather(sorted_logits, dim=-1, index=inv_idx)
    return logits


def _sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    suppress_tokens: Optional[list[int]] = None,
) -> torch.Tensor:
    """Sample next token from logits with temperature, top-k, top-p and token suppression."""
    # Suppress tokens by setting their logits to -inf
    if suppress_tokens is not None and len(suppress_tokens) > 0:
        logits = logits.clone()
        logits[..., suppress_tokens] = float("-inf")

    if temperature <= 0:
        return torch.argmax(logits, dim=-1)
    logits = logits / temperature
    logits = _top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def _crossfade(prev_tail: np.ndarray, new_head: np.ndarray) -> np.ndarray:
    """Crossfade between end of previous chunk and start of new chunk using Hann window."""
    n = min(len(prev_tail), len(new_head))
    if n <= 0:
        return new_head
    t = np.arange(n, dtype=np.float32) / max(n - 1, 1)
    fade_in = 0.5 * (1 - np.cos(np.pi * t))
    fade_out = 1 - fade_in
    return prev_tail[:n] * fade_out + new_head[:n] * fade_in


# Default blend samples for boundary blending
# ~21ms at 24kHz, matches RMS check window for better coverage
# Lower values may cause clicks, set to 0 to disable
DEFAULT_BLEND_SAMPLES = 512


def _add_ref_code_context(
    window_codes: torch.Tensor,
    ref_code_context: Optional[torch.Tensor],
    ref_code_frames: int,
    decode_window_frames: int,
) -> tuple[torch.Tensor, int]:
    """Add ref_code as context prefix when window doesn't fill decode_window_frames.

    Returns:
        tuple: (window with prefix, number of ref_prefix_frames used)
    """
    if ref_code_context is None or window_codes.shape[0] >= decode_window_frames:
        return window_codes, 0

    available_space = decode_window_frames - window_codes.shape[0]
    ref_prefix_frames = min(available_space, ref_code_frames)

    if ref_prefix_frames > 0:
        ref_prefix = ref_code_context[-ref_prefix_frames:]  # Use tail of ref_code
        return torch.cat([ref_prefix, window_codes], dim=0), ref_prefix_frames

    return window_codes, 0



# ===== _build_talker_inputs (fork 2302-2510, dedentee) =====
def _build_talker_inputs(
    self,
    input_ids: list[torch.Tensor],
    instruct_ids: Optional[list[torch.Tensor]],
    ref_ids: Optional[list[torch.Tensor]],
    voice_clone_prompt: Optional[list[dict]],
    languages: list[str],
    speakers: Optional[list[str]],
    non_streaming_mode: bool = False,
):
    """
    Build talker input embeddings, attention mask, trailing text hiddens and tts_pad_embed.
    Extracted from generate() for reuse in streaming.

    Returns:
        tuple: (talker_input_embeds, talker_attention_mask, trailing_text_hiddens, tts_pad_embed)
    """
    talker_input_embeds = [[] for _ in range(len(input_ids))]

    voice_clone_spk_embeds = None
    if voice_clone_prompt is not None:
        voice_clone_spk_embeds = self.generate_speaker_prompt(voice_clone_prompt)

    if instruct_ids is not None:
        for index, instruct_id in enumerate(instruct_ids):
            if instruct_id is not None:
                talker_input_embeds[index].append(self.talker.text_projection(
                                              self.talker.get_text_embeddings()(instruct_id)))

    trailing_text_hiddens = []
    if speakers is None:
        speakers = [None] * len(input_ids)
    for index, (input_id, language, speaker) in enumerate(zip(input_ids, languages, speakers)):
        if voice_clone_spk_embeds is None:
            if speaker == "" or speaker == None:
                speaker_embed = None
            else:
                if speaker.lower() not in self.config.talker_config.spk_id:
                    raise NotImplementedError(f"Speaker {speaker} not implemented")
                else:
                    spk_id = self.config.talker_config.spk_id[speaker.lower()]
                    speaker_embed = self.talker.get_input_embeddings()(
                                        torch.tensor(
                                            spk_id,
                                            device=self.talker.device,
                                            dtype=input_id.dtype,
                                        )
                                    )
        else:
            if voice_clone_prompt["x_vector_only_mode"][index] or voice_clone_prompt["icl_mode"][index]:
                speaker_embed = voice_clone_spk_embeds[index]
            else:
                speaker_embed = None

        assert language is not None

        if language.lower() == "auto":
            language_id = None
        else:
            if language.lower() not in self.config.talker_config.codec_language_id:
                raise NotImplementedError(f"Language {language} not implemented")
            else:
                language_id = self.config.talker_config.codec_language_id[language.lower()]

        if (language.lower() in ["chinese", "auto"] and \
               speaker != "" and speaker is not None and \
                 self.config.talker_config.spk_is_dialect[speaker.lower()] != False):
            dialect = self.config.talker_config.spk_is_dialect[speaker.lower()]
            language_id = self.config.talker_config.codec_language_id[dialect]

        tts_bos_embed, tts_eos_embed, tts_pad_embed = self.talker.text_projection(
            self.talker.get_text_embeddings()(
                torch.tensor(
                    [[self.config.tts_bos_token_id, self.config.tts_eos_token_id, self.config.tts_pad_token_id]],
                    device=self.talker.device,
                    dtype=input_id.dtype,
                )
            )
        ).chunk(3, dim=1)

        if language_id is None:
            codec_prefill_list = [[
                                    self.config.talker_config.codec_nothink_id,
                                    self.config.talker_config.codec_think_bos_id,
                                    self.config.talker_config.codec_think_eos_id,
                                ]]
        else:
            codec_prefill_list = [[
                                    self.config.talker_config.codec_think_id,
                                    self.config.talker_config.codec_think_bos_id,
                                    language_id,
                                    self.config.talker_config.codec_think_eos_id,
                                ]]

        codec_input_emebdding_0 = self.talker.get_input_embeddings()(
                                                torch.tensor(
                                                    codec_prefill_list,
                                                    device=self.talker.device,
                                                    dtype=input_id.dtype,
                                                )
                                            )
        codec_input_emebdding_1 = self.talker.get_input_embeddings()(
                                                torch.tensor(
                                                    [[
                                                        self.config.talker_config.codec_pad_id,
                                                        self.config.talker_config.codec_bos_id,
                                                    ]],
                                                    device=self.talker.device,
                                                    dtype=input_id.dtype,
                                                )
                                            )
        if speaker_embed is None:
            codec_input_emebdding = torch.cat([codec_input_emebdding_0,
                                               codec_input_emebdding_1], dim=1)
        else:
            codec_input_emebdding = torch.cat([codec_input_emebdding_0,
                                               speaker_embed.view(1, 1, -1),
                                               codec_input_emebdding_1], dim=1)

        _talker_input_embed_role = self.talker.text_projection(
                                    self.talker.get_text_embeddings()(input_id[:, :3])
                                    )

        _talker_input_embed = torch.cat((tts_pad_embed.expand(-1, codec_input_emebdding.shape[1] - 2, -1),
                                        tts_bos_embed,
                                        ), dim=1) + codec_input_emebdding[:, :-1]

        talker_input_embed = torch.cat((_talker_input_embed_role, _talker_input_embed), dim=1)

        if voice_clone_prompt is not None and voice_clone_prompt["ref_code"] is not None and voice_clone_prompt["icl_mode"][index]:
            icl_input_embed, trailing_text_hidden = self.generate_icl_prompt(
                text_id=input_id[:, 3:-5],
                ref_id=ref_ids[index][:, 3:-2],
                ref_code=voice_clone_prompt["ref_code"][index].to(self.talker.device),
                tts_pad_embed=tts_pad_embed,
                tts_eos_embed=tts_eos_embed,
                non_streaming_mode=non_streaming_mode,
            )
            talker_input_embed = torch.cat([talker_input_embed, icl_input_embed], dim=1)
        else:
            talker_input_embed = torch.cat([talker_input_embed,
                                            self.talker.text_projection(self.talker.get_text_embeddings()(input_id[:, 3:4])) + codec_input_emebdding[:, -1:]],
                                            dim=1)
            if non_streaming_mode:
                talker_input_embed = talker_input_embed[:, :-1]
                talker_input_embed = torch.cat([talker_input_embed,
                                                torch.cat((self.talker.text_projection(
                                                    self.talker.get_text_embeddings()(input_id[:, 3:-5])
                                                ), tts_eos_embed), dim=1) + self.talker.get_input_embeddings()(
                                                    torch.tensor(
                                                        [[
                                                            self.config.talker_config.codec_pad_id,
                                                        ] * (input_id[:, 3:-5].shape[1] + 1)],
                                                        device=self.talker.device,
                                                        dtype=input_id.dtype,
                                                    )
                                                ),
                                                tts_pad_embed + self.talker.get_input_embeddings()(
                                                    torch.tensor(
                                                        [[
                                                            self.config.talker_config.codec_bos_id,
                                                        ]],
                                                        device=self.talker.device,
                                                        dtype=input_id.dtype,
                                                    )
                                                )
                                                ], dim=1)
                trailing_text_hidden = tts_pad_embed
            else:
                trailing_text_hidden = torch.cat((self.talker.text_projection(
                                                    self.talker.get_text_embeddings()(input_id[:, 4:-5])
                                                ), tts_eos_embed), dim=1)
        talker_input_embeds[index].append(talker_input_embed)
        trailing_text_hiddens.append(trailing_text_hidden)

    for index, talker_input_embed in enumerate(talker_input_embeds):
        talker_input_embeds[index] = torch.cat([item for item in talker_input_embed if item is not None], dim=1)

    original_lengths = torch.tensor([t.shape[1] for t in talker_input_embeds])
    sequences = [t.squeeze(0) for t in talker_input_embeds]
    sequences_reversed = [t.flip(dims=[0]) for t in sequences]
    padded_reversed = torch.nn.utils.rnn.pad_sequence(
        sequences_reversed,
        batch_first=True,
        padding_value=0.0
    )
    talker_input_embeds = padded_reversed.flip(dims=[1])
    batch_size, max_len = talker_input_embeds.shape[0], talker_input_embeds.shape[1]
    indices = torch.arange(max_len).expand(batch_size, -1)
    num_pads = max_len - original_lengths
    talker_attention_mask = (indices >= num_pads.unsqueeze(1)).long().to(talker_input_embeds.device)

    pad_embedding_vector = tts_pad_embed.squeeze()
    sequences_to_pad = [t.squeeze(0) for t in trailing_text_hiddens]
    trailing_text_original_lengths = [s.shape[0] for s in sequences_to_pad]
    padded_hiddens = torch.nn.utils.rnn.pad_sequence(
        sequences_to_pad,
        batch_first=True,
        padding_value=0.0
    )
    arange_tensor = torch.arange(max(trailing_text_original_lengths),
                                 device=padded_hiddens.device).expand(len(trailing_text_original_lengths), -1)
    lengths_tensor = torch.tensor(trailing_text_original_lengths, device=padded_hiddens.device).unsqueeze(1)
    padding_mask = arange_tensor >= lengths_tensor
    padded_hiddens[padding_mask] = pad_embedding_vector
    trailing_text_hiddens = padded_hiddens

    return talker_input_embeds, talker_attention_mask, trailing_text_hiddens, tts_pad_embed


# ===== stream_generate_pcm (fork 2606-2945, dedentee) =====
@torch.inference_mode()
def stream_generate_pcm(
    self,
    input_ids: list[torch.Tensor],
    instruct_ids: Optional[list[torch.Tensor]] = None,
    ref_ids: Optional[list[torch.Tensor]] = None,
    voice_clone_prompt: Optional[list[dict]] = None,
    languages: Optional[list[str]] = None,
    speakers: Optional[list[str]] = None,
    non_streaming_mode: bool = False,
    # Sampling parameters for first codebook
    do_sample: bool = True,
    top_k: int = 50,
    top_p: float = 1.0,
    temperature: float = 0.9,
    # Sub-talker parameters (for remaining code groups)
    subtalker_dosample: bool = True,
    subtalker_top_k: int = 50,
    subtalker_top_p: float = 1.0,
    subtalker_temperature: float = 0.9,
    # Repetition penalty
    repetition_penalty: float = 1.0,
    repetition_penalty_window: int = 100,
    # Streaming control
    emit_every_frames: int = 8,
    decode_window_frames: int = 80,
    overlap_samples: int = 512,
    max_frames: int = 10000,
    # Optimization flags
    use_optimized_decode: bool = True,
    # Two-phase streaming: aggressive first chunk
    first_chunk_emit_every: int = 0,  # 0 = disabled, use emit_every_frames throughout
    first_chunk_decode_window: int = 48,
    first_chunk_frames: int = 48,  # Switch to stable after this many frames
) -> Generator[tuple[np.ndarray, int], None, None]:
    """
    Stream audio generation, yielding PCM chunks as they are generated.

    Args:
        input_ids: List of input token tensors
        instruct_ids: Optional instruction token tensors
        ref_ids: Optional reference token tensors
        voice_clone_prompt: Optional voice cloning prompt dict
        languages: List of language strings
        speakers: Optional list of speaker names
        non_streaming_mode: Whether to use non-streaming text mode
        do_sample: Whether to sample (vs greedy) for first codebook
        top_k: Top-k filtering for sampling
        top_p: Top-p (nucleus) filtering for sampling
        temperature: Sampling temperature
        subtalker_*: Parameters for sub-codebook prediction
        repetition_penalty: Penalty factor for previously generated tokens (1.0 = disabled)
        repetition_penalty_window: Only penalize tokens from the last N steps (0 = unlimited).
            Codec models reuse tokens heavily; unlimited tracking starves the vocabulary.
        emit_every_frames: Emit PCM chunk every N codec frames
        decode_window_frames: Window size for decoding (longer = better quality, more latency)
        overlap_samples: Overlap samples for crossfade between chunks
        max_frames: Maximum number of codec frames to generate
        use_optimized_decode: Use CUDA graph optimized decode when available (default True)
        first_chunk_emit_every: Emit interval for first chunk phase (0 = disabled, use emit_every_frames)
        first_chunk_decode_window: Decode window size for first chunk phase
        first_chunk_frames: Switch to stable settings after this many frames

    Yields:
        tuple[np.ndarray, int]: (pcm_chunk as float32 array, sample_rate)
    """
    # Build talker inputs
    talker_input_embeds, talker_attention_mask, trailing_text_hiddens, tts_pad_embed = \
        self._build_talker_inputs(
            input_ids=input_ids,
            instruct_ids=instruct_ids,
            ref_ids=ref_ids,
            voice_clone_prompt=voice_clone_prompt,
            languages=languages,
            speakers=speakers,
            non_streaming_mode=non_streaming_mode,
        )

    # Multiple EOS tokens that can terminate generation
    # Some models may emit different EOS tokens depending on context
    eos_ids = {
        self.config.talker_config.codec_eos_token_id,  # Primary codec EOS
        2150,    # Codec EOS (model-specific)
        2157,    # Secondary codec token
        151670,  # TTS special token
        self.config.tts_eos_token_id,   # 151673
        self.config.im_end_token_id,    # 151645
        151643,  # <|endoftext|>
    }

    # Build suppress_tokens list (same as in generate())
    vocab_size = self.config.talker_config.vocab_size
    suppress_tokens = [
        i for i in range(vocab_size - 1024, vocab_size)
        if i not in eos_ids
    ]

    # Mark step begin for CUDA graphs (required for torch.compile with reduce-overhead)
    torch.compiler.cudagraph_mark_step_begin()

    # Prefill: single forward pass to initialize KV cache
    out = self.talker.forward(
        inputs_embeds=talker_input_embeds,
        attention_mask=talker_attention_mask,
        use_cache=True,
        output_hidden_states=True,
        return_dict=True,
        trailing_text_hidden=trailing_text_hiddens,
        tts_pad_embed=tts_pad_embed,
        generation_step=None,
        past_hidden=None,
        past_key_values=None,
        subtalker_dosample=subtalker_dosample,
        subtalker_top_k=subtalker_top_k,
        subtalker_top_p=subtalker_top_p,
        subtalker_temperature=subtalker_temperature,
    )

    past_key_values = out.past_key_values
    past_hidden = out.past_hidden
    generation_step = out.generation_step

    # Debug removed for performance: prefill done

    # Sample first token from prefill logits
    last_logits = out.logits[:, -1, :]
    if do_sample:
        token = _sample_next_token(last_logits, temperature, top_k, top_p, suppress_tokens)
    else:
        token = torch.argmax(last_logits, dim=-1)

    # Extract ref_code for decoder context (if in ICL mode)
    # This provides stable context from the start, eliminating early voice artifacts
    ref_code_context: Optional[torch.Tensor] = None
    ref_code_frames: int = 0
    if voice_clone_prompt is not None:
        ref_code_list = voice_clone_prompt.get("ref_code", None)
        icl_mode_list = voice_clone_prompt.get("icl_mode", None)
        if ref_code_list is not None and icl_mode_list is not None:
            if ref_code_list[0] is not None and icl_mode_list[0]:
                ref_code_context = ref_code_list[0].to(self.talker.device)
                ref_code_frames = ref_code_context.shape[0]

    # Decode loop
    codes_buffer: list[torch.Tensor] = []
    decoded_tail: Optional[np.ndarray] = None
    frames_since_emit = 0
    total_frames_emitted = 0  # Track how many frames we've already emitted audio for

    # GPU-resident circular buffer for repetition penalty
    if repetition_penalty != 1.0:
        rp_window = repetition_penalty_window if repetition_penalty_window > 0 else max_frames
        rp_history = torch.full((1, rp_window), vocab_size, device=token.device, dtype=torch.long)
        rp_history[0, 0] = token[0]
        rp_step = 1

    for step_idx in range(max_frames):
        # Mark step begin for CUDA graphs to avoid tensor overwrite errors
        # This is required when using torch.compile with reduce-overhead mode
        torch.compiler.cudagraph_mark_step_begin()

        # Single-step forward
        step_out = self.talker.forward(
            input_ids=token.unsqueeze(1),
            use_cache=True,
            return_dict=True,
            output_hidden_states=False,  # Disabled: codec_ids accessed via hidden_states[1] still works
            past_key_values=past_key_values,
            past_hidden=past_hidden,
            generation_step=generation_step,
            trailing_text_hidden=trailing_text_hiddens,
            tts_pad_embed=tts_pad_embed,
            subtalker_dosample=subtalker_dosample,
            subtalker_top_k=subtalker_top_k,
            subtalker_top_p=subtalker_top_p,
            subtalker_temperature=subtalker_temperature,
        )

        # Update state for next iteration
        past_key_values = step_out.past_key_values
        past_hidden = step_out.past_hidden
        generation_step = step_out.generation_step

        # Get codec_ids from hidden_states tuple: (layer_outputs, codec_ids)
        codec_ids = step_out.hidden_states[1]  # [B, num_code_groups]

        # Check for EOS in first codebook
        # EOS token is out of range for speech tokenizer, so we must not include it
        if codec_ids[0, 0].item() in eos_ids:
            break

        # Keep on GPU to avoid CPU<->GPU transfers during decode
        codes_buffer.append(codec_ids[0].detach())

        # Sample next token for first codebook
        step_logits = step_out.logits[:, -1, :]

        # Apply repetition penalty via GPU-resident circular buffer
        if repetition_penalty != 1.0:
            presence = torch.zeros(1, vocab_size + 1, device=step_logits.device, dtype=torch.bool)
            presence.scatter_(1, rp_history, True)
            penalty_mask = presence[:, :vocab_size]
            penalized = torch.where(
                step_logits > 0,
                step_logits / repetition_penalty,
                step_logits * repetition_penalty,
            )
            step_logits = torch.where(penalty_mask, penalized, step_logits)

        if do_sample:
            token = _sample_next_token(step_logits, temperature, top_k, top_p, suppress_tokens)
        else:
            token = torch.argmax(step_logits, dim=-1)

        if repetition_penalty != 1.0:
            pos = rp_step % rp_window
            rp_history[0, pos] = token[0]
            rp_step += 1

        frames_since_emit += 1

        # Two-phase streaming: determine current phase settings
        total_frames_generated = len(codes_buffer)
        if first_chunk_emit_every > 0 and total_frames_generated < first_chunk_frames:
            # Phase 1: Aggressive settings for first chunk (lower latency)
            current_emit_every = first_chunk_emit_every
            current_decode_window = first_chunk_decode_window
            current_use_optimized = False  # Non-optimized allows flexible window size
        else:
            # Phase 2: Stable settings (better quality)
            current_emit_every = emit_every_frames
            current_decode_window = decode_window_frames
            current_use_optimized = use_optimized_decode

        if frames_since_emit < current_emit_every:
            continue
        frames_since_emit = 0

        # Decode window of codec frames to PCM
        start = max(0, len(codes_buffer) - current_decode_window)
        window_codes = torch.stack(codes_buffer[start:], dim=0)  # [T, num_code_groups]

        # Add ref_code as context prefix for stable decoder context from the start
        window, _ = _add_ref_code_context(
            window_codes, ref_code_context, ref_code_frames, current_decode_window
        )

        # Use optimized decode path when available
        # Pass pad_to_size to ensure fixed tensor size for torch.compile
        if current_use_optimized and hasattr(self.speech_tokenizer, 'decode_streaming'):
            wavs, sr = self.speech_tokenizer.decode_streaming(
                window.to(self.talker.device),
                use_optimized=True,
                pad_to_size=decode_window_frames,
            )
        else:
            wavs, sr = self.speech_tokenizer.decode([{"audio_codes": window.to(self.talker.device)}])
        # Debug removed for performance: decode time tracking

        wav = wavs[0].astype(np.float32)

        # Extract only new samples (tail of decoded window)
        # Use fixed upsample rate to avoid floating-point drift
        samples_per_frame = self.speech_tokenizer.get_decode_upsample_rate()
        step_samples = samples_per_frame * current_emit_every
        chunk = wav[-step_samples:] if step_samples > 0 else wav

        # Crossfade with previous chunk tail for smooth transition
        # Always blend boundaries to prevent clicks from sliding window re-decode artifacts
        blend_samples = overlap_samples
        if decoded_tail is not None:
            ov = min(blend_samples, len(decoded_tail), len(chunk))
            if ov > 0:
                head = _crossfade(decoded_tail[-ov:], chunk[:ov])
                chunk = np.concatenate([head, chunk[ov:]], axis=0)

        # Apply Hann fade-in to very first chunk to avoid pop at audio start
        # Always apply fade-in on first chunk to prevent pop
        blend_samples = overlap_samples
        if decoded_tail is None:
            fade_len = min(blend_samples, len(chunk))
            if fade_len > 0:
                t = np.arange(fade_len, dtype=np.float32) / max(fade_len - 1, 1)
                fade_in = 0.5 * (1 - np.cos(np.pi * t))
                chunk[:fade_len] *= fade_in

        # Save FULL chunk for next crossfade reference
        decoded_tail = chunk.copy()

        # Trim END of chunk - this region will be replaced by next chunk's crossfade
        # Don't trim if chunk would become too small
        if blend_samples > 0 and len(chunk) > blend_samples * 2:
            chunk = chunk[:-blend_samples]

        total_frames_emitted = len(codes_buffer)  # Mark these frames as emitted
        yield chunk, sr

    # Flush: decode only remaining frames that haven't been emitted yet
    remaining_frames = len(codes_buffer) - total_frames_emitted
    if remaining_frames > 0:
        # Decode a window that includes some context for quality
        context_frames = min(total_frames_emitted, decode_window_frames - remaining_frames)
        start_idx = total_frames_emitted - context_frames
        window_codes = torch.stack(codes_buffer[start_idx:], dim=0)

        # Add ref_code as context prefix for stable decoder context
        window, flush_ref_prefix_frames = _add_ref_code_context(
            window_codes, ref_code_context, ref_code_frames, decode_window_frames
        )

        wavs, sr = self.speech_tokenizer.decode([{"audio_codes": window.to(self.talker.device)}])
        wav = wavs[0].astype(np.float32)

        # Extract only the new samples (skip ref_code and context portions)
        skip_frames = flush_ref_prefix_frames + context_frames
        if skip_frames > 0:
            samples_per_frame = len(wav) / window.shape[0]
            skip_samples = int(skip_frames * samples_per_frame)
            wav = wav[skip_samples:]

        # Crossfade with previous tail
        # Always blend flush boundary
        blend_samples = overlap_samples
        if decoded_tail is not None and len(wav) > 0:
            ov = min(blend_samples, len(decoded_tail), len(wav))
            if ov > 0:
                head = _crossfade(decoded_tail[-ov:], wav[:ov])
                wav = np.concatenate([head, wav[ov:]], axis=0)

        # Apply fade-out at very end of audio to avoid pop on completion
        if blend_samples > 0 and len(wav) > blend_samples:
            fade_len = min(blend_samples, len(wav))
            t = np.arange(fade_len, dtype=np.float32) / max(fade_len - 1, 1)
            fade_out = 0.5 * (1 + np.cos(np.pi * t))  # Hann fade-out
            wav[-fade_len:] *= fade_out

        # Debug removed for performance: flush done
        yield wav, sr




M.Qwen3TTSForConditionalGeneration._build_talker_inputs = _build_talker_inputs
M.Qwen3TTSForConditionalGeneration.stream_generate_pcm = stream_generate_pcm


def stream_custom_voice(m, text, speaker, language="French", instruct=None, **kw):
    input_ids = m._tokenize_texts([m._build_assistant_text(text)])
    if instruct:
        instruct_ids = [m._tokenize_texts([m._build_instruct_text(instruct)])[0]]
    else:
        instruct_ids = [None]
    gen_kwargs = m._merge_generate_kwargs(**kw)
    for pcm_f32, sr in m.model.stream_generate_pcm(
            input_ids=input_ids, instruct_ids=instruct_ids,
            languages=[language], speakers=[speaker],
            non_streaming_mode=True,
            use_optimized_decode=False,
            emit_every_frames=12,
            decode_window_frames=80,
            overlap_samples=512,
            first_chunk_emit_every=5,
            first_chunk_decode_window=40,
            first_chunk_frames=40,
            repetition_penalty=1.05,
            max_frames=int(gen_kwargs.get("max_new_tokens", 4096) or 4096)):
        yield np.asarray(pcm_f32, dtype=np.float32), int(sr)

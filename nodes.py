from comfy_api.latest import io
import os
import sys
import hashlib
import json
from pathlib import Path
import folder_paths
import torch

from .irodori_tts.inference_runtime import (
    RuntimeKey,
    SamplingRequest,
    get_cached_runtime,
    list_available_runtime_devices,
    list_available_runtime_precisions,
    default_runtime_device
)


def _available_devices():
    return list_available_runtime_devices()

def _available_precisions(device="cuda"):
    try:
        return list_available_runtime_precisions(device)
    except:
        return ["fp32", "bf16"]


IO_IRODORI_MODEL = io.Custom("IRODORI_V2_MODEL")
IO_IRODORI_REF_CONFIG = io.Custom("IRODORI_V2_REF_CONFIG")
IO_IRODORI_CFG_CONFIG = io.Custom("IRODORI_V2_CFG_CONFIG")
IO_IRODORI_RESCALE_CONFIG = io.Custom("IRODORI_V2_RESCALE_CONFIG")

CATEGORY = "DialogueTTS/IrodoriTTS-v2"
V2_CODEC_REPO_ID = "Aratako/Semantic-DACVAE-Japanese-32dim"
V2_CODEC_FILENAME = "Semantic-DACVAE-Japanese-32dim-weights.pth"


def _unpack_optional_configs(ref_audio_config={}, cfg_config={}, rescale_config={}):
    ref_wav = ref_audio_config.get("ref_wav", None)
    ref_normalize_db = ref_audio_config.get("ref_normalize_db", None)
    ref_ensure_max = ref_audio_config.get("ref_ensure_max", False)
    max_ref_seconds = ref_audio_config.get("max_ref_seconds", 30.0)

    cfg_scale_override = cfg_config.get("cfg_scale_override", None)
    cfg_min_t = cfg_config.get("cfg_min_t", 0.5)
    cfg_max_t = cfg_config.get("cfg_max_t", 1.0)

    truncation_factor = rescale_config.get("truncation_factor", None)
    rescale_k = rescale_config.get("rescale_k", None)
    rescale_sigma = rescale_config.get("rescale_sigma", None)
    speaker_kv_scale = rescale_config.get("speaker_kv_scale", None)
    speaker_kv_min_t = rescale_config.get("speaker_kv_min_t", 0.9)
    speaker_kv_max_layers = rescale_config.get("speaker_kv_max_layers", None)

    return {
        "ref_wav": ref_wav,
        "no_ref": ref_wav is None,
        "ref_normalize_db": ref_normalize_db,
        "ref_ensure_max": ref_ensure_max,
        "max_ref_seconds": max_ref_seconds,
        "cfg_scale_override": cfg_scale_override,
        "cfg_min_t": cfg_min_t,
        "cfg_max_t": cfg_max_t,
        "truncation_factor": truncation_factor,
        "rescale_k": rescale_k,
        "rescale_sigma": rescale_sigma,
        "speaker_kv_scale": speaker_kv_scale,
        "speaker_kv_min_t": speaker_kv_min_t,
        "speaker_kv_max_layers": speaker_kv_max_layers,
    }


def _build_sampling_request(
    *,
    text,
    seed,
    num_steps,
    cfg_guidance_mode,
    cfg_scale_text,
    cfg_scale_speaker,
    context_kv_cache,
    ref_audio_config={},
    cfg_config={},
    rescale_config={},
):
    config = _unpack_optional_configs(ref_audio_config, cfg_config, rescale_config)
    return SamplingRequest(
        text=text,
        ref_wav=config["ref_wav"],
        ref_latent=None,
        no_ref=config["no_ref"],
        ref_normalize_db=config["ref_normalize_db"],
        ref_ensure_max=config["ref_ensure_max"],
        num_candidates=1,
        decode_mode="sequential",
        seconds=30.0,
        max_ref_seconds=config["max_ref_seconds"],
        max_text_len=None,
        num_steps=num_steps,
        cfg_scale_text=cfg_scale_text,
        cfg_scale_speaker=cfg_scale_speaker,
        cfg_guidance_mode=cfg_guidance_mode,
        cfg_scale=config["cfg_scale_override"],
        cfg_min_t=config["cfg_min_t"],
        cfg_max_t=config["cfg_max_t"],
        truncation_factor=config["truncation_factor"],
        rescale_k=config["rescale_k"],
        rescale_sigma=config["rescale_sigma"],
        context_kv_cache=context_kv_cache,
        speaker_kv_scale=config["speaker_kv_scale"],
        speaker_kv_min_t=config["speaker_kv_min_t"],
        speaker_kv_max_layers=config["speaker_kv_max_layers"],
        seed=seed,
        trim_tail=True,
    )


def _parse_dialogue_segments_json(dialogue_segments_json, max_utterances, skip_empty_text):
    try:
        data = json.loads(str(dialogue_segments_json or ""))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid dialogue_segments_json: {exc}") from exc

    utterances = data.get("utterances")
    if not isinstance(utterances, list):
        raise ValueError("dialogue_segments_json must contain an utterances list.")

    parsed = []
    limit = max(0, int(max_utterances))
    for fallback_index, item in enumerate(utterances, start=1):
        if limit and len(parsed) >= limit:
            break
        if not isinstance(item, dict):
            raise ValueError(f"utterances[{fallback_index - 1}] must be an object.")

        speaker = str(item.get("speaker", "")).strip()
        text = str(item.get("text", "")).strip()
        if skip_empty_text and not text:
            continue

        parsed.append(
            {
                "index": item.get("index", fallback_index),
                "speaker": speaker,
                "text": text,
            }
        )

    return parsed


def _resolve_dialogue_ref_config(speaker, speaker_a_key, speaker_b_key, ref_audio_config_a, ref_audio_config_b, unknown_speaker_policy):
    if speaker == speaker_a_key:
        return ref_audio_config_a
    if speaker == speaker_b_key:
        return ref_audio_config_b
    if unknown_speaker_policy == "skip":
        return None
    if unknown_speaker_policy == "use_a":
        return ref_audio_config_a
    if unknown_speaker_policy == "use_b":
        return ref_audio_config_b
    raise ValueError(f"Unknown dialogue speaker: {speaker!r}")


def _resolve_dialogue_seed(seed, seed_mode, utterance_offset):
    if seed_mode == "fixed":
        return int(seed)
    if seed_mode == "randomize":
        return None
    return int(seed) + int(utterance_offset)


def _apply_edge_fade(audio, sample_rate, fade_ms):
    fade_samples = int(int(sample_rate) * max(0, int(fade_ms)) / 1000)
    if fade_samples <= 0 or audio.shape[-1] <= 1:
        return audio

    fade_samples = min(fade_samples, max(1, int(audio.shape[-1]) // 2))
    if fade_samples <= 0:
        return audio

    faded = audio.clone()
    fade_in = torch.linspace(
        0.0,
        1.0,
        fade_samples,
        dtype=faded.dtype,
        device=faded.device,
    ).view(1, -1)
    fade_out = torch.linspace(
        1.0,
        0.0,
        fade_samples,
        dtype=faded.dtype,
        device=faded.device,
    ).view(1, -1)
    faded[:, :fade_samples] = faded[:, :fade_samples] * fade_in
    faded[:, -fade_samples:] = faded[:, -fade_samples:] * fade_out
    return faded


def _resolve_v2_codec_path() -> str:
    checkpoint_codec = folder_paths.get_full_path("checkpoints", V2_CODEC_FILENAME)
    if checkpoint_codec:
        return checkpoint_codec

    models_dir = getattr(folder_paths, "models_dir", None)
    if models_dir:
        for relative in (
            Path("vae") / V2_CODEC_FILENAME,
            Path("checkpoints") / V2_CODEC_FILENAME,
        ):
            candidate = Path(models_dir) / relative
            if candidate.is_file():
                return str(candidate)

    cache_root = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--Aratako--Semantic-DACVAE-Japanese-32dim"
        / "snapshots"
    )
    if cache_root.is_dir():
        for candidate in sorted(cache_root.glob("*/weights.pth"), reverse=True):
            if candidate.is_file():
                return str(candidate)

    return V2_CODEC_REPO_ID



# ===============================================
# Irodori Model Loader
# ===============================================
class IrodoriTTS_v2ModelLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        model_files = folder_paths.get_filename_list("checkpoints")
        devices = _available_devices()
        precisions = _available_precisions()

        return io.Schema(
            node_id="IrodoriTTS-v2ModelLoader", 
            display_name="IrodoriTTS-v2 Model Loader", 
            category=CATEGORY, 
            inputs=[
                io.Combo.Input("model_name", options=model_files), 
                io.Combo.Input("model_device", options=devices), 
                io.Combo.Input("model_precision", options=precisions), 
                io.Combo.Input("codec_device", options=devices), 
                io.Combo.Input("codec_precision", options=precisions), 
                io.Boolean.Input("enable_watermark", default=False), 
            ], 
            outputs=[
                IO_IRODORI_MODEL.Output(display_name="irodori_model")
            ], 
        )
    
    @classmethod
    def execute(
        cls, 
        model_name: str, 
        model_device: str, 
        model_precision: str, 
        codec_device: str, 
        codec_precision: str, 
        enable_watermark: bool
    ):
        checkpoint_path = folder_paths.get_full_path("checkpoints", model_name)
        if not checkpoint_path:
            checkpoint_path = model_name
        
        runtime_key = RuntimeKey(
            checkpoint=checkpoint_path,
            model_device=model_device,
            codec_repo=_resolve_v2_codec_path(),
            model_precision=model_precision,
            codec_device=codec_device,
            codec_precision=codec_precision,
            enable_watermark=enable_watermark,
            compile_model=False,
            compile_dynamic=False,
        )
        
        runtime, _ = get_cached_runtime(runtime_key)
        return io.NodeOutput(runtime)

# ===============================================
# Irodori Reference Audio
# ===============================================
class IrodoriTTS_v2ReferenceAudio(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        input_dir = folder_paths.get_input_directory()
        files = folder_paths.filter_files_content_types(os.listdir(input_dir), ["audio", "video"])

        return io.Schema(
            node_id="IrodoriTTS-v2ReferenceAudio",
            display_name="IrodoriTTS-v2 Reference Audio",
            category=CATEGORY, 
            inputs=[
                io.Combo.Input("ref_audio", options=files), 
                io.Boolean.Input("normalize_ref_audio", default=False), 
                io.Float.Input("max_ref_seconds", default=30.0, min=1.0, max=120.0, step=1.0), 
                
            ], 
            outputs=[IO_IRODORI_REF_CONFIG.Output(display_name="ref_audio_config")],
        )

    @classmethod
    def execute(cls, ref_audio, normalize_ref_audio, max_ref_seconds):
        audio_path = folder_paths.get_annotated_filepath(ref_audio)
        config = {
            "ref_wav": audio_path, 
            "ref_normalize_db": -16.0 if normalize_ref_audio else None, 
            "ref_ensure_max": normalize_ref_audio, 
            "max_ref_seconds": max_ref_seconds, 
        }
        
        return io.NodeOutput(config)
    
    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        ref_audio = kwargs.get("ref_audio")
        audio_path = folder_paths.get_annotated_filepath(ref_audio)
        m = hashlib.sha256()
        with open(audio_path, "rb") as f:
            m.update(f.read())
        return m.digest().hex()
    
    @classmethod
    def validate_inputs(cls, **kwargs):
        ref_audio = kwargs.get("ref_audio")
        if not folder_paths.exists_annotated_filepath(ref_audio):
            return "Invalid audio file: {}".format(ref_audio)
        return True
    

# ===============================================
# Irodori Advanced CFG
# ===============================================
class IrodoriTTS_v2AdvancedCFG(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="IrodoriTTS-v2AdvancedCFG", 
            display_name="IrodoriTTS-v2 Advanced CFG", 
            category=CATEGORY, 
            inputs=[
                io.Float.Input("cfg_scale_override", default=-1.0, min=-1.0, max=10.0, step=0.1, tooltip="Set to > 0 to override"), 
                io.Float.Input("cfg_min_t", default=0.5, min=0.0, max=1.0, step=0.05), 
                io.Float.Input("cfg_max_t", default=1.0, min=0.0, max=1.0, step=0.05), 
            ], 
            outputs=[
                IO_IRODORI_CFG_CONFIG.Output(display_name="cfg_config"), 
            ], 
        )
    
    @classmethod
    def execute(cls, cfg_scale_override: float, cfg_min_t: float, cfg_max_t: float):
        config = {
            "cfg_scale_override": cfg_scale_override if cfg_scale_override > 0 else None, 
            "cfg_min_t": cfg_min_t, 
            "cfg_max_t": cfg_max_t, 
        }
        return io.NodeOutput(config)


# ===============================================
# Irodori Rescale Config
# ===============================================
class IrodoriTTS_v2RescaleConfig(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="IrodoriTTS-v2RescaleConfig", 
            display_name="IrodoriTTS-v2 Rescale Config", 
            category=CATEGORY, 
            inputs=[
                io.Float.Input(
                    "truncation_factor", 
                    default=-1.0, 
                    min=-1.0, 
                    max=1.0, 
                    step=0.05, 
                    tooltip="Set > 0 to enable"
                ), 
                io.Float.Input(
                    "rescale_k", 
                    default=-1.0, 
                    min=-1.0, 
                    max=10.0, 
                    step=0.1, 
                    tooltip="Set > 0 to enable. Typical range: 2.0 - 6.0"
                ), 
                io.Float.Input(
                    "rescale_sigma", 
                    default=-1.0, 
                    min=-1.0, 
                    max=3.0, 
                    step=0.01, 
                    tooltip="Set > 0 to enable (requires rescale_k). Typical range: 0.1 - 1.0"
                ), 
                io.Float.Input(
                    "speaker_kv_scale", 
                    default=-1.0, 
                    min=-1.0, 
                    max=5.0, 
                    step=0.05, 
                    tooltip="Set > 0 to scale speaker K/V strength. Typical range: 1.0 - 3.0"
                ), 
                io.Float.Input(
                    "speaker_kv_min_t", 
                    default=0.9, 
                    min=0.0, 
                    max=1.0, 
                    step=0.05, 
                    tooltip="KV scale is applied while t >= this value, then reverted"
                ), 
                io.Int.Input(
                    "speaker_kv_max_layers", 
                    default=-1, 
                    min=-1, 
                    max=32, 
                    step=1, 
                    tooltip="Max transformer layers to apply speaker_kv_scale to. -1 = all layers"
                ), 
            ], 
            outputs=[
                IO_IRODORI_RESCALE_CONFIG.Output(display_name="rescale_config"), 
            ], 
        )
    
    @classmethod
    def execute(cls, truncation_factor, rescale_k, rescale_sigma, speaker_kv_scale, speaker_kv_min_t, speaker_kv_max_layers):
        config = {
            "truncation_factor": truncation_factor if truncation_factor > 0 else None,
            "rescale_k": rescale_k if rescale_k > 0 else None,
            "rescale_sigma": rescale_sigma if rescale_sigma > 0 else None,
            "speaker_kv_scale": speaker_kv_scale if speaker_kv_scale > 0 else None,
            "speaker_kv_min_t": speaker_kv_min_t,
            "speaker_kv_max_layers": speaker_kv_max_layers if speaker_kv_max_layers >= 0 else None,
        }
        return io.NodeOutput(config)
    
    
# ===============================================
# Irodori Sampler
# ===============================================
class IrodoriTTS_v2Sampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="IrodoriTTS-v2Sampler", 
            display_name="IrodoriTTS-v2 Sampler", 
            category=CATEGORY, 
            inputs=[
                IO_IRODORI_MODEL.Input("model", display_name="irodori_model"), 
                io.String.Input("text", multiline=True), 
                io.Int.Input("seed", default=0, min=0, max=sys.maxsize), 
                io.Int.Input("num_steps", default=40, min=1, max=120), 
                io.Combo.Input("cfg_guidance_mode", options=["independent", "joint", "alternating"], default="independent"), 
                io.Float.Input("cfg_scale_text", default=3.0, min=0.0, max=10.0, step=0.1), 
                io.Float.Input("cfg_scale_speaker", default=5.0, min=0.0, max=10.0, step=0.1), 
                io.Boolean.Input("context_kv_cache", default=True), 
                
                IO_IRODORI_REF_CONFIG.Input("ref_audio_config", display_name="ref_audio_config", optional=True), 
                IO_IRODORI_CFG_CONFIG.Input("cfg_config", display_name="cfg_config", optional=True), 
                IO_IRODORI_RESCALE_CONFIG.Input("rescale_config", display_name="rescale_config", optional=True), 
            ], 
            outputs=[
                io.Audio.Output(), 
            ], 
        )
    
    @classmethod
    def execute(cls, model, text, seed, num_steps, cfg_guidance_mode, cfg_scale_text, cfg_scale_speaker, context_kv_cache, ref_audio_config={}, cfg_config={}, rescale_config={}):
        req = _build_sampling_request(
            text=text,
            num_steps=num_steps,
            cfg_guidance_mode=cfg_guidance_mode,
            cfg_scale_text=cfg_scale_text,
            cfg_scale_speaker=cfg_scale_speaker,
            context_kv_cache=context_kv_cache,
            seed=seed,
            ref_audio_config=ref_audio_config,
            cfg_config=cfg_config,
            rescale_config=rescale_config,
        )
        
        result = model.synthesize(req, log_fn=print)
        
        audio_tensor = result.audio
        # Result is [channels, samples]. ComfyUI expects [batch, channels, samples]
        if audio_tensor.dim() == 2:
            audio_tensor = audio_tensor.unsqueeze(0)

        out_audio = {"waveform": audio_tensor, "sample_rate": result.sample_rate}
        return io.NodeOutput(out_audio)


# ===============================================
# Irodori Dialogue TTS
# ===============================================
class IrodoriTTS_v2DialogueTTS(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="IrodoriTTS-v2DialogueTTS",
            display_name="IrodoriTTS-v2 Dialogue TTS",
            category=CATEGORY,
            inputs=[
                IO_IRODORI_MODEL.Input("model", display_name="irodori_model"),
                io.String.Input("dialogue_segments_json", multiline=True),
                io.String.Input("speaker_a_key", default="A"),
                io.String.Input("speaker_b_key", default="B"),
                io.Int.Input("seed", default=0, min=0, max=sys.maxsize),
                io.Combo.Input(
                    "seed_mode",
                    options=["increment_per_utterance", "fixed", "randomize"],
                    default="increment_per_utterance",
                ),
                io.Int.Input("num_steps", default=40, min=1, max=120),
                io.Combo.Input("cfg_guidance_mode", options=["independent", "joint", "alternating"], default="independent"),
                io.Float.Input("cfg_scale_text", default=3.0, min=0.0, max=10.0, step=0.1),
                io.Float.Input("cfg_scale_speaker", default=5.0, min=0.0, max=10.0, step=0.1),
                io.Boolean.Input("context_kv_cache", default=True),
                io.Int.Input("silence_ms", default=300, min=0, max=10000, step=10),
                io.Int.Input("fade_ms", default=10, min=0, max=200, step=1),
                io.Int.Input("max_utterances", default=100, min=0, max=1000, step=1),
                io.Boolean.Input("skip_empty_text", default=True),
                io.Combo.Input(
                    "unknown_speaker_policy",
                    options=["error", "skip", "use_a", "use_b"],
                    default="error",
                ),
                IO_IRODORI_REF_CONFIG.Input("ref_audio_config_a", display_name="ref_audio_config_a", optional=True),
                IO_IRODORI_REF_CONFIG.Input("ref_audio_config_b", display_name="ref_audio_config_b", optional=True),
                IO_IRODORI_CFG_CONFIG.Input("cfg_config", display_name="cfg_config", optional=True),
                IO_IRODORI_RESCALE_CONFIG.Input("rescale_config", display_name="rescale_config", optional=True),
            ],
            outputs=[
                io.Audio.Output(display_name="combined_audio"),
                io.String.Output(display_name="dialogue_timing_json"),
                io.String.Output(display_name="log_text"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model,
        dialogue_segments_json,
        speaker_a_key,
        speaker_b_key,
        seed,
        seed_mode,
        num_steps,
        cfg_guidance_mode,
        cfg_scale_text,
        cfg_scale_speaker,
        context_kv_cache,
        silence_ms,
        fade_ms,
        max_utterances,
        skip_empty_text,
        unknown_speaker_policy,
        ref_audio_config_a={},
        ref_audio_config_b={},
        cfg_config={},
        rescale_config={},
    ):
        utterances = _parse_dialogue_segments_json(
            dialogue_segments_json,
            max_utterances=max_utterances,
            skip_empty_text=skip_empty_text,
        )
        speaker_a_key = str(speaker_a_key or "A").strip()
        speaker_b_key = str(speaker_b_key or "B").strip()

        sample_rate = int(getattr(getattr(model, "codec", None), "sample_rate", 24000))
        silence_samples = int(sample_rate * max(0, int(silence_ms)) / 1000)
        silence = None

        audio_parts = []
        timing_items = []
        log_lines = []
        cursor_samples = 0
        synthesized_count = 0
        total_utterances = len(utterances)

        for offset, utterance in enumerate(utterances):
            speaker = utterance["speaker"]
            ref_config = _resolve_dialogue_ref_config(
                speaker=speaker,
                speaker_a_key=speaker_a_key,
                speaker_b_key=speaker_b_key,
                ref_audio_config_a=ref_audio_config_a or {},
                ref_audio_config_b=ref_audio_config_b or {},
                unknown_speaker_policy=unknown_speaker_policy,
            )
            if ref_config is None:
                message = (
                    f"[IrodoriTTS-v2 Dialogue TTS] "
                    f"skip utterance {offset + 1}/{total_utterances} "
                    f"index={utterance['index']} speaker={speaker}"
                )
                print(message)
                log_lines.append(message)
                continue

            utterance_seed = _resolve_dialogue_seed(seed, seed_mode, offset)
            start_message = (
                f"[IrodoriTTS-v2 Dialogue TTS] "
                f"processing utterance {offset + 1}/{total_utterances} "
                f"index={utterance['index']} speaker={speaker} seed={utterance_seed}"
            )
            print(start_message)
            log_lines.append(start_message)

            req = _build_sampling_request(
                text=utterance["text"],
                num_steps=num_steps,
                cfg_guidance_mode=cfg_guidance_mode,
                cfg_scale_text=cfg_scale_text,
                cfg_scale_speaker=cfg_scale_speaker,
                context_kv_cache=context_kv_cache,
                seed=utterance_seed,
                ref_audio_config=ref_config,
                cfg_config=cfg_config or {},
                rescale_config=rescale_config or {},
            )

            result = model.synthesize(req, log_fn=print)
            if int(result.sample_rate) != sample_rate:
                raise ValueError(
                    f"Unexpected sample rate change: {result.sample_rate} != {sample_rate}"
                )

            audio = result.audio
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
            if audio.dim() != 2:
                raise ValueError(f"Expected utterance audio [channels, samples], got shape {tuple(audio.shape)}")

            audio = _apply_edge_fade(audio, sample_rate=sample_rate, fade_ms=fade_ms)

            if silence_samples > 0 and synthesized_count > 0:
                if silence is None or silence.shape[0] != audio.shape[0]:
                    silence = torch.zeros(
                        (audio.shape[0], silence_samples),
                        dtype=audio.dtype,
                        device=audio.device,
                    )
                audio_parts.append(silence)
                cursor_samples += silence_samples

            start_samples = cursor_samples
            audio_parts.append(audio)
            cursor_samples += int(audio.shape[-1])
            end_samples = cursor_samples
            synthesized_count += 1

            timing_items.append(
                {
                    "index": utterance["index"],
                    "speaker": speaker,
                    "start_sec": start_samples / sample_rate,
                    "end_sec": end_samples / sample_rate,
                    "text": utterance["text"],
                }
            )
            done_message = (
                f"[IrodoriTTS-v2 Dialogue TTS] "
                f"completed utterance {offset + 1}/{total_utterances} "
                f"index={utterance['index']} speaker={speaker} "
                f"seed={utterance_seed} samples={audio.shape[-1]}"
            )
            print(done_message)
            log_lines.append(done_message)

        if audio_parts:
            combined = torch.cat(audio_parts, dim=-1)
        else:
            combined = torch.zeros((1, 1), dtype=torch.float32)

        out_audio = {"waveform": combined.unsqueeze(0), "sample_rate": sample_rate}
        timing_json = json.dumps(
            {
                "schema": "kantan.dialogue_timing.v1",
                "sample_rate": sample_rate,
                "silence_ms": int(silence_ms),
                "fade_ms": int(fade_ms),
                "utterances": timing_items,
            },
            ensure_ascii=False,
        )
        log_text = "\n".join(log_lines)

        return io.NodeOutput(out_audio, timing_json, log_text)


# ===============================================
# Irodori Emoji Selector
# ===============================================
class IrodoriTTS_v2EmojiSelector(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="IrodoriTTS-v2EmojiSelector", 
            display_name="IrodoriTTS-v2 Emoji Selector", 
            category=CATEGORY, 
            inputs=[], 
            outputs=[], 
        )
    
    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput()
    


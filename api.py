import os
import re
import sys
import time
import json
import logging
import subprocess
import tempfile
import multiprocessing as mp
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Any
from concurrent.futures import ProcessPoolExecutor
import asyncio
from collections import defaultdict
import traceback
import struct

# ---------- FORÇA SPAWN PARA MULTIPROCESSAMENTO ----------
mp.set_start_method('spawn', force=True)

# ---------- DEPENDÊNCIAS ----------
try:
    import torch
    import librosa
    from omnivoice import OmniVoice, OmniVoiceGenerationConfig
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "omnivoice", "librosa", "torch"])
    import torch
    import librosa
    from omnivoice import OmniVoice, OmniVoiceGenerationConfig

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

# ---------- CONFIGURAÇÃO DE LOGS (APENAS INFO) ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(processName)s | %(name)s | %(message)s",
)
logger = logging.getLogger("omnivoice-api")

# ---------- DIRETÓRIOS ----------
BASE_DIR = Path("/app")
VOICES_DIR = BASE_DIR / "voices"
AMBIENT_DIR = BASE_DIR / "ambient"
EFFECTS_DIR = BASE_DIR / "effects"
VOICES_DIR.mkdir(exist_ok=True)
AMBIENT_DIR.mkdir(exist_ok=True)
EFFECTS_DIR.mkdir(exist_ok=True)

# ---------- CONTADOR DE NÚCLEOS ----------
_cpu_counter = mp.Value('i', 0)
_cpu_lock = mp.Lock()

# ---------- WORKERS ----------
TTS_WORKERS = 1
MIX_WORKERS = int(os.getenv("MIX_WORKERS", 4))
logger.info(f"Workers: TTS={TTS_WORKERS}, Mix={MIX_WORKERS}")

# ---------- TAXA DE AMOSTRAGEM ----------
TARGET_SR = 24000

# ============================================================================
# CABEÇALHO WAV MANUAL (RÁPIDO)
# ============================================================================
def write_wav_header(sample_rate: int, num_samples: int, num_channels: int = 1, bits: int = 16) -> bytes:
    byte_rate = sample_rate * num_channels * bits // 8
    block_align = num_channels * bits // 8
    data_size = num_samples * num_channels * bits // 8
    riff_size = 36 + data_size
    return (
        b'RIFF' + struct.pack('<I', riff_size) + b'WAVE' +
        b'fmt ' + struct.pack('<I', 16) + struct.pack('<H', 1) +
        struct.pack('<H', num_channels) + struct.pack('<I', sample_rate) +
        struct.pack('<I', byte_rate) + struct.pack('<H', block_align) +
        struct.pack('<H', bits) + b'data' + struct.pack('<I', data_size)
    )

# ============================================================================
# PRÉ-PROCESSAMENTO DE ÁUDIO
# ============================================================================
def preprocess_audio(file_path: Path, target_sr: int = TARGET_SR) -> np.ndarray:
    audio, sr = librosa.load(str(file_path), sr=None, mono=True)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio.astype(np.float32)

# ============================================================================
# INICIALIZADOR TTS (GPU) – COM OTIMIZAÇÕES
# ============================================================================
def _init_tts_worker():
    with _cpu_lock:
        cpu_id = _cpu_counter.value
        _cpu_counter.value += 1
    total_cpus = os.cpu_count()
    cpu_id = cpu_id % total_cpus
    try:
        os.sched_setaffinity(0, {cpu_id})
        logger.info(f"TTS Worker no núcleo {cpu_id}")
    except Exception:
        pass

    logger.info("📥 Carregando OmniVoice (float16)...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map=device, dtype=dtype)
    
    # 🔥 OTIMIZAÇÃO 1: Compilação do modelo (ganho 1.2x-2x)
    logger.info("⚡ Compilando modelo com torch.compile...")
    model = torch.compile(model)

    logger.info(f"✅ OmniVoice carregado em {device} com {dtype}")
    mod = sys.modules['__main__']
    mod._omnivoice_model = model
    mod._sample_rate = TARGET_SR

    # Pré‑carga das vozes
    logger.info("📥 Pré‑carregando vozes...")
    voice_cache = {}
    for name, path in VOICE_REF_PATHS.items():
        try:
            audio = preprocess_audio(path)
            voice_cache[name] = (audio, TARGET_SR)
            logger.info(f"   ✅ {name} ({len(audio)/TARGET_SR:.1f}s)")
        except Exception as e:
            logger.error(f"   ❌ {name}: {e}")
    mod._ref_audio_cache = voice_cache

    # Pré‑carga dos efeitos
    logger.info("📥 Pré‑carregando efeitos...")
    effect_cache = {}
    for name, path in VOICE_REF_PATHS.items():
        for wav in path.parent.glob("*.wav"):
            if wav == path:
                continue
            try:
                audio = preprocess_audio(wav)
                effect_cache[wav.name] = (audio, TARGET_SR)
                logger.info(f"   ✅ Efeito: {wav.name}")
            except Exception:
                pass
    for wav in EFFECTS_DIR.glob("*.wav"):
        try:
            audio = preprocess_audio(wav)
            effect_cache[wav.name] = (audio, TARGET_SR)
            logger.info(f"   ✅ Efeito global: {wav.name}")
        except Exception:
            pass
    mod._effect_cache = effect_cache
    logger.info(f"✅ {len(voice_cache)} vozes, {len(effect_cache)} efeitos carregados")

# ============================================================================
# INICIALIZADOR DE MIXAGEM (CPU)
# ============================================================================
def _init_mix_worker():
    with _cpu_lock:
        cpu_id = _cpu_counter.value
        _cpu_counter.value += 1
    total_cpus = os.cpu_count()
    cpu_id = cpu_id % total_cpus
    try:
        os.sched_setaffinity(0, {cpu_id})
        logger.info(f"Mix Worker no núcleo {cpu_id}")
    except Exception:
        pass

# ============================================================================
# MAPEAMENTO DE VOZES
# ============================================================================
VOICE_REF_PATHS: Dict[str, Path] = {}
def load_all_voices():
    for item in VOICES_DIR.iterdir():
        if not item.is_dir():
            continue
        name = item.name
        candidates = list(item.glob(f"{name}.wav")) + list(item.glob(f"{name}.WAV"))
        if candidates:
            VOICE_REF_PATHS[name] = candidates[0]
        elif list(item.glob("reference.wav")):
            VOICE_REF_PATHS[name] = item / "reference.wav"
        elif list(item.glob("*.wav")):
            VOICE_REF_PATHS[name] = list(item.glob("*.wav"))[0]
    for wav in VOICES_DIR.glob("*.wav"):
        name = wav.stem
        if name not in VOICE_REF_PATHS:
            VOICE_REF_PATHS[name] = wav
    logger.info(f"✅ {len(VOICE_REF_PATHS)} vozes mapeadas")

load_all_voices()

# ============================================================================
# FUNÇÕES TTS (OTIMIZADAS)
# ============================================================================
def get_ref_audio(voice_name: str):
    mod = sys.modules['__main__']
    cache = mod._ref_audio_cache
    if voice_name not in cache:
        raise ValueError(f"Voz '{voice_name}' não encontrada")
    return cache[voice_name]

def get_effect_audio(effect_name: str):
    mod = sys.modules['__main__']
    cache = mod._effect_cache
    if effect_name not in cache:
        raise FileNotFoundError(f"Efeito '{effect_name}' não encontrado")
    return cache[effect_name]

def synthesize_omnivoice(text: str, voice_name: str, speed: float, guidance_scale: float):
    mod = sys.modules['__main__']
    model = mod._omnivoice_model
    sample_rate = mod._sample_rate
    ref_audio = get_ref_audio(voice_name)

    duration = max(0.5, len(text) * 0.1 / speed)
    duration = min(duration, 30.0)

    # 🔥 OTIMIZAÇÃO 2: num_step reduzido para 16 (ganho 2x-4x)
    gen_config = OmniVoiceGenerationConfig(
        num_step=16,
        guidance_scale=guidance_scale,
    )

    with torch.no_grad():
        result = model.generate(
            text=text,
            ref_audio=ref_audio,
            generation_config=gen_config,
            duration=duration,
        )

    if isinstance(result, (list, tuple)):
        audio = result[0]
    else:
        audio = result
    if torch.is_tensor(audio):
        audio = audio.cpu().numpy()
    if audio.ndim > 1:
        audio = audio.squeeze()
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    return audio, sample_rate

def process_fragments(voice_name, text, speed, guidance_scale, effects, speakers):
    t_start = time.perf_counter()
    is_dialog = bool(speakers)
    if not is_dialog:
        if not voice_name:
            raise ValueError("voice_name é obrigatório")
        speaker_map = {None: (voice_name, speed, guidance_scale)}
        current_role = None
    else:
        speaker_map = {}
        for spk in speakers:
            gs = spk.get('guidance_scale', guidance_scale)
            spd = spk.get('speed', speed)
            speaker_map[spk['role']] = (spk['voice'], spd, gs)
        current_role = None

    parts = re.split(r'(\[.*?\])', text)
    parts = [p.strip() for p in parts if p.strip()]
    segments = []

    for part in parts:
        if is_dialog and part.startswith('[') and part.endswith(']'):
            role = part[1:-1]
            if role in speaker_map:
                current_role = role
            continue

        if part in effects:
            effect_name = effects[part]
            audio, sr = get_effect_audio(effect_name)
            pcm = (audio * 32767).astype(np.int16).tobytes()
            segments.append({'pcm_bytes': pcm, 'sample_rate': sr, 'num_samples': len(audio)})
            continue

        if is_dialog:
            if current_role is None:
                raise ValueError("Speaker não definido")
            v_name, spd, gs = speaker_map[current_role]
        else:
            v_name = voice_name
            spd = speed
            gs = guidance_scale

        audio, sr = synthesize_omnivoice(part, v_name, spd, gs)
        pcm = (audio * 32767).astype(np.int16).tobytes()
        segments.append({'pcm_bytes': pcm, 'sample_rate': sr, 'num_samples': len(audio)})

    logger.info(f"🔊 TTS concluído em {time.perf_counter()-t_start:.3f}s | {len(segments)} segmentos")
    return segments

# ============================================================================
# MIXAGEM (CPU)
# ============================================================================
def mix_and_export_task(segments_data, ambient_cfg, target_rate=24000):
    t0 = time.perf_counter()
    temp_files = []
    ffmpeg_cmd = ["ffmpeg", "-y"]

    try:
        for data in segments_data:
            pcm = data['pcm_bytes']
            sr = data['sample_rate']
            ns = data.get('num_samples', len(pcm)//2)
            header = write_wav_header(sr, ns, 1, 16)
            wav_data = header + pcm
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                f.write(wav_data)
                temp_files.append(f.name)

        if ambient_cfg.get('enabled') and ambient_cfg.get('file'):
            amb_path = AMBIENT_DIR / f"{ambient_cfg['file']}.wav"
            if amb_path.exists():
                audio = preprocess_audio(amb_path, target_rate)
                pcm = (audio * 32767).astype(np.int16).tobytes()
                header = write_wav_header(target_rate, len(audio), 1, 16)
                wav_data = header + pcm
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                    f.write(wav_data)
                    temp_files.append(f.name)

        if not temp_files:
            raise ValueError("Nenhum áudio para mixar")

        for f in temp_files:
            ffmpeg_cmd.extend(["-i", f])

        ffmpeg_cmd.extend([
            "-filter_complex", f"amix=inputs={len(temp_files)}:duration=longest",
            "-ar", str(target_rate), "-ac", "1",
            "-c:a", "pcm_s16le", "-f", "wav", "pipe:1"
        ])

        result = subprocess.run(ffmpeg_cmd, capture_output=True, check=True)
        wav_bytes = result.stdout
        logger.info(f"🔀 Mixagem em {time.perf_counter()-t0:.3f}s")
        return wav_bytes

    except Exception as e:
        logger.error(f"❌ Mixagem erro: {e}")
        raise
    finally:
        for f in temp_files:
            try:
                os.unlink(f)
            except:
                pass

# ============================================================================
# POOLS
# ============================================================================
tts_pool = ProcessPoolExecutor(max_workers=TTS_WORKERS, initializer=_init_tts_worker)
mix_pool = ProcessPoolExecutor(max_workers=MIX_WORKERS, initializer=_init_mix_worker)

# ============================================================================
# MODELOS PYDANTIC
# ============================================================================
class AmbientConfig(BaseModel):
    enabled: bool = False
    file: Optional[str] = None
    volume_db: float = Field(default=-15.0, ge=-60.0, le=12.0)

class SpeakerMapping(BaseModel):
    role: str
    voice: str
    speed: float = Field(default=1.0, ge=0.3, le=3.0)
    guidance_scale: Optional[float] = Field(default=None, ge=0.5, le=5.0)

class TTSRequest(BaseModel):
    voice: Optional[str] = None
    text: str = Field(..., min_length=1)
    speed: float = Field(default=1.0, ge=0.3, le=3.0)
    guidance_scale: float = Field(default=1.8, ge=0.5, le=5.0)  # <-- ajustado
    effects: Dict[str, str] = Field(default_factory=dict)
    ambient: AmbientConfig = Field(default_factory=AmbientConfig)
    speakers: List[SpeakerMapping] = Field(default_factory=list)

# ============================================================================
# FASTAPI
# ============================================================================
app = FastAPI(title="OmniVoice Otimizado")
stats = defaultdict(list)
stats_lock = asyncio.Lock()

@app.post("/synthesize")
async def synthesize(req: TTSRequest):
    t_total = time.perf_counter()
    logger.info(f"📨 '{req.text[:40]}...'")

    speakers_list = []
    if req.speakers:
        for spk in req.speakers:
            speakers_list.append({
                'role': spk.role,
                'voice': spk.voice,
                'speed': spk.speed,
                'guidance_scale': spk.guidance_scale if spk.guidance_scale is not None else req.guidance_scale,
            })

    ambient_dict = req.ambient.model_dump() if hasattr(req.ambient, 'model_dump') else req.ambient.dict()
    loop = asyncio.get_running_loop()

    try:
        t_tts = time.perf_counter()
        segments = await loop.run_in_executor(
            tts_pool, process_fragments,
            req.voice, req.text, req.speed, req.guidance_scale,
            req.effects, speakers_list
        )
        t_tts = time.perf_counter() - t_tts

        t_mix = time.perf_counter()
        wav = await loop.run_in_executor(
            mix_pool, mix_and_export_task,
            segments, ambient_dict, TARGET_SR
        )
        t_mix = time.perf_counter() - t_mix
    except Exception as e:
        logger.error(f"❌ Erro: {e}")
        raise HTTPException(500, str(e))

    total = time.perf_counter() - t_total
    async with stats_lock:
        stats['total'].append(total)
        stats['tts'].append(t_tts)
        stats['mix'].append(t_mix)

    logger.info(f"⏱️ Total {total:.3f}s (TTS={t_tts:.3f}s Mix={t_mix:.3f}s)")
    return Response(content=wav, media_type="audio/wav")

# ============================================================================
# ENDPOINTS
# ============================================================================
@app.get("/stats")
async def get_stats():
    async with stats_lock:
        if not stats['total']:
            return {"message": "Nenhuma requisição ainda"}
        return {
            k: {
                "mean": sum(v)/len(v),
                "min": min(v),
                "max": max(v),
                "p95": sorted(v)[int(0.95*len(v))],
                "count": len(v)
            }
            for k, v in stats.items()
        }

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "voices": list(VOICE_REF_PATHS.keys()),
        "workers": {"tts": TTS_WORKERS, "mix": MIX_WORKERS},
        "gpu": torch.cuda.is_available()
    }

@app.get("/ready")
async def ready():
    return Response(status_code=200 if VOICE_REF_PATHS else 503)

@app.get("/live")
async def live():
    return Response(status_code=200)

# ============================================================================
# EXECUÇÃO
# ============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

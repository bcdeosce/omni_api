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
from concurrent.futures import ProcessPoolExecutor, TimeoutError
import asyncio
from collections import defaultdict
import traceback
import struct
import gc

mp.set_start_method('spawn', force=True)

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(processName)s | %(name)s | %(message)s",
)
logger = logging.getLogger("omnivoice-api")

BASE_DIR = Path("/app")
VOICES_DIR = BASE_DIR / "voices"
AMBIENT_DIR = BASE_DIR / "ambient"
EFFECTS_DIR = BASE_DIR / "effects"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

VOICES_DIR.mkdir(exist_ok=True)
AMBIENT_DIR.mkdir(exist_ok=True)
EFFECTS_DIR.mkdir(exist_ok=True)

_cpu_counter = mp.Value('i', 0)
_cpu_lock = mp.Lock()

TTS_WORKERS = 1
MIX_WORKERS = int(os.getenv("MIX_WORKERS", 4))
logger.info(f"Workers: TTS={TTS_WORKERS}, Mix={MIX_WORKERS}")

TARGET_SR = 24000
NUM_STEP = int(os.getenv("OMNIVOICE_NUM_STEP", "8"))
logger.info(f"Num steps: {NUM_STEP}")

PRECISION = os.getenv("OMNIVOICE_PRECISION", "bf16")

if PRECISION == "bf16":
    MODEL_PATH = "k2-fsa/OmniVoice"
else:
    MODEL_PATH = MODELS_DIR / f"OmniVoice_{PRECISION.upper()}"

DEFAULT_EFFECTS = {
    "[tosse]": "tosse.wav",
    "[suspiro]": "suspiro.wav",
    "[inspiracao]": "inspiracao.wav",
}

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

def preprocess_audio(file_path: Path, target_sr: int = TARGET_SR) -> np.ndarray:
    audio, sr = librosa.load(str(file_path), sr=None, mono=True)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio.astype(np.float32)

def process_audio_tensor(audio_tensor):
    if torch.is_tensor(audio_tensor):
        audio = audio_tensor.cpu().numpy()
    else:
        audio = audio_tensor
    if audio.ndim > 1:
        audio = audio.squeeze()
    audio = np.clip(audio, -1.0, 1.0)
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    return audio

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

    logger.info(f"📥 Carregando OmniVoice ({PRECISION}) de {MODEL_PATH}...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    if PRECISION == "bf16":
        model = OmniVoice.from_pretrained(str(MODEL_PATH), device_map=device, dtype=dtype)
    else:
        base_model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map=device, dtype=torch.bfloat16)
        state_file = MODEL_PATH / "quantized_state.pt"
        if state_file.exists():
            state_dict = torch.load(state_file, map_location="cpu", weights_only=True)
            base_model.load_state_dict(state_dict, strict=False)
            logger.info(f"✅ Estado quantizado carregado de {state_file}")
        else:
            logger.warning(f"⚠️ Arquivo quantizado não encontrado em {state_file}. Usando BF16 puro.")
        model = base_model

    logger.info(f"✅ Modelo carregado em {device}")

    mod = sys.modules['__main__']
    mod._omnivoice_model = model
    mod._sample_rate = TARGET_SR

    logger.info("📥 Pré-carregando vozes e ref_text...")
    voice_cache = {}
    for name, path in VOICE_REF_PATHS.items():
        try:
            audio = preprocess_audio(path)
            config_path = path.parent / f"{name}.config.json"
            ref_text = None
            if config_path.exists():
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                        ref_text = config.get('ref_text')
                except Exception:
                    pass
            if not ref_text:
                txt_path = path.parent / f"{name}.txt"
                if txt_path.exists():
                    try:
                        with open(txt_path, 'r', encoding='utf-8') as f:
                            ref_text = f.read().strip()
                    except Exception:
                        pass
            if not ref_text:
                ref_text = " "
            voice_cache[name] = {
                'audio': audio,
                'sr': TARGET_SR,
                'ref_text': ref_text
            }
            logger.info(f"   ✅ {name} ({len(audio)/TARGET_SR:.1f}s)")
        except Exception as e:
            logger.error(f"   ❌ {name}: {e}")
    mod._voice_cache = voice_cache

    logger.info("📥 Pré-carregando efeitos...")
    effect_cache = {}
    for name, path in VOICE_REF_PATHS.items():
        for wav in path.parent.glob("*.wav"):
            if wav == path:
                continue
            try:
                audio = preprocess_audio(wav)
                effect_cache[(name, wav.name)] = (audio, TARGET_SR)
            except Exception:
                pass
    for wav in EFFECTS_DIR.glob("*.wav"):
        try:
            audio = preprocess_audio(wav)
            effect_cache[('global', wav.name)] = (audio, TARGET_SR)
        except Exception:
            pass
    mod._effect_cache = effect_cache
    logger.info(f"✅ {len(voice_cache)} vozes, {len(effect_cache)} efeitos carregados")

    # Warm-up
    logger.info("🔥 Iniciando warm-up...")
    try:
        voices_list = list(voice_cache.keys())
        if len(voices_list) >= 2:
            voice1, voice2 = voices_list[0], voices_list[1]
        else:
            voice1 = voices_list[0]
            voice2 = voices_list[0]

        warmup_text = f"[voz1] Teste de aquecimento. [voz2] Preparando o modelo."
        warmup_speakers = [
            {"role": "voz1", "voice": voice1, "speed": 1.0, "guidance_scale": 1.8},
            {"role": "voz2", "voice": voice2, "speed": 1.0, "guidance_scale": 1.8},
        ]
        _ = process_fragments(
            voice_name=None,
            text=warmup_text,
            speed=1.0,
            guidance_scale=1.8,
            effects={},
            speakers=warmup_speakers,
            warmup=True
        )
        logger.info("✅ Warm-up concluído.")
    except Exception as e:
        logger.error(f"❌ Warm-up falhou: {e}")
        raise

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

def get_voice_data(voice_name: str):
    mod = sys.modules['__main__']
    cache = mod._voice_cache
    if voice_name not in cache:
        raise ValueError(f"Voz '{voice_name}' não encontrada")
    return cache[voice_name]

def get_effect_audio(effect_name: str, voice_name: str):
    mod = sys.modules['__main__']
    cache = mod._effect_cache
    key = (voice_name, effect_name)
    if key in cache:
        return cache[key]
    key_global = ('global', effect_name)
    if key_global in cache:
        return cache[key_global]
    return None

def synthesize_omnivoice_batch(
    texts: List[str],
    voice_name: str,
    speed: float,
    guidance_scale: float,
) -> Tuple[List[np.ndarray], float]:
    t0 = time.perf_counter()
    if not texts:
        return [], 0.0
    
    mod = sys.modules['__main__']
    model = mod._omnivoice_model
    sample_rate = mod._sample_rate
    voice_data = get_voice_data(voice_name)

    ref_audio = voice_data['audio']
    ref_text = voice_data['ref_text']

    durations = [max(0.5, len(text) * 0.1 / speed) for text in texts]
    durations = [min(d, 30.0) for d in durations]

    gen_config = OmniVoiceGenerationConfig(
        num_step=NUM_STEP,
        guidance_scale=guidance_scale,
    )

    with torch.no_grad():
        result = model.generate(
            text=texts,
            ref_audio=(ref_audio, sample_rate),
            ref_text=ref_text,
            generation_config=gen_config,
            duration=durations,
        )

    audios = []
    for audio in result:
        audios.append(process_audio_tensor(audio))
    
    elapsed = time.perf_counter() - t0
    return audios, elapsed

def process_fragments(voice_name, text, speed, guidance_scale, effects, speakers, warmup=False):
    t_start = time.perf_counter()
    is_dialog = bool(speakers)

    all_effects = DEFAULT_EFFECTS.copy()
    all_effects.update(effects or {})

    # ---------- FASE 1: PARSING ----------
    t_parse_start = time.perf_counter()
    
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

    tag_pattern = re.compile(r'(\[[^\]]*\])')
    parts = tag_pattern.split(text)
    parts = [p for p in parts if p]

    ordered_segments = []
    voice_texts = defaultdict(list)
    
    current_text = ""
    current_role = None

    for part in parts:
        if part.startswith('[') and part.endswith(']'):
            tag_name = part[1:-1]
            if is_dialog and tag_name in speaker_map:
                if current_text.strip():
                    idx = len(ordered_segments)
                    ordered_segments.append({'type': 'text', 'voice': current_role, 'text': current_text.strip()})
                    voice_texts[current_role].append((idx, current_text.strip()))
                    current_text = ""
                current_role = tag_name
                continue

            effect_file = all_effects.get(part) or all_effects.get(tag_name)
            if effect_file:
                if current_text.strip():
                    idx = len(ordered_segments)
                    ordered_segments.append({'type': 'text', 'voice': current_role, 'text': current_text.strip()})
                    voice_texts[current_role].append((idx, current_text.strip()))
                    current_text = ""
                current_voice = speaker_map[current_role][0] if current_role else voice_name
                effect_data = get_effect_audio(effect_file, current_voice)
                if effect_data is not None:
                    audio, sr = effect_data
                    pcm = (audio * 32767).astype(np.int16).tobytes()
                    ordered_segments.append({
                        'type': 'effect',
                        'pcm_bytes': pcm,
                        'sample_rate': sr,
                        'num_samples': len(audio)
                    })
                    if not warmup:
                        logger.info(f"🎬 Efeito '{effect_file}' adicionado (voz: {current_voice})")
                else:
                    current_text += part
                continue

            current_text += part
        else:
            current_text += part

    if current_text.strip():
        idx = len(ordered_segments)
        ordered_segments.append({'type': 'text', 'voice': current_role, 'text': current_text.strip()})
        voice_texts[current_role].append((idx, current_text.strip()))

    t_parse = time.perf_counter() - t_parse_start

    # ---------- FASE 2: BATCH POR VOZ ----------
    t_batch_start = time.perf_counter()
    voice_audios = {}
    batch_times = {}
    
    for role, items in voice_texts.items():
        texts = [item[1] for item in items]
        if role in speaker_map:
            v_name, spd, gs = speaker_map[role]
        else:
            v_name = voice_name
            spd = speed
            gs = guidance_scale
        try:
            audios, elapsed = synthesize_omnivoice_batch(texts, v_name, spd, gs)
            voice_audios[role] = audios
            batch_times[v_name] = elapsed
            if not warmup:
                logger.info(f"🎤 Voz '{v_name}': {len(texts)} frags em {elapsed:.3f}s")
        except Exception as e:
            logger.error(f"❌ Erro no batch para voz '{role}': {e}")
            raise
    t_batch = time.perf_counter() - t_batch_start

    # ---------- FASE 3: RECONSTRUÇÃO DA ORDEM ----------
    t_reconstruct_start = time.perf_counter()
    text_index_per_voice = defaultdict(int)
    final_segments = []

    for seg in ordered_segments:
        if seg['type'] == 'text':
            role = seg['voice']
            idx = text_index_per_voice[role]
            audio = voice_audios[role][idx]
            pcm = (audio * 32767).astype(np.int16).tobytes()
            final_segments.append({
                'pcm_bytes': pcm,
                'sample_rate': TARGET_SR,
                'num_samples': len(audio)
            })
            text_index_per_voice[role] += 1
        elif seg['type'] == 'effect':
            final_segments.append({
                'pcm_bytes': seg['pcm_bytes'],
                'sample_rate': seg['sample_rate'],
                'num_samples': seg['num_samples']
            })

    t_reconstruct = time.perf_counter() - t_reconstruct_start

    total_elapsed = time.perf_counter() - t_start
    if not warmup:
        logger.info(f"🔊 TTS: {total_elapsed:.3f}s | parse={t_parse:.3f}s batch={t_batch:.3f}s reconstr={t_reconstruct:.3f}s | {len(final_segments)} segs")
    
    return {
        'segments': final_segments,
        'timing': {
            'total_tts': total_elapsed,
            'parse': t_parse,
            'batch': t_batch,
            'reconstruct': t_reconstruct,
            'batch_per_voice': batch_times
        }
    }

def concat_and_export_task(segments_data, ambient_cfg, target_rate=24000):
    t0 = time.perf_counter()
    temp_files = []

    try:
        # Escreve cada segmento em um arquivo temporário
        for data in segments_data:
            pcm = data['pcm_bytes']
            sr = data['sample_rate']
            ns = data.get('num_samples', len(pcm)//2)
            header = write_wav_header(sr, ns, 1, 16)
            wav_data = header + pcm
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                f.write(wav_data)
                temp_files.append(f.name)

        if not temp_files:
            raise ValueError("Nenhum áudio para concatenar")

        # Concatenação da fala + efeitos
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            concat_file = f.name

        concat_cmd = ["ffmpeg", "-y"]
        for f in temp_files:
            concat_cmd.extend(["-i", f])

        n = len(temp_files)
        inputs = "".join(f"[{i}:a]" for i in range(n))
        filter_complex = f"{inputs}concat=n={n}:v=0:a=1[a]"

        concat_cmd.extend([
            "-filter_complex", filter_complex,
            "-map", "[a]",
            "-ar", str(target_rate),
            "-ac", "1",
            "-c:a", "pcm_s16le",
            "-f", "wav",
            concat_file
        ])
        subprocess.run(concat_cmd, capture_output=True, check=True)

        # Verifica se ambiente está habilitado
        wav_bytes = None
        if ambient_cfg.get('enabled') and ambient_cfg.get('file'):
            amb_path = AMBIENT_DIR / f"{ambient_cfg['file']}.wav"
            if amb_path.exists():
                # Obtém duração da fala para loop do ambiente
                probe_cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", concat_file]
                result_probe = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
                speech_duration = float(result_probe.stdout.strip())

                # Carrega e prepara o ambiente
                audio = preprocess_audio(amb_path, target_rate)
                pcm = (audio * 32767).astype(np.int16).tobytes()
                header = write_wav_header(target_rate, len(audio), 1, 16)
                amb_wav_data = header + pcm
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                    f.write(amb_wav_data)
                    amb_file = f.name
                    temp_files.append(amb_file)

                vol_db = ambient_cfg.get('volume_db', -15.0)

                # Loop infinito do ambiente, ajuste de volume, corte para a duração da fala, e mixagem
                cmd_amb = [
                    "ffmpeg", "-y",
                    "-stream_loop", "-1", "-i", amb_file,
                    "-i", concat_file,
                    "-filter_complex",
                    f"[0:a]volume={vol_db}dB,atrim=0:{speech_duration}[amb_loop];[1:a][amb_loop]amix=inputs=2:duration=longest",
                    "-ar", str(target_rate),
                    "-ac", "1",
                    "-c:a", "pcm_s16le",
                    "-f", "wav",
                    "pipe:1"
                ]
                result = subprocess.run(cmd_amb, capture_output=True, check=True)
                wav_bytes = result.stdout
            else:
                with open(concat_file, 'rb') as f:
                    wav_bytes = f.read()
        else:
            with open(concat_file, 'rb') as f:
                wav_bytes = f.read()

        # Limpeza
        for f in temp_files:
            try:
                os.unlink(f)
            except:
                pass
        try:
            os.unlink(concat_file)
        except:
            pass

        elapsed = time.perf_counter() - t0
        logger.info(f"🔀 Concatenação em {elapsed:.3f}s")
        return wav_bytes

    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg erro: {e.stderr.decode()[:200]}")
        raise RuntimeError(f"Falha na concatenação: {e.stderr.decode()[:200]}")
    except Exception as e:
        logger.error(f"❌ Concatenação erro: {e}")
        raise
    finally:
        for f in temp_files:
            try:
                os.unlink(f)
            except:
                pass

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
    guidance_scale: float = Field(default=1.8, ge=0.5, le=5.0)
    effects: Dict[str, str] = Field(default_factory=dict)
    ambient: AmbientConfig = Field(default_factory=AmbientConfig)
    speakers: List[SpeakerMapping] = Field(default_factory=list)

tts_pool = ProcessPoolExecutor(max_workers=TTS_WORKERS, initializer=_init_tts_worker)
mix_pool = ProcessPoolExecutor(max_workers=MIX_WORKERS, initializer=_init_mix_worker)

app = FastAPI(title="OmniVoice Batch por Voz")
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
        t_tts_start = time.perf_counter()
        result = await loop.run_in_executor(
            tts_pool, process_fragments,
            req.voice, req.text, req.speed, req.guidance_scale,
            req.effects, speakers_list,
            False
        )
        segments = result['segments']
        tts_timing = result['timing']
        t_tts = time.perf_counter() - t_tts_start

        t_mix_start = time.perf_counter()
        wav = await loop.run_in_executor(
            mix_pool, concat_and_export_task,
            segments, ambient_dict, TARGET_SR
        )
        t_mix = time.perf_counter() - t_mix_start
    except Exception as e:
        logger.error(f"❌ Erro: {e}")
        raise HTTPException(500, str(e))

    total = time.perf_counter() - t_total
    
    timing_data = {
        'total': total,
        'tts': t_tts,
        'mix': t_mix,
        'parse': tts_timing.get('parse', 0),
        'batch': tts_timing.get('batch', 0),
        'reconstruct': tts_timing.get('reconstruct', 0),
        'total_tts': tts_timing.get('total_tts', t_tts),
    }
    batch_per_voice = tts_timing.get('batch_per_voice', {})
    for voice, bt in batch_per_voice.items():
        timing_data[f'batch_{voice}'] = bt

    async with stats_lock:
        for key, value in timing_data.items():
            stats[key].append(value)

    logger.info(f"⏱️ Total {total:.3f}s (TTS={t_tts:.3f}s Mix={t_mix:.3f}s parse={timing_data['parse']:.3f}s batch={timing_data['batch']:.3f}s)")
    return Response(content=wav, media_type="audio/wav")

@app.get("/stats")
async def get_stats():
    async with stats_lock:
        if not stats['total']:
            return {"message": "Nenhuma requisição ainda"}
        report = {}
        for key, values in stats.items():
            sorted_vals = sorted(values)
            report[key] = {
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "p95": sorted_vals[int(0.95 * len(sorted_vals))] if len(sorted_vals) > 1 else sorted_vals[0],
                "count": len(values)
            }
        return report

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "voices": list(VOICE_REF_PATHS.keys()),
        "workers": {"tts": TTS_WORKERS, "mix": MIX_WORKERS},
        "gpu": torch.cuda.is_available(),
        "precision": PRECISION,
        "num_step": NUM_STEP
    }

@app.get("/ready")
async def ready():
    return Response(status_code=200 if VOICE_REF_PATHS else 503)

@app.get("/live")
async def live():
    return Response(status_code=200)

if __name__ == "__main__":
    import uvicorn

    logger.info("🚀 Inicializando worker TTS com aquecimento...")

    def do_warmup():
        try:
            voice = "adolescente_masculino"
            if voice not in VOICE_REF_PATHS:
                voice = next(iter(VOICE_REF_PATHS.keys()))
            warmup_text = "Olá, aquecimento do servidor. [tosse] Teste de efeito."
            warmup_speakers = [
                {"role": "speaker1", "voice": voice, "speed": 1.0, "guidance_scale": 1.8}
            ]
            logger.info(f"🔥 Enviando tarefa de aquecimento...")
            future = tts_pool.submit(
                process_fragments,
                None,
                warmup_text,
                1.0,
                1.8,
                {"[tosse]": "tosse.wav"},
                warmup_speakers,
                True
            )
            result = future.result(timeout=120)
            logger.info(f"✅ Aquecimento concluído! {len(result['segments'])} segmentos.")
            return True
        except TimeoutError:
            logger.error("❌ Tempo esgotado no aquecimento (120s).")
            return False
        except Exception as e:
            logger.error(f"❌ Falha no aquecimento: {e}")
            traceback.print_exc()
            return False

    success = do_warmup()
    if not success:
        logger.warning("⚠️ Aquecimento falhou, mas o servidor será iniciado mesmo assim.")

    logger.info("🚀 Iniciando servidor FastAPI...")
    uvicorn.run(app, host="0.0.0.0", port=8000)

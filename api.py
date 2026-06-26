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

# ---------- Instalação automática do OmniVoice e dependências ----------
try:
    import torch
    import soundfile as sf
    import librosa
    from omnivoice import OmniVoice, OmniVoiceGenerationConfig
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "omnivoice", "soundfile", "librosa", "torch"])
    import torch
    import soundfile as sf
    import librosa
    from omnivoice import OmniVoice, OmniVoiceGenerationConfig

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

# ---------- Configuração de logs ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(processName)s | %(name)s | %(message)s",
)
logger = logging.getLogger("omnivoice-api")

# ---------- Diretórios ----------
BASE_DIR = Path("/app")
VOICES_DIR = BASE_DIR / "voices"
AMBIENT_DIR = BASE_DIR / "ambient"
EFFECTS_DIR = BASE_DIR / "effects"

VOICES_DIR.mkdir(exist_ok=True)
AMBIENT_DIR.mkdir(exist_ok=True)
EFFECTS_DIR.mkdir(exist_ok=True)

# ---------- Contador global para afinidade de núcleos ----------
_cpu_counter = mp.Value('i', 0)
_cpu_lock = mp.Lock()

# ---------- Workers (configuração para GPU) ----------
TTS_WORKERS = int(os.getenv("TTS_WORKERS", 1))
MIX_WORKERS = int(os.getenv("MIX_WORKERS", 5))
logger.info(f"Workers: TTS={TTS_WORKERS} processo(s), Mix={MIX_WORKERS} processos")

# ---------- Inicializador dos workers TTS (carrega o modelo OmniVoice na GPU) ----------
def _init_tts_worker():
    # Configura ambiente para GPU
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["ORT_NUM_THREADS"] = "1"
    ort.set_default_logger_severity(3)

    # Afinidade de CPU (opcional, mas mantido)
    with _cpu_lock:
        cpu_id = _cpu_counter.value
        _cpu_counter.value += 1
    total_cpus = os.cpu_count()
    if cpu_id >= total_cpus:
        cpu_id = cpu_id % total_cpus
    try:
        os.sched_setaffinity(0, {cpu_id})
        logger.info(f"TTS Worker fixado ao núcleo {cpu_id}")
    except Exception as e:
        logger.warning(f"Falha ao definir afinidade no TTS: {e}")

    # Carrega o modelo OmniVoice
    logger.info("📥 Carregando modelo OmniVoice...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map=device, dtype=dtype)
    logger.info(f"✅ Modelo OmniVoice carregado em {device} com dtype {dtype}")

    # Armazena o modelo e configurações no módulo global do processo
    mod = sys.modules['__main__']
    mod._omnivoice_model = model
    mod._device = device
    mod._dtype = dtype
    mod._sample_rate = 24000  # taxa do OmniVoice

    # PRÉ-CARREGA TODAS AS VOZES NA RAM
    logger.info("📥 Pré-carregando todas as vozes de referência na RAM...")
    cache = {}
    for voice_name, ref_path in VOICE_REF_PATHS.items():
        try:
            audio, sr = librosa.load(str(ref_path), sr=None, mono=True)
            audio = audio.astype(np.float32)
            cache[voice_name] = (audio, sr)
            logger.info(f"   ✅ '{voice_name}' ({len(audio)/sr:.1f}s)")
        except Exception as e:
            logger.error(f"   ❌ Falha ao carregar '{voice_name}': {e}")
    mod._ref_audio_cache = cache
    logger.info(f"✅ {len(cache)} vozes pré-carregadas na RAM")

# ---------- Inicializador dos workers de mixagem ----------
def _init_mix_worker():
    os.environ["OMP_NUM_THREADS"] = "1"
    with _cpu_lock:
        cpu_id = _cpu_counter.value
        _cpu_counter.value += 1
    total_cpus = os.cpu_count()
    if cpu_id >= total_cpus:
        cpu_id = cpu_id % total_cpus
    try:
        os.sched_setaffinity(0, {cpu_id})
        logger.info(f"Mix Worker fixado ao núcleo {cpu_id}")
    except Exception as e:
        logger.warning(f"Falha ao definir afinidade na mixagem: {e}")

# ---------- Carregamento de vozes (pastas com arquivos .wav) ----------
VOICE_REF_PATHS: Dict[str, Path] = {}  # nome_da_voz -> caminho para o .wav

def load_all_voices():
    for item in VOICES_DIR.iterdir():
        if item.is_dir():
            voice_name = item.name
            wav_files = list(item.glob("*.wav"))
            if not wav_files:
                continue
            # Usa o primeiro .wav encontrado (espera-se um único)
            ref_path = wav_files[0]
            VOICE_REF_PATHS[voice_name] = ref_path
            logger.info(f"Voz registrada: {voice_name} -> {ref_path}")
    # Também aceita arquivos .wav diretamente na raiz (backward compatibility)
    for wav_file in VOICES_DIR.glob("*.wav"):
        voice_name = wav_file.stem
        if voice_name not in VOICE_REF_PATHS:
            VOICE_REF_PATHS[voice_name] = wav_file
            logger.info(f"Voz personalizada (raiz) registrada: {voice_name}")

load_all_voices()
logger.info(f"Total de vozes disponíveis: {len(VOICE_REF_PATHS)}")

# ---------- Funções auxiliares para o worker TTS ----------
def get_ref_audio(voice_name: str) -> Tuple[np.ndarray, int]:
    """Retorna (áudio, sample_rate) da voz referência, já em cache."""
    mod = sys.modules['__main__']
    cache = getattr(mod, '_ref_audio_cache', {})
    if voice_name not in cache:
        raise ValueError(f"Voz '{voice_name}' não encontrada no cache")
    return cache[voice_name]

def synthesize_omnivoice(
    text: str,
    voice_name: str,
    duration_factor: float,
    guidance_scale: float,
) -> np.ndarray:
    """
    Gera áudio usando OmniVoice a partir de um texto e uma voz de referência.
    Retorna áudio em float32 com sample_rate = 24000.
    """
    mod = sys.modules['__main__']
    model = mod._omnivoice_model
    device = mod._device
    sample_rate = mod._sample_rate

    # Obtém áudio de referência (já em cache)
    ref_audio, ref_sr = get_ref_audio(voice_name)
    # Se a taxa da referência for diferente, resample
    if ref_sr != sample_rate:
        ref_audio = librosa.resample(ref_audio, orig_sr=ref_sr, target_sr=sample_rate)

    # Calcula duração estimada (em segundos) a partir do fator de duração
    duration = max(0.5, len(text) * duration_factor)  # mínimo 0.5s
    duration = min(duration, 30.0)  # limite máximo

    # Cria configuração de geração
    gen_config = OmniVoiceGenerationConfig(
        num_step=60,  # pode ser ajustado
        guidance_scale=guidance_scale,
    )

    # Gera áudio
    with torch.no_grad():
        audio_list = model.generate(
            text=text,
            ref_audio=ref_audio,
            generation_config=gen_config,
            duration=duration,
        )
    if not audio_list or len(audio_list) == 0:
        raise RuntimeError("Falha ao gerar áudio com OmniVoice")
    audio = audio_list[0]  # primeiro (e único) canal

    # Garante que está em float32 e no formato correto
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    return audio, sample_rate

# ---------- Mixagem com FFmpeg (executada nos workers de mixagem) ----------
def mix_and_export_task(segments_data, ambient_cfg, target_rate=22050):
    """
    Recebe uma lista de segmentos (cada um com 'pcm_bytes' ou 'effect' com caminho) e mixa.
    Retorna bytes do WAV final.
    """
    t0 = time.perf_counter()
    temp_files = []
    ffmpeg_cmd = ["ffmpeg", "-y"]

    try:
        # 1. Criar arquivos temporários para cada segmento
        for data in segments_data:
            if 'pcm_bytes' in data:
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                    f.write(data['pcm_bytes'])
                    temp_files.append(f.name)
            elif 'effect' in data:
                # Efeito: procurar no diretório da voz ou global
                voice_dir = VOICES_DIR / data['voice']
                effect_path = voice_dir / data['effect']
                if not effect_path.exists():
                    effect_path = EFFECTS_DIR / data['effect']
                if not effect_path.exists():
                    raise FileNotFoundError(f"Efeito '{data['effect']}' não encontrado")
                # Copiar para temp para padronizar
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                    with open(effect_path, 'rb') as src:
                        f.write(src.read())
                    temp_files.append(f.name)
            else:
                continue

        # 2. Adicionar ambiente, se habilitado
        if ambient_cfg.get('enabled') and ambient_cfg.get('file'):
            ambient_path = AMBIENT_DIR / f"{ambient_cfg['file']}.wav"
            if ambient_path.exists():
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                    with open(ambient_path, 'rb') as src:
                        f.write(src.read())
                    temp_files.append(f.name)

        if not temp_files:
            raise ValueError("Nenhum arquivo para mixar")

        # 3. Montar comando FFmpeg
        for f in temp_files:
            ffmpeg_cmd.extend(["-i", f])

        # Filtro amix: todos os inputs, duração = maior
        filter_complex = f"amix=inputs={len(temp_files)}:duration=longest"
        ffmpeg_cmd.extend([
            "-filter_complex", filter_complex,
            "-ar", str(target_rate),
            "-ac", "1",
            "-c:a", "pcm_s16le",
            "-f", "wav",
            "pipe:1"
        ])

        # 4. Executar
        result = subprocess.run(ffmpeg_cmd, capture_output=True, check=True)
        wav_bytes = result.stdout

        t_total = time.perf_counter() - t0
        logger.debug(f"Mixagem FFmpeg concluída em {t_total:.3f}s | {len(temp_files)} arquivos")
        return wav_bytes

    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg erro: {e.stderr.decode()}")
        raise RuntimeError("Falha na mixagem com FFmpeg")
    finally:
        for f in temp_files:
            try:
                os.unlink(f)
            except:
                pass

# ---------- Função que processa uma requisição INTEIRA dentro de um worker TTS ----------
def process_entire_request(
    voice_name: Optional[str],
    text: str,
    speed: float,               # será usado como duration_factor
    guidance_scale: float,
    effects: Dict[str, str],
    speakers: List[Dict],
    ambient_cfg: Dict,
) -> bytes:
    """
    Executada em um worker TTS (GPU). Processa todos os fragmentos sequencialmente,
    monta lista de segmentos e envia para mixagem.
    Retorna bytes do WAV final.
    """
    t_start = time.perf_counter()

    # 1. Construir speaker_map
    is_dialog = bool(speakers)
    if not is_dialog:
        if not voice_name:
            raise ValueError("voice_name é obrigatório no modo simples")
        speaker_map = {None: (voice_name, speed, guidance_scale)}
        current_role = None
    else:
        speaker_map = {}
        for spk in speakers:
            # spk é dict com role, voice, speed (duration_factor), guidance_scale
            gs = spk.get('guidance_scale', guidance_scale)
            spd = spk.get('speed', speed)
            speaker_map[spk['role']] = (spk['voice'], spd, gs)
        current_role = None

    # 2. Dividir texto
    parts = re.split(r'(\[.*?\])', text)
    parts = [p.strip() for p in parts if p.strip()]

    # 3. Processar cada parte sequencialmente
    segments = []  # lista de dicts: {'pcm_bytes': bytes, 'sample_rate': int} ou {'effect': str, 'voice': str}
    for part in parts:
        # Verificar se é tag de speaker (modo diálogo)
        if is_dialog and part.startswith('[') and part.endswith(']'):
            role = part[1:-1]
            if role in speaker_map:
                current_role = role
            continue

        # Verificar se é efeito (mapeado no dicionário effects)
        if part in effects:
            effect_file = effects[part]
            # Determinar voz para o efeito (se for diálogo, usa a voz atual)
            voice_for_eff = speaker_map[current_role][0] if is_dialog and current_role else voice_name
            segments.append({'effect': effect_file, 'voice': voice_for_eff})
            continue

        # Senão, é texto para síntese com OmniVoice
        if is_dialog:
            if current_role is None:
                raise ValueError("Nenhum speaker definido antes do texto. Use [papel] no início.")
            v_name, spd, gs = speaker_map[current_role]
        else:
            v_name = voice_name
            spd = speed
            gs = guidance_scale

        # Sintetizar com OmniVoice
        try:
            audio, sr = synthesize_omnivoice(part, v_name, spd, gs)
            # Converte para PCM 16-bit (int16) para compatibilidade com mixagem
            pcm_int16 = (audio * 32767).astype(np.int16).tobytes()
            segments.append({'pcm_bytes': pcm_int16, 'sample_rate': sr})
        except Exception as e:
            logger.error(f"Erro na síntese do fragmento '{part[:30]}...': {e}")
            raise

    # 4. Enviar para mixagem (chamada síncrona, pois já estamos em um worker)
    wav_bytes = mix_and_export_task(segments, ambient_cfg, target_rate=22050)

    t_total = time.perf_counter() - t_start
    logger.info(f"Worker TTS processou requisição em {t_total:.3f}s | {len(segments)} segmentos")
    return wav_bytes

# ---------- Pools de processos ----------
tts_pool = ProcessPoolExecutor(
    max_workers=TTS_WORKERS,
    initializer=_init_tts_worker
)
mix_pool = ProcessPoolExecutor(
    max_workers=MIX_WORKERS,
    initializer=_init_mix_worker
)

# ---------- Modelos Pydantic ----------
class AmbientConfig(BaseModel):
    enabled: bool = False
    file: Optional[str] = None
    volume_db: float = Field(default=-15.0, ge=-60.0, le=12.0)

class SpeakerMapping(BaseModel):
    role: str
    voice: str
    speed: float = Field(default=1.0, ge=0.3, le=3.0, description="Fator de duração (duration_factor)")
    guidance_scale: Optional[float] = Field(default=None, ge=0.5, le=5.0, description="Intensidade da emoção/estilo")

class TTSRequest(BaseModel):
    voice: Optional[str] = None
    text: str = Field(..., min_length=1)
    speed: float = Field(default=1.0, ge=0.3, le=3.0, description="Fator de duração (duration_factor)")
    guidance_scale: float = Field(default=2.0, ge=0.5, le=5.0, description="Guidance scale (emoção/estilo)")
    effects: Dict[str, str] = Field(default_factory=dict)
    ambient: AmbientConfig = Field(default_factory=AmbientConfig)
    speakers: List[SpeakerMapping] = Field(default_factory=list)

# ---------- FastAPI ----------
app = FastAPI(title="OmniVoice TTS API (GPU)")

# ---------- Estatísticas ----------
stats = defaultdict(list)
stats_lock = asyncio.Lock()

@app.post("/synthesize", response_class=Response)
async def synthesize(req: TTSRequest):
    t_total_start = time.perf_counter()

    # 1. Preparar argumentos para o worker TTS
    speakers_list = []
    if req.speakers:
        for spk in req.speakers:
            speakers_list.append({
                'role': spk.role,
                'voice': spk.voice,
                'speed': spk.speed,
                'guidance_scale': spk.guidance_scale if spk.guidance_scale is not None else req.guidance_scale,
            })

    # Serializar ambient_cfg
    try:
        ambient_dict = req.ambient.model_dump()
    except AttributeError:
        ambient_dict = req.ambient.dict()

    # 2. Enviar requisição inteira para um worker TTS (apenas um, se TTS_WORKERS=1)
    loop = asyncio.get_running_loop()
    try:
        wav_bytes = await loop.run_in_executor(
            tts_pool,
            process_entire_request,
            req.voice,
            req.text,
            req.speed,
            req.guidance_scale,
            req.effects,
            speakers_list,
            ambient_dict
        )
    except Exception as e:
        logger.error(f"Erro no processamento: {e}")
        raise HTTPException(500, f"Falha no processamento: {str(e)}")
    t_total = time.perf_counter() - t_total_start

    # 3. Acumular estatísticas
    async with stats_lock:
        stats['total'].append(t_total)

    logger.info(f"⏱️ Requisição concluída em {t_total:.3f}s")
    return Response(content=wav_bytes, media_type="audio/wav")

# ---------- Endpoint de estatísticas ----------
@app.get("/stats")
async def get_stats():
    async with stats_lock:
        if not stats['total']:
            return {"message": "Nenhuma requisição processada ainda."}
        report = {}
        for key, values in stats.items():
            report[key] = {
                "count": len(values),
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "p95": sorted(values)[int(0.95 * len(values))] if len(values) > 1 else values[0],
            }
        return report

# ---------- Endpoints de saúde ----------
@app.get("/started")
async def started():
    return Response(status_code=200, content="started")

@app.get("/ready")
async def ready():
    if VOICE_REF_PATHS:
        return Response(status_code=200, content="ready")
    return Response(status_code=503, content="loading models")

@app.get("/live")
async def live():
    return Response(status_code=200, content="alive")

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "voices": list(VOICE_REF_PATHS.keys()),
        "workers": {"tts": TTS_WORKERS, "mix": MIX_WORKERS},
        "gpu_available": torch.cuda.is_available()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

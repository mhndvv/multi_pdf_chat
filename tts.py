# tts.py
import io
import torch
import soundfile as sf
from transformers import AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration

_MODEL_ID = "parler-tts/parler-tts-tiny-v1"
_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

_model = None
_tokenizer = None

_DEFAULT_DESC = (
    "A friendly female voice speaking clear, neutral English at a moderate pace "
    "with natural intonation and studio clarity."
)


def _ensure_loaded():
    global _model, _tokenizer
    if _model is None or _tokenizer is None:
        _model = ParlerTTSForConditionalGeneration.from_pretrained(_MODEL_ID).to(_DEVICE)
        _tokenizer = AutoTokenizer.from_pretrained(_MODEL_ID)


def tts_to_wav_bytes(text: str, description: str | None = None) -> bytes:
    """
    Convert text to WAV bytes using Parler-TTS Tiny.
    """
    _ensure_loaded()
    desc = description or _DEFAULT_DESC
    if not text or not text.strip():
        text = "No answer available."

    with torch.inference_mode():
        desc_ids = _tokenizer(desc, return_tensors="pt").input_ids.to(_DEVICE)
        prompt_ids = _tokenizer(text, return_tensors="pt").input_ids.to(_DEVICE)
        audio_tensor = _model.generate(
            input_ids=desc_ids, prompt_input_ids=prompt_ids
        )

    audio = audio_tensor.cpu().numpy().squeeze()
    sr = _model.config.sampling_rate

    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    buf.seek(0)
    return buf.read()

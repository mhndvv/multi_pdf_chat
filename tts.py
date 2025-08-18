import io
import torch
import soundfile as sf
from transformers import AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration

# Use Hugging Face Parler-TTS tiny model 
_MODEL_ID = "parler-tts/parler-tts-tiny-v1"

_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

_model = None
_tokenizer = None

# Default description controls how the voice should sound
_DEFAULT_DESC = (
    "A friendly female voice speaking clear, neutral English at a moderate pace "
    "with natural intonation and studio clarity."
)

def _ensure_loaded():
    """
    Lazy-load the Text-to-Speech (TTS) model and tokenizer once.

    This function initializes the global TTS model and tokenizer if they 
    are not already loaded. The model is moved to the target device.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Raises
    ------
    OSError
        If the model or tokenizer cannot be loaded from the given path or model ID.
    """
    global _model, _tokenizer
    if _model is None or _tokenizer is None:
        _model = ParlerTTSForConditionalGeneration.from_pretrained(_MODEL_ID).to(_DEVICE)
        _tokenizer = AutoTokenizer.from_pretrained(_MODEL_ID)


def tts_to_wav_bytes(text: str, description: str | None = None) -> bytes:
    """
    Generate speech audio (WAV) from text using Parler-TTS.

    The function encodes the input text and an optional description 
    (voice style), synthesizes speech, and returns WAV audio bytes 
    suitable for streaming or download.

    Parameters
    ----------
    text : str
        The text to convert into speech. If empty, a default placeholder 
        ("No answer available.") is spoken.
    description : str, optional
        A description controlling voice style. If None, a default voice 
        description is used.

    Returns
    -------
    bytes
        The generated speech as a WAV audio file stored in memory.

    Raises
    ------
    RuntimeError
        If model inference fails during audio generation.
    ValueError
        If both `text` and the default placeholder are empty strings.
    OSError
        If writing the WAV audio to memory fails.
    """
    _ensure_loaded()

    # Use default description if none is provided
    desc = description or _DEFAULT_DESC
    if not text or not text.strip():
        text = "No answer available."

    with torch.inference_mode():
        # Encode description (voice style) and input text
        desc_ids = _tokenizer(desc, return_tensors="pt").input_ids.to(_DEVICE)
        prompt_ids = _tokenizer(text, return_tensors="pt").input_ids.to(_DEVICE)

        # Generate audio tensor from model
        audio_tensor = _model.generate(
            input_ids=desc_ids,
            prompt_input_ids=prompt_ids
        )

    # Convert tensor to numpy array and get sampling rate
    audio = audio_tensor.cpu().numpy().squeeze()
    sr = _model.config.sampling_rate

    # Save audio into memory buffer as WAV
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    buf.seek(0)
    return buf.read()
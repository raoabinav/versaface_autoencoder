# MaskGCT 8-D Audio Autoencoder

An audio autoencoder that extracts **8-dimensional continuous semantic latents** from audio. Built on [MaskGCT](https://github.com/open-mmlab/Amphion), this system enables:

- **8-D Latent Extraction**: Audio → 8-D continuous vectors (50 Hz)
- **Audio Reconstruction**: 8-D latents → Audio
- **Voice Conversion**: Source content + Reference voice → Output
- **Text-to-Speech**: Reference voice + Target text → Output

---

## Quick Start

```python
from utils.util import load_config
from models.tts.metis.audio_tokenizer import AudioTokenizer
from models.tts.metis.semantic_8d_wrappers import Metis8dEncoder, Metis8dDecoder
import soundfile as sf

# Initialize
device = "cuda:0"
cfg = load_config("models/tts/metis/config/base.json")
audio_tokenizer = AudioTokenizer(cfg, device)
encoder = Metis8dEncoder(audio_tokenizer)
decoder = Metis8dDecoder(cfg, audio_tokenizer)

# Encode audio to 8-D latents
_, z_8d = encoder.encode_from_path("audio.wav")
print(f"8-D latents: {z_8d.shape}")  # [1, T, 8]

# Decode back to audio
waveform = decoder.decode_from_z(z_8d)
sf.write("reconstructed.wav", waveform, 24000)
```

---

## Architecture

```
Audio (16kHz) → W2vBert2 → 1024-D features → RepCodec → 8-D latents
                                                           ↓
                                            [T, 8] continuous vectors @ 50Hz
                                                           ↓
                         8-D latents → Quantize → S2A → DAC → Audio (24kHz)
```

| Parameter | Value |
|-----------|-------|
| Input sample rate | 16 kHz (semantic), 24 kHz (acoustic) |
| Output sample rate | 24 kHz |
| Latent dimensions | 8 |
| Latent frame rate | 50 Hz (20ms per frame) |
| Semantic codebook | 8192 tokens |
| Acoustic quantizers | 12 layers |

---

## Installation

```bash
# Clone and setup
git clone <repository-url>
cd versaface_autoencoder
conda create -n versaface python=3.10
conda activate versaface
pip install -r requirements.txt
```

Model checkpoints (~5GB) are auto-downloaded from HuggingFace on first run.

---

## API Reference

### Metis8dEncoder

```python
encoder = Metis8dEncoder(audio_tokenizer)

# Extract 8-D latents
feat_1024d, z_8d = encoder.encode_from_path("audio.wav")
# feat_1024d: [1, T, 1024] - SSL features
# z_8d: [1, T, 8] - 8-D continuous latents

# Extract acoustic codes (for voice conversion)
acoustic_codes = encoder.encode_acoustic_from_path("reference.wav")
# acoustic_codes: [1, T, 12]

# Extract both semantic and acoustic (for voice conversion prompt)
semantic_codes, acoustic_codes = encoder.encode_prompt_from_path("reference.wav")
```

### Metis8dDecoder

```python
decoder = Metis8dDecoder(cfg, audio_tokenizer)

# Reconstruction (no voice conversion)
waveform = decoder.decode_from_z(z_8d)

# Voice conversion (source content + reference voice)
waveform = decoder.decode_from_z(
    z_8d,                          # Content from source
    prompt_acoustic_code=acoustic, # Voice from reference
    prompt_semantic_code=semantic  # Context from reference
)
```

---

## Voice Conversion

Transfer content from source audio to voice of reference audio.

```bash
python example_voice_conversion.py \
    --source source.wav \
    --reference reference.wav \
    --output output.wav
```

```python
# Extract content from source
_, z_8d = encoder.encode_from_path("source.wav")

# Extract voice from reference
semantic, acoustic = encoder.encode_prompt_from_path("reference.wav")

# Convert
output = decoder.decode_from_z(z_8d, acoustic, semantic)
sf.write("output.wav", output, 24000)
```

---

## Text-to-Speech

Generate speech in a reference voice.

```python
from models.tts.maskgct.maskgct_utils import g2p_
from models.tts.maskgct.maskgct_t2s import MaskGCT_T2S

# Load T2S model
t2s_model = MaskGCT_T2S(cfg=cfg.model.t2s_model)
# ... load checkpoint ...

# Extract reference features
prompt_semantic, _ = semantic_codec.quantize(feat)
prompt_acoustic = encoder.encode_acoustic_from_path("reference.wav")

# Convert text to phonemes
prompt_text = "What the reference speaker says"
target_text = "What you want to generate"
_, prompt_phones = g2p_(prompt_text, 'en')
_, target_phones = g2p_(target_text, 'en')

# Generate semantic codes for target text
predict_semantic = t2s_model.reverse_diffusion(
    prompt_semantic, target_len, phone_ids, n_timesteps=25
)

# Generate acoustic codes and decode
# ... S2A and DAC ...
```

See `notebook.ipynb` for complete TTS example.

---

## Batch Embedding Extraction

Extract 8-D embeddings from audio datasets.

```bash
python extract_audio_embeddings.py \
    --audio-dir /data/audios \
    --embedding-dir /data/embeddings \
    --json-dir /data/jsons
```

Output: `.npy` files with shape `[T, 8]` (float32).

---

## Project Structure

```
versaface_autoencoder/
├── models/
│   ├── tts/
│   │   ├── metis/
│   │   │   ├── audio_tokenizer.py       # AudioTokenizer
│   │   │   ├── semantic_8d_wrappers.py  # Metis8dEncoder, Metis8dDecoder
│   │   │   ├── config/base.json         # Model config
│   │   │   └── test_voice_conversion/   # Demo audio files
│   │   └── maskgct/
│   │       ├── maskgct_s2a.py           # S2A model
│   │       ├── maskgct_t2s.py           # T2S model (for TTS)
│   │       ├── maskgct_utils.py         # Builders
│   │       └── g2p/                     # Text-to-phoneme
│   └── codec/                           # Audio codecs
├── utils/util.py                        # Config loading
├── example_voice_conversion.py          # VC CLI
├── extract_audio_embeddings.py          # Batch extraction
├── notebook.ipynb                       # Interactive demo
└── requirements.txt
```

---

## Demo Files

Located in `models/tts/metis/test_voice_conversion/`:

| File | Description |
|------|-------------|
| `source.wav` | Source audio (female speaker) |
| `trump.wav` | Reference audio (Trump) |
| `tts_final.wav` | TTS output: Trump voice + custom text |
| `vc_maskgct_style.wav` | VC output: source content + Trump voice |
| `roundtrip_reconstruction.wav` | Autoencoder roundtrip |

Located in `models/tts/metis/test_8d_wrapper/`:

| File | Description |
|------|-------------|
| `roundtrip_source.wav` | Original source audio |
| `roundtrip_no_prompt.wav` | Reconstructed from 8-D latents |
| `interp_*.wav` | Latent interpolation (0% to 100%) |

---

## Troubleshooting

### Transformers Version Error
```bash
pip install transformers==4.40.0
```

### GPU Out of Memory
```bash
python extract_audio_embeddings.py --device cpu
```

---

## License

MIT License

## Acknowledgments

Built on [Amphion MaskGCT](https://github.com/open-mmlab/Amphion).

# VersaFace Autoencoder

A neural audio autoencoder that extracts **8-dimensional continuous semantic latents** from audio. Built on Metis TTS, this system enables voice conversion and batch embedding extraction for large-scale audio datasets.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [API Reference](#api-reference)
  - [Metis8dEncoder](#metis8dencoder)
  - [Metis8dDecoder](#metis8ddecoder)
- [Voice Conversion](#voice-conversion)
- [Batch Embedding Extraction](#batch-embedding-extraction)
- [Test Results](#test-results)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)

---

## Overview

### Architecture

```
Audio (16kHz) → w2v-bert-2.0 → 1024-D features → RepCodec → 8-D latents
                                                              ↓
                                               [T, 8] continuous vectors @ 50Hz
```

### Key Specifications

| Parameter | Value |
|-----------|-------|
| Input sample rate | 16 kHz (semantic), 24 kHz (acoustic) |
| Output sample rate | 24 kHz |
| Latent dimensions | 8 |
| Latent frame rate | 50 Hz (20ms per frame) |
| Semantic codebook size | 8192 |
| Acoustic quantizers | 12 |

---

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ RAM
- 5GB disk space for model checkpoints

### Setup

```bash
# Clone repository
git clone <repository-url>
cd versaface_autoencoder

# Create conda environment
conda create -n versaface python=3.10
conda activate versaface

# Install dependencies
pip install -r requirements.txt
```

Model checkpoints are automatically downloaded from HuggingFace on first run (~5GB).

---

## API Reference

### Metis8dEncoder

Extracts 8-D continuous semantic latents from audio.

#### Initialization

```python
from utils.util import load_config
from models.tts.metis.audio_tokenizer import AudioTokenizer
from models.tts.metis.semantic_8d_wrappers import Metis8dEncoder

device = "cuda:0"
cfg = load_config("models/tts/metis/config/base.json")
audio_tokenizer = AudioTokenizer(cfg, device)
encoder = Metis8dEncoder(audio_tokenizer)
```

#### Methods

##### `encode_from_path(wav_path: str) -> Tuple[Tensor, Tensor]`

Extract 8-D latents from an audio file.

```python
feat_1024d, z_8d = encoder.encode_from_path("audio.wav")
# feat_1024d: [1, T, 1024] - SSL features
# z_8d: [1, T, 8] - 8-D continuous latents
```

**Parameters:**
- `wav_path`: Path to audio file (any sample rate, auto-resampled to 16kHz)

**Returns:**
- `feat_1024d`: `Tensor[1, T, 1024]` - Normalized SSL features
- `z_8d`: `Tensor[1, T, 8]` - 8-D continuous latents

##### `encode_acoustic_from_path(wav_path: str) -> Tensor`

Extract acoustic codes for voice conversion style transfer.

```python
acoustic_codes = encoder.encode_acoustic_from_path("reference.wav")
# acoustic_codes: [1, T, 12] - 12-layer acoustic codes
```

**Parameters:**
- `wav_path`: Path to audio file (auto-resampled to 24kHz)

**Returns:**
- `acoustic_codes`: `Tensor[1, T, 12]` - Acoustic codes (long)

---

### Metis8dDecoder

Decodes 8-D latents back to audio waveform.

#### Initialization

```python
from models.tts.metis.semantic_8d_wrappers import Metis8dDecoder

decoder = Metis8dDecoder(cfg, audio_tokenizer)
```

#### Methods

##### `decode_from_z(z_8d: Tensor, prompt_acoustic_code: Tensor = None) -> np.ndarray`

Decode 8-D latents to waveform, optionally with voice conversion.

```python
# Without voice conversion (reconstruction)
waveform = decoder.decode_from_z(z_8d)

# With voice conversion (source content + reference style)
waveform = decoder.decode_from_z(z_8d, prompt_acoustic_code=acoustic_codes)
```

**Parameters:**
- `z_8d`: `Tensor[1, T, 8]` - 8-D continuous latents (content)
- `prompt_acoustic_code`: `Tensor[1, T_prompt, 12]` - Acoustic codes for style (optional)

**Returns:**
- `waveform`: `np.ndarray[T_samples]` - Audio waveform at 24kHz

---

## Voice Conversion

Voice conversion transfers the **content** from a source audio to the **style/voice** of a reference audio.

### Command Line

```bash
python example_voice_conversion.py \
    --source path/to/source.wav \
    --reference path/to/reference.wav \
    --output output.wav
```

### Python API

```python
import torch
import soundfile as sf
from utils.util import load_config
from models.tts.metis.audio_tokenizer import AudioTokenizer
from models.tts.metis.semantic_8d_wrappers import Metis8dEncoder, Metis8dDecoder

# Initialize
device = "cuda:0"
cfg = load_config("models/tts/metis/config/base.json")
audio_tokenizer = AudioTokenizer(cfg, device)
encoder = Metis8dEncoder(audio_tokenizer)
decoder = Metis8dDecoder(cfg, audio_tokenizer)

# Extract content from source
_, z_8d = encoder.encode_from_path("source.wav")

# Extract style from reference
prompt_acoustic = encoder.encode_acoustic_from_path("reference.wav")

# Voice conversion
output = decoder.decode_from_z(z_8d, prompt_acoustic_code=prompt_acoustic)

# Save
sf.write("output.wav", output, 24000)
```

### Notes

- Reference audio should be **1-2 seconds** for best results
- Output length matches source audio length
- Source provides "what to say", reference provides "how to say it"

---

## Batch Embedding Extraction

Extract 8-D embeddings from large audio datasets, saving as `.npy` files with matching folder structure.

### Output Format

```
Input:  /data/versaface/audios/part_003/00/05/00055b60d58d197ad1de21f37092fccd.wav
Output: /data/versaface/audio_embeddings/part_003/00/05/00055b60d58d197ad1de21f37092fccd.npy
```

Each `.npy` file contains a `float32` array of shape `[T, 8]` where `T` = number of frames at 50Hz.

### Usage

```bash
# Full extraction
python extract_audio_embeddings.py \
    --audio-dir /data/versaface/audios \
    --embedding-dir /data/versaface/audio_embeddings \
    --json-dir /data/versaface/jsons

# Process specific part
python extract_audio_embeddings.py --part part_003

# Test with limited files
python extract_audio_embeddings.py --max-files 100

# Resume interrupted extraction (auto-skips existing files)
python extract_audio_embeddings.py
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--audio-dir` | `/data/common/versaface/audios` | Input audio directory |
| `--embedding-dir` | `/data/common/versaface/audio_embeddings` | Output NPY directory |
| `--json-dir` | `/data/common/versaface/jsons` | Success JSON directory |
| `--part` | None | Process only specific part (e.g., `part_003`) |
| `--max-files` | None | Limit files for testing |
| `--batch-size` | 100 | Save progress every N files |
| `--device` | `cuda:0` | Device for inference |

### Output Files

1. **NPY Embeddings**: `{embedding-dir}/part_XXX/XX/XX/hash.npy`
   - Shape: `[T, 8]` where T ≈ duration_seconds × 50
   - Dtype: `float32`

2. **Success JSON**: `{json-dir}/success_total_audio.json`
   - List of successfully processed audio paths
   - Format matches video JSON structure

---

## Test Results

### Voice Conversion Test

```
$ python example_voice_conversion.py \
    --source models/tts/metis/test_voice_conversion/test_voice_conversion_source.wav \
    --reference models/tts/metis/test_voice_conversion/test_voice_conversion_reference.wav \
    --output models/tts/metis/test_voice_conversion/output_fixed.wav

[STEP 1] Encoding source audio to 8-D semantic latents...
✓ 8-D latents shape: torch.Size([1, 210, 8])

[STEP 2] Extracting acoustic codes from reference audio...
✓ Acoustic codes shape: torch.Size([1, 75, 12])
  Reference duration: ~1.50 seconds

[STEP 3] Converting voice...
✓ Generated waveform: 100800 samples
  Duration: 4.20 seconds

✅ VOICE CONVERSION COMPLETE
```

**Result**: Source audio (4.2s) + Reference style (1.5s) → Output (4.2s, full length preserved)

### Embedding Extraction Demo

A working demo is included in `test_audio_extraction/` with 3 sample audio files.

**Run the demo:**

```bash
python extract_audio_embeddings.py \
    --audio-dir ./test_audio_extraction/audios \
    --embedding-dir ./test_audio_extraction/embeddings \
    --json-dir ./test_audio_extraction
```

**Output:**

```
======================================================================
EXTRACTING EMBEDDINGS
======================================================================
Processing 3 audio files
Extracting embeddings: 100%|██████████| 3/3 [00:03<00:00,  1.30s/it]

✅ EXTRACTION COMPLETE
Total successful: 3
Embeddings saved to: ./test_audio_extraction/embeddings
```

**Demo folder structure:**

```
test_audio_extraction/
├── audios/                           # Input audio files
│   ├── sample_01/audio_001.wav
│   ├── sample_02/audio_002.wav
│   └── sample_03/audio_003.wav
├── embeddings/                       # Output NPY files (same structure)
│   ├── sample_01/audio_001.npy       # Shape: [255, 8], dtype: float32
│   ├── sample_02/audio_002.npy       # Shape: [214, 8], dtype: float32
│   └── sample_03/audio_003.npy       # Shape: [134, 8], dtype: float32
└── success_total_audio.json          # List of processed audio paths
```

**Success JSON content:**

```json
[
    "./test_audio_extraction/audios/sample_01/audio_001.wav",
    "./test_audio_extraction/audios/sample_02/audio_002.wav",
    "./test_audio_extraction/audios/sample_03/audio_003.wav"
]
```

### Production Usage

For the full versaface dataset:

```bash
python extract_audio_embeddings.py \
    --audio-dir /data/common/versaface/audios \
    --embedding-dir /data/common/versaface/audio_embeddings \
    --json-dir /data/common/versaface/jsons
```

---

## Project Structure

```
versaface_autoencoder/
├── models/
│   ├── tts/
│   │   ├── metis/
│   │   │   ├── audio_tokenizer.py       # AudioTokenizer class
│   │   │   ├── semantic_8d_wrappers.py  # Metis8dEncoder, Metis8dDecoder
│   │   │   ├── config/base.json         # Model configuration
│   │   │   ├── ckpt/                    # Downloaded checkpoints
│   │   │   └── test_voice_conversion/   # Voice conversion test files
│   │   └── maskgct/
│   │       ├── maskgct_s2a.py           # S2A diffusion model
│   │       ├── maskgct_utils.py         # Model builders
│   │       └── llama_nar.py             # DiffLlama architecture
│   └── codec/
│       ├── kmeans/
│       │   ├── repcodec_model.py        # RepCodec semantic codec
│       │   └── vocos.py                 # VocosBackbone
│       └── amphion_codec/
│           ├── codec.py                 # Acoustic encoder/decoder
│           ├── vocos.py                 # Vocos decoder
│           └── quantize/                # VQ modules
├── utils/
│   └── util.py                          # load_config, JsonHParams
├── test_audio_extraction/               # Demo for embedding extraction
│   ├── audios/                          # Sample audio files
│   ├── embeddings/                      # Output NPY embeddings
│   └── success_total_audio.json         # Processed file list
├── example_voice_conversion.py          # Voice conversion CLI
├── extract_audio_embeddings.py          # Batch extraction script
├── notebook.ipynb                       # Interactive examples
├── requirements.txt                     # Dependencies
└── README.md                            # This file
```

---

## Troubleshooting

### Transformers Version Error

```
LlamaAttention.forward() missing 1 required positional argument: 'position_embeddings'
```

**Fix:** Install exact version:
```bash
pip install transformers==4.40.0
```

### Permission Denied

```
PermissionError: [Errno 13] Permission denied: '/data/common/versaface/audio_embeddings'
```

**Fix:** Use a writable output directory:
```bash
python extract_audio_embeddings.py --embedding-dir /data/abi/embeddings --json-dir /data/abi/jsons
```

### GPU Out of Memory

**Fix:**
```bash
# Use CPU
python extract_audio_embeddings.py --device cpu

# Or clear cache in code
torch.cuda.empty_cache()
```

---

## License

MIT License

## Citation

If you use this code, please cite the original Metis TTS and Amphion projects.

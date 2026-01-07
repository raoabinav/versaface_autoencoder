# VersaFace Autoencoder

A neural audio autoencoder based on Metis TTS that extracts 8-dimensional continuous semantic latents from audio and enables voice conversion. This system can convert source audio content to match a reference voice's prosody and style.

## Features

- **8-D Semantic Encoding**: Extract 8-dimensional continuous latents from audio waveforms
- **Voice Conversion**: Convert source voice content using reference voice style/prosody
- **High-Quality Audio**: 24kHz output with natural voice characteristics
- **GPU Accelerated**: Supports CUDA for faster processing

## Architecture Overview

### Encoding Pipeline (Audio → 8-D Latents)

1. **Input**: Raw audio waveform (16kHz)
2. **Feature Extraction**: w2v-bert-2.0 (SeamlessM4T) extracts 1024-dimensional features from layer 17
3. **Normalization**: Features are normalized (mean subtraction, std division)
4. **Semantic Codec (RepCodec)**:
   - VocosBackbone encoder processes normalized features
   - Factorized Vector Quantization projects 1024-D → 8-D latents, then quantizes
5. **Output**: 8-D continuous latents [B, T, 8] and semantic codes [B, T] (discrete token indices)

### Voice Conversion Pipeline

1. **Source Audio** → 8-D semantic latents (content: "what to say")
2. **Reference Audio** → Acoustic codes (style: "how to say it")
3. **Decoding**: 8-D latents + Acoustic prompt → Converted audio (source content with reference style)

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- 8GB+ RAM
- 5GB+ disk space for model checkpoints

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd versaface_autoencoder
```

### Step 2: Create Virtual Environment

```bash
# Using conda (recommended)
conda create -n versaface python=3.10
conda activate versaface

# Or using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: The first time you run the code, model checkpoints will be automatically downloaded from HuggingFace (approximately 5GB). This may take 5-10 minutes depending on your internet connection.

## Quick Start

### Basic Usage: Voice Conversion

```python
import os
import torch
from utils.util import load_config
from models.tts.metis.audio_tokenizer import AudioTokenizer
from models.tts.metis.semantic_8d_wrappers import Metis8dEncoder, Metis8dDecoder
import soundfile as sf

# Setup
device = "cuda:0" if torch.cuda.is_available() else "cpu"
cfg_path = "models/tts/metis/config/base.json"
cfg = load_config(cfg_path)

# Initialize models (this downloads checkpoints on first run)
print("Loading models...")
audio_tokenizer = AudioTokenizer(cfg, device)
encoder = Metis8dEncoder(audio_tokenizer)
decoder = Metis8dDecoder(cfg, audio_tokenizer)

# Voice conversion
source_audio = "path/to/source.wav"  # Content (what to say)
reference_audio = "path/to/reference.wav"  # Style (how to say it, 1-2 seconds recommended)

# Encode source to 8-D latents
_, z_8d = encoder.encode_from_path(source_audio)

# Extract acoustic codes from reference (for style)
prompt_acoustic = encoder.encode_acoustic_from_path(reference_audio)

# Decode with voice conversion
output_wav = decoder.decode_from_z(z_8d, prompt_acoustic_code=prompt_acoustic)

# Save result
sf.write("output.wav", output_wav, 24000)
print("Voice conversion complete!")
```

### Using the Jupyter Notebook

The included `notebook.ipynb` provides a complete example with step-by-step execution:

1. Open the notebook: `jupyter notebook notebook.ipynb`
2. Run cells sequentially (Cells 0-1 initialize models, Cell 5 demonstrates voice conversion)
3. Place your audio files in `models/tts/metis/test audios/`:
   - `source-vc.wav` - Source audio (content)
   - `prompt-vc.wav` - Reference audio (style, 1-2 seconds recommended)

## Usage Examples

### Extract 8-D Latents from Audio

```python
encoder = Metis8dEncoder(audio_tokenizer)

# Extract 8-D continuous latents
feat_1024d, z_8d = encoder.encode_from_path("audio.wav")
print(f"8-D latents shape: {z_8d.shape}")  # [1, T, 8]
```

### Voice Conversion

```python
# Source provides content, reference provides style
z_8d_source = encoder.encode_from_path("source.wav")[1]
prompt_acoustic = encoder.encode_acoustic_from_path("reference.wav")

# Convert voice
output = decoder.decode_from_z(z_8d_source, prompt_acoustic_code=prompt_acoustic)
sf.write("converted.wav", output, 24000)
```

### Important Notes

- **Reference Audio Length**: For best results, use a short reference audio (1-2 seconds). Longer references will reduce the output length because the model calculates: `output_length = semantic_length - prompt_length`
- **Audio Format**: Input audio can be any sample rate; it will be automatically resampled to 16kHz for encoding and 24kHz for decoding
- **GPU Memory**: Model loading requires ~4GB GPU memory. Use CPU if GPU is unavailable (slower)

## Project Structure

```
versaface_autoencoder/
├── models/
│   ├── tts/
│   │   ├── metis/
│   │   │   ├── audio_tokenizer.py      # Audio tokenization
│   │   │   ├── semantic_8d_wrappers.py # Encoder/Decoder wrappers
│   │   │   ├── config/                 # Model configuration
│   │   │   ├── test audios/            # Test audio files
│   │   │   └── result audios/          # Output directory
│   │   └── maskgct/                    # S2A (semantic-to-acoustic) models
│   └── codec/                          # Audio codecs
├── utils/                               # Utility functions
├── notebook.ipynb                       # Interactive examples
├── requirements.txt                     # Python dependencies
└── README.md                            # This file
```

## Components

- **w2v-bert-2.0**: Self-supervised learning model for extracting semantic features from audio
- **RepCodec**: Semantic codec that compresses 1024-D features to 8-D continuous latents
- **Factorized Vector Quantization**: Quantization module for discrete code generation
- **S2A Models**: Semantic-to-acoustic conversion models (2-stage reverse diffusion)

## Troubleshooting

### Transformers Version Compatibility

If you encounter errors like `LlamaAttention.forward() missing 1 required positional argument: 'position_embeddings'`, ensure you're using transformers 4.40.0:

```bash
pip install transformers==4.40.0
```

### Model Download Issues

If model downloads fail:
- Check your internet connection
- Ensure you have sufficient disk space (~5GB)
- Models are downloaded to `models/tts/metis/ckpt/` on first run

### GPU Out of Memory

If you run out of GPU memory:
- Use CPU instead: `device = "cpu"`
- Process shorter audio segments
- Clear cache between operations: `torch.cuda.empty_cache()`

## License

MIT License

## Citation

If you use this code, please cite the original Metis TTS and Amphion projects.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

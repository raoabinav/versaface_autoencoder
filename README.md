# 8-D Semantic Encoder

A neural audio encoder that extracts 8-dimensional continuous latents from audio waveforms.

## Architecture Overview

### 8-D Semantic Encoder (Input → Output)

1. **Input**: Raw audio waveform (16kHz)
2. **Feature Extraction**: w2v-bert-2.0 (SeamlessM4T) extracts 1024-dimensional features from layer 17
3. **Normalization**: Features are normalized (mean subtraction, std division)
4. **Semantic Codec (RepCodec)**:
   - VocosBackbone encoder processes normalized features
   - Factorized Vector Quantization projects 1024-D → 8-D latents, then quantizes
5. **Output**: 8-D continuous latents [B, T, 8] and semantic codes [B, T] (discrete token indices)

## Components

- **w2v-bert-2.0**: Self-supervised learning model for extracting semantic features from audio
- **RepCodec**: Semantic codec that compresses 1024-D features to 8-D continuous latents
- **Factorized Vector Quantization**: Quantization module that projects high-dimensional features to low-dimensional discrete codes

## License

MIT License

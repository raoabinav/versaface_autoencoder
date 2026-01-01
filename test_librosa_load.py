#!/usr/bin/env python3
"""Test and demonstrate how librosa.load() handles resampling."""

import numpy as np
import librosa

print("=" * 70)
print("HOW librosa.load() WORKS")
print("=" * 70)

# Simulate different input audio files
print("\n1. Understanding librosa.load() behavior:")
print("-" * 70)

# Create test audio at different sample rates
print("\nCreating test audio signals...")
audio_24k = np.random.randn(24000).astype(np.float32)  # 1 second at 24kHz
audio_48k = np.random.randn(48000).astype(np.float32)  # 1 second at 48kHz
audio_44_1k = np.random.randn(44100).astype(np.float32)  # 1 second at 44.1kHz

print(f"  Original 24kHz audio: {len(audio_24k)} samples (1 second)")
print(f"  Original 48kHz audio: {len(audio_48k)} samples (1 second)")
print(f"  Original 44.1kHz audio: {len(audio_44_1k)} samples (1 second)")

# Test resampling
print("\n2. Testing librosa.resample() (what librosa.load() uses internally):")
print("-" * 70)

# Resample 24kHz -> 16kHz
resampled_24k_to_16k = librosa.resample(audio_24k, orig_sr=24000, target_sr=16000)
print(f"  24kHz -> 16kHz: {len(audio_24k)} -> {len(resampled_24k_to_16k)} samples")
print(f"    Ratio: {len(resampled_24k_to_16k)/len(audio_24k):.4f} (expected 16/24 = 0.6667)")
print(f"    Duration preserved: ~{len(resampled_24k_to_16k)/16000:.2f}s")

# Resample 48kHz -> 16kHz
resampled_48k_to_16k = librosa.resample(audio_48k, orig_sr=48000, target_sr=16000)
print(f"  48kHz -> 16kHz: {len(audio_48k)} -> {len(resampled_48k_to_16k)} samples")
print(f"    Ratio: {len(resampled_48k_to_16k)/len(audio_48k):.4f} (expected 16/48 = 0.3333)")

# Resample 44.1kHz -> 16kHz
resampled_44_1k_to_16k = librosa.resample(audio_44_1k, orig_sr=44100, target_sr=16000)
print(f"  44.1kHz -> 16kHz: {len(audio_44_1k)} -> {len(resampled_44_1k_to_16k)} samples")
print(f"    Ratio: {len(resampled_44_1k_to_16k)/len(audio_44_1k):.4f} (expected 16/44.1 = 0.3628)")

print("\n3. How librosa.load() works:")
print("-" * 70)
print("""
When you call:
  librosa.load(wav_path, sr=16000)

librosa.load() does:
  1. Loads audio file (using soundfile or audioread backend)
  2. Gets native sample rate from file
  3. If native_sr != target_sr (16000):
     - Calls librosa.resample() internally
     - Uses 'soxr_hq' (high quality) resampling by default
  4. Returns (audio_array, target_sr)

So it AUTOMATICALLY handles resampling!
""")

print("4. Resampling algorithm:")
print("-" * 70)
print("""
Default: res_type='soxr_hq' (Sox Resampler High Quality)
- Uses soxr library (if available)
- High-quality resampling algorithm
- Preserves audio quality well

Alternative options:
- 'kaiser_best': Highest quality (slower)
- 'kaiser_fast': Faster, slightly lower quality
- 'scipy': Uses scipy.signal.resample
- 'fft': FFT-based resampling
""")

print("5. In our code:")
print("-" * 70)
print("""
# This works with ANY input sample rate:
wav_16k = librosa.load(wav_path, sr=16000, mono=True)[0]

What happens:
1. File is 24kHz -> librosa.load() resamples to 16kHz
2. File is 48kHz -> librosa.load() resamples to 16kHz  
3. File is 16kHz -> librosa.load() returns as-is (no resampling)
4. File is 44.1kHz -> librosa.load() resamples to 16kHz

The resampling happens AUTOMATICALLY inside librosa.load()!
""")

print("\n" + "=" * 70)
print("SUMMARY: librosa.load() is the preprocessing layer!")
print("=" * 70)
print("""
- Automatically resamples to target rate
- Works with any input sample rate
- Uses high-quality resampling (soxr_hq)
- No manual resampling needed
- This is why our code works with 24kHz, 48kHz, etc.
""")


#!/usr/bin/env python3
"""
Simple example script for voice conversion using VersaFace Autoencoder.

Usage:
    python example_voice_conversion.py --source path/to/source.wav --reference path/to/reference.wav --output output.wav
"""

import os
import sys
import argparse
import torch
import soundfile as sf

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from utils.util import load_config
from models.tts.metis.audio_tokenizer import AudioTokenizer
from models.tts.metis.semantic_8d_wrappers import Metis8dEncoder, Metis8dDecoder


def main():
    parser = argparse.ArgumentParser(
        description="Voice conversion using VersaFace Autoencoder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python example_voice_conversion.py --source source.wav --reference ref.wav --output result.wav
  
  # Use CPU instead of GPU
  python example_voice_conversion.py --source source.wav --reference ref.wav --output result.wav --device cpu
        """
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to source audio file (provides content - what to say)"
    )
    parser.add_argument(
        "--reference",
        type=str,
        required=True,
        help="Path to reference audio file (provides style - how to say it, 1-2 seconds recommended)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output audio file"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda", "cuda:0", "cuda:1"],
        help="Device to use (default: auto-detect, prefers CUDA if available)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="models/tts/metis/config/base.json",
        help="Path to config file (default: models/tts/metis/config/base.json)"
    )

    args = parser.parse_args()

    # Validate input files
    if not os.path.exists(args.source):
        print(f"Error: Source audio file not found: {args.source}")
        sys.exit(1)
    
    if not os.path.exists(args.reference):
        print(f"Error: Reference audio file not found: {args.reference}")
        sys.exit(1)

    # Setup device
    if args.device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    print(f"Source audio: {args.source}")
    print(f"Reference audio: {args.reference}")
    print(f"Output: {args.output}")
    print()

    # Load config
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    
    print("Loading configuration...")
    cfg = load_config(args.config)

    # Initialize models
    print("\n" + "=" * 70)
    print("INITIALIZING MODELS")
    print("=" * 70)
    print("(This may take a few minutes on first run as models are downloaded)")
    print()

    print("Loading AudioTokenizer...")
    audio_tokenizer = AudioTokenizer(cfg, device)
    print("✓ AudioTokenizer loaded")

    print("Loading Metis8dEncoder...")
    encoder = Metis8dEncoder(audio_tokenizer)
    print("✓ Encoder loaded")

    print("Loading Metis8dDecoder...")
    decoder = Metis8dDecoder(cfg, audio_tokenizer)
    print("✓ Decoder loaded")
    print()

    # Voice conversion
    print("=" * 70)
    print("VOICE CONVERSION")
    print("=" * 70)
    print()

    print("[STEP 1] Encoding source audio to 8-D semantic latents...")
    _, z_8d_source = encoder.encode_from_path(args.source)
    print(f"✓ 8-D latents shape: {z_8d_source.shape}")

    print("\n[STEP 2] Extracting acoustic codes from reference audio...")
    prompt_acoustic_code = encoder.encode_acoustic_from_path(args.reference)
    print(f"✓ Acoustic codes shape: {prompt_acoustic_code.shape}")
    print(f"  Reference duration: ~{prompt_acoustic_code.shape[1] / 50:.2f} seconds")
    
    if prompt_acoustic_code.shape[1] > 100:
        print(f"  ⚠️  Warning: Reference audio is long. For best results, use 1-2 seconds.")
        print(f"     Output length will be: ~{z_8d_source.shape[1] - prompt_acoustic_code.shape[1]} tokens")

    print("\n[STEP 3] Converting voice...")
    output_wav = decoder.decode_from_z(z_8d_source, prompt_acoustic_code=prompt_acoustic_code)
    print(f"✓ Generated waveform: {len(output_wav)} samples")
    print(f"  Duration: {len(output_wav) / 24000:.2f} seconds")

    # Save output
    print(f"\n[STEP 4] Saving output...")
    sf.write(args.output, output_wav, 24000)
    print(f"✓ Saved to: {args.output}")

    print("\n" + "=" * 70)
    print("✅ VOICE CONVERSION COMPLETE")
    print("=" * 70)
    print(f"\nOutput audio has:")
    print(f"  - Content from: {os.path.basename(args.source)}")
    print(f"  - Style from: {os.path.basename(args.reference)}")
    print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()


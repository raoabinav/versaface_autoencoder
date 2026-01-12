#!/usr/bin/env python3
"""
Test the 8-D wrapper end-to-end.

Tests:
1. Encode → Decode roundtrip (reconstruction quality)
2. Voice conversion (different source/reference speakers)
"""

import os
import sys
import torch
import numpy as np
import soundfile as sf

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from utils.util import load_config
from models.tts.metis.audio_tokenizer import AudioTokenizer
from models.tts.metis.semantic_8d_wrappers import Metis8dEncoder, Metis8dDecoder


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Setup paths
    test_dir = "models/tts/metis/test_voice_conversion"
    output_dir = "models/tts/metis/test_8d_wrapper"
    os.makedirs(output_dir, exist_ok=True)
    
    source_path = f"{test_dir}/source.wav"
    reference_path = f"{test_dir}/trump.wav"
    
    if not os.path.exists(source_path) or not os.path.exists(reference_path):
        print(f"Error: Test files not found in {test_dir}")
        print("Please ensure source.wav and trump.wav exist")
        sys.exit(1)
    
    # Load config and models
    print("\n" + "=" * 70)
    print("LOADING MODELS")
    print("=" * 70)
    
    cfg = load_config("models/tts/metis/config/base.json")
    audio_tokenizer = AudioTokenizer(cfg, device)
    encoder = Metis8dEncoder(audio_tokenizer)
    decoder = Metis8dDecoder(cfg, audio_tokenizer)
    
    print("✓ All models loaded\n")
    
    # =========================================================================
    # TEST 1: Roundtrip (Encode → Decode, same speaker)
    # =========================================================================
    print("=" * 70)
    print("TEST 1: ROUNDTRIP (Encode → Decode)")
    print("=" * 70)
    print("This tests if audio survives the 8-D bottleneck without voice transfer.\n")
    
    # Encode source to 8-D
    print("[1.1] Encoding source audio to 8-D latents...")
    feat_1024d, z_8d = encoder.encode_from_path(source_path)
    print(f"      1024-D features shape: {feat_1024d.shape}")
    print(f"      8-D latents shape: {z_8d.shape}")
    
    # Decode back without any prompt (no voice transfer)
    print("[1.2] Decoding 8-D latents back to audio (no voice prompt)...")
    roundtrip_wav = decoder.decode_from_z(z_8d)
    print(f"      Output waveform: {len(roundtrip_wav)} samples ({len(roundtrip_wav)/24000:.2f}s)")
    
    # Save roundtrip output
    roundtrip_path = f"{output_dir}/roundtrip_source.wav"
    sf.write(roundtrip_path, roundtrip_wav, 24000)
    print(f"      Saved: {roundtrip_path}")
    
    # =========================================================================
    # TEST 2: Self-reconstruction (use same audio as source AND reference)
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 2: SELF-RECONSTRUCTION")
    print("=" * 70)
    print("Uses source audio as both content AND style reference.\n")
    
    print("[2.1] Extracting prompt codes from source audio...")
    prompt_semantic, prompt_acoustic = encoder.encode_prompt_from_path(source_path)
    print(f"      Prompt semantic: {prompt_semantic.shape}")
    print(f"      Prompt acoustic: {prompt_acoustic.shape}")
    
    print("[2.2] Decoding with self-prompt...")
    self_recon_wav = decoder.decode_from_z(
        z_8d,
        prompt_acoustic_code=prompt_acoustic,
        prompt_semantic_code=prompt_semantic
    )
    print(f"      Output waveform: {len(self_recon_wav)} samples ({len(self_recon_wav)/24000:.2f}s)")
    
    self_recon_path = f"{output_dir}/self_reconstruction.wav"
    sf.write(self_recon_path, self_recon_wav, 24000)
    print(f"      Saved: {self_recon_path}")
    
    # =========================================================================
    # TEST 3: Voice Conversion (different source and reference)
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 3: VOICE CONVERSION (female → Trump)")
    print("=" * 70)
    print("Source: female speech (content)")
    print("Reference: Trump (voice/style)\n")
    
    print("[3.1] Extracting prompt codes from Trump reference...")
    trump_semantic, trump_acoustic = encoder.encode_prompt_from_path(reference_path)
    print(f"      Trump semantic: {trump_semantic.shape}")
    print(f"      Trump acoustic: {trump_acoustic.shape}")
    
    print("[3.2] Voice conversion via 8-D wrapper...")
    vc_wav = decoder.decode_from_z(
        z_8d,  # Content from source (female)
        prompt_acoustic_code=trump_acoustic,  # Voice from Trump
        prompt_semantic_code=trump_semantic   # Speaker context from Trump
    )
    print(f"      Output waveform: {len(vc_wav)} samples ({len(vc_wav)/24000:.2f}s)")
    
    vc_path = f"{output_dir}/vc_8d_wrapper.wav"
    sf.write(vc_path, vc_wav, 24000)
    print(f"      Saved: {vc_path}")
    
    # =========================================================================
    # TEST 4: Encode Trump and decode with source voice
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 4: REVERSE VC (Trump content → female voice)")
    print("=" * 70)
    print("Source: Trump speech (content)")
    print("Reference: female (voice/style)\n")
    
    print("[4.1] Encoding Trump audio to 8-D latents...")
    _, z_8d_trump = encoder.encode_from_path(reference_path)
    print(f"      Trump 8-D latents shape: {z_8d_trump.shape}")
    
    print("[4.2] Extracting prompt codes from female source...")
    female_semantic, female_acoustic = encoder.encode_prompt_from_path(source_path)
    
    print("[4.3] Reverse voice conversion...")
    reverse_vc_wav = decoder.decode_from_z(
        z_8d_trump,  # Content from Trump
        prompt_acoustic_code=female_acoustic,  # Voice from female
        prompt_semantic_code=female_semantic   # Speaker context from female
    )
    print(f"      Output waveform: {len(reverse_vc_wav)} samples ({len(reverse_vc_wav)/24000:.2f}s)")
    
    reverse_vc_path = f"{output_dir}/vc_reverse_trump_to_female.wav"
    sf.write(reverse_vc_path, reverse_vc_wav, 24000)
    print(f"      Saved: {reverse_vc_path}")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST COMPLETE - SUMMARY")
    print("=" * 70)
    print(f"\nAll outputs saved to: {output_dir}/")
    print("\nGenerated files:")
    print(f"  1. roundtrip_source.wav      - Source encoded→decoded (no voice change)")
    print(f"  2. self_reconstruction.wav   - Source with self as reference")
    print(f"  3. vc_8d_wrapper.wav         - Female content → Trump voice")
    print(f"  4. vc_reverse_trump_to_female.wav - Trump content → Female voice")
    print("\nExpected results:")
    print("  - Test 1 & 2: Should sound like original source (content preserved)")
    print("  - Test 3: Should have Trump's voice with female's words")
    print("  - Test 4: Should have female's voice with Trump's words")
    print()


if __name__ == "__main__":
    main()

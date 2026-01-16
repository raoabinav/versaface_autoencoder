#!/usr/bin/env python3
"""
Text-to-Speech using MaskGCT.

Generate speech in a reference speaker's voice from text input.

Example:
    python example_tts.py \
        --text "Hello, I am speaking in Trump's voice." \
        --reference models/tts/metis/test_voice_conversion/trump.wav \
        --output output.wav

The reference audio provides the voice/style.
The text provides what to say.
"""

import os
import sys
import argparse
import torch
import numpy as np
import librosa
import soundfile as sf
import safetensors.torch

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from utils.util import load_config
from models.tts.maskgct.maskgct_utils import (
    build_s2a_model, build_semantic_model, 
    build_semantic_codec, build_acoustic_codec, g2p_
)
from models.tts.maskgct.maskgct_t2s import MaskGCT_T2S
from transformers import SeamlessM4TFeatureExtractor


def transcribe_audio(audio_path):
    """Transcribe audio using Whisper."""
    try:
        import whisper
        audio, _ = librosa.load(audio_path, sr=16000)
        model = whisper.load_model('base')
        result = model.transcribe(audio)
        return result['text'].strip()
    except ImportError:
        print("Warning: whisper not installed. Using placeholder prompt text.")
        return "hello"


def main():
    parser = argparse.ArgumentParser(
        description="Text-to-Speech using MaskGCT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Text to synthesize"
    )
    parser.add_argument(
        "--reference",
        type=str,
        required=True,
        help="Path to reference audio (provides voice/style)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output audio file"
    )
    parser.add_argument(
        "--prompt-text",
        type=str,
        default=None,
        help="Text spoken in reference audio (auto-transcribed if not provided)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (default: auto-detect)"
    )

    args = parser.parse_args()

    # Validate
    if not os.path.exists(args.reference):
        print(f"Error: Reference audio not found: {args.reference}")
        sys.exit(1)

    # Device
    if args.device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"Device: {device}")
    print(f"Reference: {args.reference}")
    print(f"Target text: \"{args.text}\"")
    print()

    # Config and paths
    cfg = load_config("models/tts/metis/config/base.json")
    ckpt_base = "./models/tts/maskgct/ckpt"

    # Load models
    print("Loading models...")
    semantic_model, semantic_mean, semantic_std = build_semantic_model(device)
    semantic_codec = build_semantic_codec(cfg.model.semantic_codec, device)
    codec_encoder, codec_decoder = build_acoustic_codec(cfg.model.acoustic_codec, device)

    t2s_cfg = cfg.model.t2s_model
    t2s_cfg.use_phone_cond = True
    t2s_model = MaskGCT_T2S(cfg=t2s_cfg)
    t2s_model.eval().to(device)
    safetensors.torch.load_model(t2s_model, f"{ckpt_base}/t2s_model/model.safetensors")

    s2a_1layer = build_s2a_model(cfg.model.s2a_model.s2a_1layer, device)
    s2a_full = build_s2a_model(cfg.model.s2a_model.s2a_full, device)
    safetensors.torch.load_model(s2a_1layer, f"{ckpt_base}/s2a_model/s2a_model_1layer/model.safetensors")
    safetensors.torch.load_model(s2a_full, f"{ckpt_base}/s2a_model/s2a_model_full/model.safetensors")

    safetensors.torch.load_model(semantic_codec, f"{ckpt_base}/semantic_codec/model.safetensors")
    safetensors.torch.load_model(codec_encoder, f"{ckpt_base}/acoustic_codec/model.safetensors")
    safetensors.torch.load_model(codec_decoder, f"{ckpt_base}/acoustic_codec/model_1.safetensors")

    processor = SeamlessM4TFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
    print("✓ Models loaded")

    # Load reference audio
    print("\nExtracting reference features...")
    ref_16k = librosa.load(args.reference, sr=16000)[0]
    ref_24k = librosa.load(args.reference, sr=24000)[0]

    inputs = processor(ref_16k, sampling_rate=16000, return_tensors="pt")
    input_features = inputs["input_features"][0].unsqueeze(0).to(device)
    attention_mask = inputs["attention_mask"][0].unsqueeze(0).to(device)

    with torch.no_grad():
        vq_emb = semantic_model(
            input_features=input_features,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        feat = vq_emb.hidden_states[17]
        feat = (feat - semantic_mean) / semantic_std
        prompt_semantic_code, _ = semantic_codec.quantize(feat)

        ref_tensor = torch.tensor(ref_24k).unsqueeze(0).to(device)
        vq_emb_ac = codec_encoder(ref_tensor.unsqueeze(1))
        _, vq, _, _, _ = codec_decoder.quantizer(vq_emb_ac)
        prompt_acoustic_code = vq.permute(1, 2, 0)

    print(f"✓ Reference: {len(ref_16k)/16000:.1f}s, semantic: {prompt_semantic_code.shape[1]} tokens")

    # Get prompt text (what reference speaker says)
    if args.prompt_text:
        prompt_text = args.prompt_text
    else:
        print("\nTranscribing reference audio...")
        prompt_text = transcribe_audio(args.reference)
    print(f"✓ Prompt text: \"{prompt_text}\"")

    # G2P
    print("\nConverting to phonemes...")
    _, prompt_phone_id = g2p_(prompt_text, "en")
    _, target_phone_id = g2p_(args.text, "en")
    phone_id = torch.tensor(prompt_phone_id + target_phone_id, dtype=torch.long).unsqueeze(0).to(device)
    print(f"✓ Phonemes: {len(prompt_phone_id)} prompt + {len(target_phone_id)} target")

    # Target length
    target_len = int(len(ref_16k) / 16000 * 50 * len(target_phone_id) / max(len(prompt_phone_id), 1))
    target_len = min(max(target_len, 50), 500)
    print(f"✓ Target length: {target_len} tokens (~{target_len/50:.1f}s)")

    # T2S
    print("\nGenerating semantic codes...")
    with torch.no_grad():
        predict_semantic = t2s_model.reverse_diffusion(
            prompt_semantic_code,
            target_len,
            phone_id,
            n_timesteps=25,
            cfg=2.5,
            rescale_cfg=0.75,
        )
        combined_semantic = torch.cat([prompt_semantic_code, predict_semantic], dim=-1)
    print(f"✓ Generated: {predict_semantic.shape[1]} tokens")

    # S2A
    print("Generating acoustic codes...")
    with torch.no_grad():
        cond = s2a_1layer.cond_emb(combined_semantic)
        pred_1layer = s2a_1layer.reverse_diffusion(
            cond=cond, prompt=prompt_acoustic_code,
            temp=1.5, filter_thres=0.98, n_timesteps=[25],
            cfg=2.5, rescale_cfg=0.75,
        )

        cond = s2a_full.cond_emb(combined_semantic)
        pred_full = s2a_full.reverse_diffusion(
            cond=cond, prompt=prompt_acoustic_code,
            temp=1.5, filter_thres=0.98,
            n_timesteps=[25, 10, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            cfg=2.5, rescale_cfg=0.75, gt_code=pred_1layer,
        )

    # Decode
    print("Decoding to audio...")
    with torch.no_grad():
        vq_emb = codec_decoder.vq2emb(pred_full.permute(2, 0, 1), n_quantizers=12)
        audio = codec_decoder(vq_emb)[0][0].cpu().numpy()

    # Save
    sf.write(args.output, audio, 24000)
    print(f"\n✅ Saved: {args.output} ({len(audio)/24000:.2f}s)")


if __name__ == "__main__":
    main()

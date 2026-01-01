# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import numpy as np
import librosa

import safetensors


from transformers import SeamlessM4TFeatureExtractor

from models.tts.maskgct.maskgct_utils import (
    build_semantic_model,
    build_semantic_codec,
    build_acoustic_codec,
)

from huggingface_hub import snapshot_download


class AudioTokenizer:
    def __init__(self, cfg, device):

        self.cfg = cfg
        self.device = device

        self.processor = SeamlessM4TFeatureExtractor.from_pretrained(
            "facebook/w2v-bert-2.0"
        )

        self.semantic_model, self.semantic_mean, self.semantic_std = (
            build_semantic_model(self.device)
        )

        # build semantic codec
        self.semantic_codec = build_semantic_codec(
            self.cfg.model.semantic_codec, self.device
        )

        # load semantic codec
        semantic_code_dir = snapshot_download(
            repo_id="amphion/MaskGCT",
            repo_type="model",
            local_dir="./models/tts/metis/ckpt",
            allow_patterns=["semantic_codec/model.safetensors"],
        )
        semantic_code_ckpt = os.path.join(
            semantic_code_dir, "semantic_codec/model.safetensors"
        )
        # Use strict=False to handle potential shared tensor issues
        try:
            safetensors.torch.load_model(self.semantic_codec, semantic_code_ckpt, strict=False)
        except RuntimeError as e:
            if "shared tensor" in str(e).lower():
                # Fallback: load manually
                from safetensors import safe_open
                with safe_open(semantic_code_ckpt, framework="pt", device=str(self.device)) as f:
                    state_dict = {key: f.get_tensor(key) for key in f.keys()}
                    state_dict = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in state_dict.items()}
                    self.semantic_codec.load_state_dict(state_dict, strict=False)
            else:
                raise

        # build acoustic codec
        self.codec_encoder, self.codec_decoder = build_acoustic_codec(
            self.cfg.model.acoustic_codec, self.device
        )

        # download acoustic codec ckpt
        acoustic_code_dir = snapshot_download(
            repo_id="amphion/MaskGCT",
            repo_type="model",
            local_dir="./models/tts/metis/ckpt",
            allow_patterns=["acoustic_codec/model.safetensors"],
        )
        codec_encoder_ckpt = os.path.join(
            acoustic_code_dir, "acoustic_codec/model.safetensors"
        )

        # download acoustic codec ckpt
        codec_decoder_ckpt = snapshot_download(
            repo_id="amphion/MaskGCT",
            repo_type="model",
            local_dir="./models/tts/metis/ckpt",
            allow_patterns=["acoustic_codec/model_1.safetensors"],
        )
        codec_decoder_ckpt = os.path.join(
            codec_decoder_ckpt, "acoustic_codec/model_1.safetensors"
        )

        # load acoustic codec
        # Handle shared tensor issues by loading manually
        from safetensors import safe_open
        
        # Load encoder
        with safe_open(codec_encoder_ckpt, framework="pt", device=str(self.device)) as f:
            state_dict = {key: f.get_tensor(key) for key in f.keys()}
            self.codec_encoder.load_state_dict(state_dict, strict=False)
        
        # Load decoder (this one has the shared tensor issue)
        with safe_open(codec_decoder_ckpt, framework="pt", device=str(self.device)) as f:
            state_dict = {}
            for key in f.keys():
                tensor = f.get_tensor(key)
                # Skip problematic shared tensor keys if they cause issues
                if "istft.window" in key and tensor.storage_offset() > 0:
                    # This is a shared tensor view, skip it
                    continue
                state_dict[key] = tensor
            self.codec_decoder.load_state_dict(state_dict, strict=False)

    def __call__(
        self,
        speech_16k: np.ndarray = None,
        speech: np.ndarray = None,
        speech_path: str = None,
    ):
        """Extract semantic and acoustic codes from audio."""
        if speech_path is not None:
            speech_16k = librosa.load(speech_path, sr=16000)[0]
            speech = librosa.load(speech_path, sr=24000)[0]

        semantic_code, rec_feat = self.wav2semantic(speech_16k)
        acoustic_code = self.wav2acoustic(speech)
        return semantic_code, rec_feat, acoustic_code

    @torch.no_grad()
    def wav2semantic(self, speech: np.ndarray):
        """Extract semantic codes from 16kHz audio."""
        input_features, attention_mask = self._extract_features(speech)
        input_features = input_features.unsqueeze(0).to(self.device)
        attention_mask = attention_mask.unsqueeze(0).to(self.device)
        semantic_code, rec_feat = self._extract_semantic_code(
            input_features, attention_mask
        )
        return semantic_code, rec_feat

    @torch.no_grad()
    def wav2acoustic(self, speech: np.ndarray):
        """Extract acoustic codes from 24kHz audio."""
        speech = torch.tensor(speech).unsqueeze(0).to(self.device)
        acoustic_code = self._extract_acoustic_code(speech)
        return acoustic_code

    @torch.no_grad()
    def wav2semantic_feat(self, speech: np.ndarray):
        """Extract semantic features (1024-D) from 16kHz audio."""
        input_features, attention_mask = self._extract_features(speech)
        input_features = input_features.unsqueeze(0).to(self.device)
        attention_mask = attention_mask.unsqueeze(0).to(self.device)
        semantic_feat = self._extract_semantic_feat(input_features, attention_mask)
        return semantic_feat

    @torch.no_grad()
    def _extract_features(
        self,
        speech: np.ndarray,
    ):
        inputs = self.processor(speech, sampling_rate=16000, return_tensors="pt")
        input_features = inputs["input_features"][0]
        attention_mask = inputs["attention_mask"][0]
        return input_features, attention_mask

    @torch.no_grad()
    def _extract_semantic_code(self, input_features, attention_mask):
        vq_emb = self.semantic_model(
            input_features=input_features,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        feat = vq_emb.hidden_states[17]  # (B, T, C)
        feat = (feat - self.semantic_mean.to(feat)) / self.semantic_std.to(feat)

        semantic_code, rec_feat = self.semantic_codec.quantize(feat)
        return semantic_code, rec_feat

    @torch.no_grad()
    def _extract_semantic_feat(self, input_features, attention_mask):
        vq_emb = self.semantic_model(
            input_features=input_features,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        feat = vq_emb.hidden_states[17]  # (B, T, C)
        feat = (feat - self.semantic_mean.to(feat)) / self.semantic_std.to(feat)
        return feat

    @torch.no_grad()
    def _extract_acoustic_code(self, speech):
        vq_emb = self.codec_encoder(speech.unsqueeze(1))
        _, vq, _, _, _ = self.codec_decoder.quantizer(vq_emb)
        # vq is all_indices: [num_quantizers, B, T]
        # After permute(1, 2, 0): [B, T, num_quantizers]
        acoustic_code = vq.permute(1, 2, 0)  # [num_quantizers, B, T] -> [B, T, num_quantizers]
        return acoustic_code

    @torch.no_grad()
    def code2wav(self, acoustic_code):
        vq_emb = self.codec_decoder.vq2emb(
            acoustic_code.permute(2, 0, 1), n_quantizers=12
        )
        wav = self.codec_decoder(vq_emb)
        wav = wav[0][0].cpu().numpy()
        return wav

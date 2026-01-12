# models/tts/metis/semantic_8d_wrappers.py

import os
import warnings
import logging
from typing import Optional, Tuple

import numpy as np
import torch
import librosa
import safetensors.torch

from models.tts.metis.audio_tokenizer import AudioTokenizer
from models.tts.maskgct.maskgct_utils import build_s2a_model
from huggingface_hub import snapshot_download

# Configure logging
logger = logging.getLogger(__name__)

# Audio processing constants
SEMANTIC_SAMPLE_RATE = 16000  # Hz - Input sample rate for semantic encoding
ACOUSTIC_SAMPLE_RATE = 24000  # Hz - Input/output sample rate for acoustic processing
OUTPUT_SAMPLE_RATE = 24000  # Hz - Final output waveform sample rate
SEMANTIC_RATE_HZ = 50  # Semantic code rate in Hz (tokens per second)

# Model architecture constants
W2V_BERT_LAYER = 17  # Layer index for feature extraction from w2v-bert-2.0
ACOUSTIC_QUANTIZERS = 12  # Number of quantizers in acoustic codec
SEMANTIC_CODEBOOK_SIZE = 8192  # Size of semantic codebook

# Prompt length constants
MAX_PROMPT_LENGTH_TOKENS = 100  # Maximum prompt tokens (recommended: 1-2 seconds)
MAX_PROMPT_LENGTH_SECONDS = 2.0  # Maximum prompt duration in seconds
PROMPT_LENGTH_RATIO = 3  # Prompt should be at most 1/N of semantic length


class Metis8dEncoder:
    """
    Wraps AudioTokenizer to expose 8-D continuous latents BEFORE VQ quantization.
    
    Handles all ENCODING operations:
    - Audio → 8-D latents (semantic encoding)
    - Audio → acoustic codes (acoustic encoding)
    
    Use this class to extract both semantic and acoustic representations from audio.
    The encoder is responsible for all encoding tasks; the decoder only handles decoding.
    """

    def __init__(self, audio_tokenizer: AudioTokenizer):
        self.audio_tok = audio_tokenizer
        self.device = audio_tokenizer.device
        self.semantic_model = audio_tokenizer.semantic_model
        self.semantic_codec = audio_tokenizer.semantic_codec
        self.semantic_mean = audio_tokenizer.semantic_mean
        self.semantic_std = audio_tokenizer.semantic_std
        self.processor = audio_tokenizer.processor
        self.semantic_model.eval()
        self.semantic_codec.eval()

        # Buffer to hold 8D latents
        self._z8_buffer: Optional[torch.Tensor] = None

        # Find the quantizer's in_project layer (this creates the 8-D latents)
        self._in_project_module = self._get_in_project_module()
        
        # Register hook on in_project to capture 8-D latents
        self._hook_handle = self._in_project_module.register_forward_hook(
            self._in_project_forward_hook
        )

    def _get_in_project_module(self) -> torch.nn.Module:
        """Get FactorizedVectorQuantize.in_project layer: RepCodec.quantizer.quantizers[0].in_project"""
        try:
            return self.semantic_codec.quantizer.quantizers[0].in_project
        except (AttributeError, IndexError) as e:
            raise RuntimeError(f"Could not find in_project layer: {e}")

    def _in_project_forward_hook(self, module, inputs, output):
        """Capture 8-D latents: [B, 8, T] → [B, T, 8]"""
        self._z8_buffer = output.transpose(1, 2).detach()

    @torch.no_grad()
    def encode_waveform_16k(
        self,
        wav_16k: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        16kHz waveform → (1024-D features, 8-D latents)
        
        Note: Input should be at 16kHz. If loading from file, use encode_from_path()
        which automatically handles resampling via librosa.load(sr=16000).
        """
        feat_norm = self._compute_ssl_feat(wav_16k)
        self._z8_buffer = None

        self.semantic_codec.quantize(feat_norm)  # Triggers hook
        
        if self._z8_buffer is None:
            raise RuntimeError("Hook did not fire - check quantize() implementation")
        
        return feat_norm, self._z8_buffer


    @torch.no_grad()
    def _compute_ssl_feat(self, wav_16k: np.ndarray) -> torch.Tensor:
        """
        Compute normalized SSL features from 16kHz audio.
        
        Args:
            wav_16k: Audio waveform at 16kHz sample rate
            
        Returns:
            Normalized SSL features tensor of shape [1, T, 1024]
            
        Raises:
            RuntimeError: If audio processing fails
        """
        if len(wav_16k) == 0:
            raise ValueError("Input audio is empty")
        
        inputs = self.processor(
            wav_16k,
            sampling_rate=SEMANTIC_SAMPLE_RATE,
            return_tensors="pt",
        )
        input_features = inputs["input_features"][0].unsqueeze(0).to(self.device)
        attention_mask = inputs["attention_mask"][0].unsqueeze(0).to(self.device)

        vq_emb = self.semantic_model(
            input_features=input_features,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        feat = vq_emb.hidden_states[W2V_BERT_LAYER]  # [1, T_ssl, 1024]
        feat_norm = (feat - self.semantic_mean.to(feat)) / self.semantic_std.to(feat)
        return feat_norm

    @torch.no_grad()
    def encode_from_path(self, wav_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load .wav file and encode to (1024-D features, 8-D latents).
        
        The audio is automatically resampled to 16kHz by librosa.load() if needed.
        This works with any input sampling rate (24kHz, 48kHz, etc.).
        
        Args:
            wav_path: Path to .wav file (any sampling rate)
            
        Returns:
            Tuple containing:
            - feat_1024d: [1, T_ssl, 1024] normalized SSL features
            - z_8d: [1, T_ssl, 8] 8-D continuous latents
            
        Raises:
            FileNotFoundError: If audio file doesn't exist
            ValueError: If audio file cannot be loaded
        """
        if not os.path.exists(wav_path):
            raise FileNotFoundError(
                f"Audio file not found: {wav_path}\n"
                f"Please ensure the file exists and the path is correct."
            )
        
        try:
            # librosa.load() automatically resamples to 16kHz if input is different rate
            wav_16k, _ = librosa.load(wav_path, sr=SEMANTIC_SAMPLE_RATE, mono=True)
            if len(wav_16k) == 0:
                raise ValueError(f"Audio file is empty: {wav_path}")
            return self.encode_waveform_16k(wav_16k)
        except Exception as e:
            raise ValueError(
                f"Failed to load audio file: {wav_path}\n"
                f"Error: {str(e)}\n"
                f"Please ensure the file is a valid audio format (WAV, MP3, etc.)"
            ) from e

    @torch.no_grad()
    def wav2acoustic(self, wav_24k: np.ndarray) -> torch.Tensor:
        """
        Extract acoustic codes from 24kHz audio.
        
        Args:
            wav_24k: 24kHz waveform [T]
            
        Returns:
            acoustic_code: [1, T_ac, 12] acoustic codes (long tensor)
        """
        acoustic_code = self.audio_tok.wav2acoustic(wav_24k)  # [T, 12]
        # Convert to [1, T, 12] format expected by S2A models
        if acoustic_code.dim() == 2:
            acoustic_code = acoustic_code.unsqueeze(0)  # [1, T, 12]
        return acoustic_code.long()

    @torch.no_grad()
    def encode_acoustic_from_path(self, wav_path: str) -> torch.Tensor:
        """
        Load .wav file and extract acoustic codes.
        
        The audio is automatically resampled to 24kHz by librosa.load() if needed.
        This works with any input sampling rate.
        
        Args:
            wav_path: Path to .wav file (any sampling rate, will be resampled to 24kHz)
            
        Returns:
            acoustic_code: [1, T_ac, 12] acoustic codes (long tensor)
            
        Raises:
            FileNotFoundError: If audio file doesn't exist
            ValueError: If audio file cannot be loaded
        """
        if not os.path.exists(wav_path):
            raise FileNotFoundError(
                f"Audio file not found: {wav_path}\n"
                f"Please ensure the file exists and the path is correct."
            )
        
        try:
            # librosa.load() automatically resamples to 24kHz if input is different rate
            wav_24k, _ = librosa.load(wav_path, sr=ACOUSTIC_SAMPLE_RATE, mono=True)
            if len(wav_24k) == 0:
                raise ValueError(f"Audio file is empty: {wav_path}")
            return self.wav2acoustic(wav_24k)
        except Exception as e:
            raise ValueError(
                f"Failed to load audio file: {wav_path}\n"
                f"Error: {str(e)}\n"
                f"Please ensure the file is a valid audio format (WAV, MP3, etc.)"
            ) from e

    @torch.no_grad()
    def encode_full_from_path(
        self, wav_path: str
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract both semantic (8-D latents) and acoustic codes from audio.
        This is useful when you want to use the same audio as both semantic and acoustic prompt.
        
        Args:
            wav_path: Path to .wav file
            
        Returns:
            feat_1024d: [1, T_ssl, 1024] normalized SSL features
            z_8d: [1, T_ssl, 8] 8-D continuous latents
            acoustic_code: [1, T_ac, 12] acoustic codes
        """
        # Load audio at both sample rates
        wav_16k = librosa.load(wav_path, sr=16000, mono=True)[0]
        wav_24k = librosa.load(wav_path, sr=24000, mono=True)[0]
        
        # Extract semantic features and 8-D latents
        feat_1024d, z_8d = self.encode_waveform_16k(wav_16k)
        
        # Extract acoustic codes
        acoustic_code = self.wav2acoustic(wav_24k)
        
        return feat_1024d, z_8d, acoustic_code

        
class Metis8dDecoder:
    """
    8-D latents → waveform via codebook quantization → S2A → codec
    
    Implements standard prompt + semantic design:
    - Semantic codes (from 8-D latents) provide CONTENT (what to say)
    - Acoustic prompt codes provide STYLE/VOICE (how to say it)
    
    The decoder only handles DECODING. For encoding (extracting acoustic codes),
    use Metis8dEncoder methods.
    """

    def __init__(self, cfg, audio_tokenizer: AudioTokenizer):
        # build S2A models exactly like Metis._build_s2a_model
        self.cfg = cfg
        self.audio_tok = audio_tokenizer
        self.device = audio_tokenizer.device
        self.semantic_codec = audio_tokenizer.semantic_codec
        self.codec_decoder = audio_tokenizer.codec_decoder
        self.s2a_model_1layer, self.s2a_model_full = self._build_s2a_models()
        self.s2a_model_1layer.eval()
        self.s2a_model_full.eval()

        # grab the learned codebook for 8D quantization
        self.codebook = self._find_8d_codebook().to(self.device)  # [K, 8]

    # ---------------- S2A setup ----------------

    def _build_s2a_models(self) -> Tuple[torch.nn.Module, torch.nn.Module]:
        """
        Build and load S2A (semantic-to-acoustic) models.
        
        Downloads model checkpoints from HuggingFace on first run.
        This may take 5-10 minutes depending on internet connection.
        
        Returns:
            Tuple of (s2a_model_1layer, s2a_model_full) models
        """
        logger.info("Building S2A models...")
        s2a_model_1layer = build_s2a_model(
            self.cfg.model.s2a_model.s2a_1layer, self.device
        )
        s2a_model_full = build_s2a_model(
            self.cfg.model.s2a_model.s2a_full, self.device
        )

        # Download checkpoints from HuggingFace
        logger.info("Downloading S2A model checkpoints (this may take 5-10 minutes)...")
        try:
            s2a_1layer_dir = snapshot_download(
                repo_id="amphion/MaskGCT",
                repo_type="model",
                local_dir="./models/tts/metis/ckpt",
                allow_patterns=["s2a_model/s2a_model_1layer/model.safetensors"],
            )
            logger.info("✓ Downloaded s2a_model_1layer")
        except Exception as e:
            raise RuntimeError(
                f"Failed to download s2a_model_1layer checkpoint.\n"
                f"Error: {str(e)}\n"
                f"Please check your internet connection and try again."
            ) from e
        
        try:
            s2a_full_dir = snapshot_download(
                repo_id="amphion/MaskGCT",
                repo_type="model",
                local_dir="./models/tts/metis/ckpt",
                allow_patterns=["s2a_model/s2a_model_full/model.safetensors"],
            )
            logger.info("✓ Downloaded s2a_model_full")
        except Exception as e:
            raise RuntimeError(
                f"Failed to download s2a_model_full checkpoint.\n"
                f"Error: {str(e)}\n"
                f"Please check your internet connection and try again."
            ) from e
        
        s2a_1layer_ckpt = os.path.join(
            s2a_1layer_dir, "s2a_model/s2a_model_1layer/model.safetensors"
        )
        s2a_full_ckpt = os.path.join(
            s2a_full_dir, "s2a_model/s2a_model_full/model.safetensors"
        )

        safetensors.torch.load_model(s2a_model_1layer, s2a_1layer_ckpt)
        safetensors.torch.load_model(s2a_model_full, s2a_full_ckpt)

        return s2a_model_1layer, s2a_model_full

    def _find_8d_codebook(self) -> torch.Tensor:
        """Get [K, 8] codebook from FactorizedVectorQuantize"""
        try:
            codebook = self.semantic_codec.quantizer.quantizers[0].codebook.weight
            return codebook.detach()
        except (AttributeError, IndexError) as e:
            raise RuntimeError(f"Could not find 8-D codebook: {e}")

    @torch.no_grad()
    def _latent8d_to_semantic_ids(self, z_8d: torch.Tensor) -> torch.Tensor:
        """Nearest-neighbor quantization: [1, T, 8] → [1, T] semantic codes"""
        if z_8d.dim() == 2:
            z_8d = z_8d.unsqueeze(0)
        
        z = z_8d.squeeze(0).to(self.device)  # [T, 8]
        dists = torch.cdist(z, self.codebook)  # [T, K]
        return dists.argmin(dim=1).unsqueeze(0).long()  # [1, T]

    @torch.no_grad()
    def _semantic2acoustic(
        self,
        semantic_code: torch.Tensor,
        prompt_acoustic_code: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Convert semantic codes to acoustic codes using S2A models.
        
        Implements standard prompt + semantic design:
        - semantic_code → embedded as 'cond' (content conditioning)
        - prompt_acoustic_code → concatenated as prefix 'prompt' (style/voice reference)
        
        The S2A models process: [PROMPT | TARGET] where:
        - PROMPT: Fixed acoustic codes (never masked, provides style)
        - TARGET: Generated acoustic codes (conditioned on semantic + prompt)
        
        IMPORTANT: The S2A reverse_diffusion calculates target_len = cond.shape[1] - prompt_len.
        For voice conversion, we want output_len = semantic_len (not semantic_len - prompt_len).
        So we EXTEND the semantic conditioning by prepending dummy tokens equal to prompt_len.
        This way: target_len = (semantic_len + prompt_len) - prompt_len = semantic_len ✓
        
        Args:
            semantic_code: [1, T_sem] semantic token IDs (content - what to say)
            prompt_acoustic_code: Optional [1, T_prompt, 12] acoustic codes (style - how to say it)
                                  If None, uses empty prompt [1, 0, 12]
                                  Should be extracted using Metis8dEncoder methods
        
        Returns:
            acoustic_code: [1, T_ac, 12] generated acoustic codes (same length as semantic_code)
        """
        semantic_len = semantic_code.shape[1]
        
        # Handle prompt: ensure correct shape and dtype
        if prompt_acoustic_code is None:
            prompt_acoustic_code = torch.zeros(1, 0, 12, device=self.device, dtype=torch.long)
        else:
            # Ensure correct shape: [1, T_prompt, 12]
            if prompt_acoustic_code.dim() == 2:
                prompt_acoustic_code = prompt_acoustic_code.unsqueeze(0)  # [1, T, 12]
            prompt_acoustic_code = prompt_acoustic_code.to(self.device).long()
            
            # Limit prompt length to reasonable maximum (1-2 seconds)
            # Longer prompts don't improve quality and slow down generation
            prompt_len = prompt_acoustic_code.shape[1]
            max_prompt_len = min(MAX_PROMPT_LENGTH_TOKENS, semantic_len)
            
            if prompt_len > max_prompt_len:
                prompt_acoustic_code = prompt_acoustic_code[:, :max_prompt_len, :]
                prompt_duration = max_prompt_len / SEMANTIC_RATE_HZ
                logger.info(
                    f"Prompt truncated from {prompt_len} to {max_prompt_len} tokens "
                    f"(~{prompt_duration:.1f}s). This is normal for voice conversion."
                )
        
        prompt_len = prompt_acoustic_code.shape[1]
        
        # CRITICAL FIX FOR VOICE CONVERSION:
        # The S2A model calculates: target_len = cond.shape[1] - prompt_len
        # For voice conversion, we want: output_len = semantic_len (full source length)
        # 
        # Solution: Extend semantic conditioning by prepending dummy tokens
        # so that: (semantic_len + prompt_len) - prompt_len = semantic_len
        #
        # The dummy tokens will be aligned with the acoustic prompt (which is fixed/not generated)
        # so their exact values don't matter - we use zeros (will be embedded as token 0)
        if prompt_len > 0:
            # Prepend dummy semantic tokens to match prompt length
            dummy_semantic = torch.zeros(1, prompt_len, device=self.device, dtype=semantic_code.dtype)
            semantic_code_extended = torch.cat([dummy_semantic, semantic_code], dim=1)
            logger.debug(
                f"Extended semantic from {semantic_len} to {semantic_code_extended.shape[1]} tokens "
                f"(prepended {prompt_len} dummy tokens for prompt alignment)"
            )
        else:
            semantic_code_extended = semantic_code
        
        # Stage 1: Coarse prediction (1 quantizer)
        # cond: semantic codes embedded → provides content
        # prompt: acoustic codes as prefix → provides style
        predict_1layer = self.s2a_model_1layer.reverse_diffusion(
            cond=self.s2a_model_1layer.cond_emb(semantic_code_extended),  # Extended conditioning
            prompt=prompt_acoustic_code,                                   # Style prefix
            temp=1.5,
            filter_thres=0.98,
            n_timesteps=[40],
            cfg=0,
            rescale_cfg=0.75,
        )

        # Stage 2: Refined prediction (12 quantizers)
        # Uses stage 1 output as gt_code for better quality
        return self.s2a_model_full.reverse_diffusion(
            cond=self.s2a_model_full.cond_emb(semantic_code_extended),  # Extended conditioning
            prompt=prompt_acoustic_code,                                 # Style prefix
            temp=1.5,
            filter_thres=0.98,
            n_timesteps=[40, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            cfg=2.5,
            rescale_cfg=0.75,
            gt_code=predict_1layer,  # Use coarse prediction as guidance
        )

    @torch.no_grad()
    def decode_from_z(
        self,
        z_8d: torch.Tensor,
        prompt_acoustic_code: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        """
        8-D latents [1, T, 8] → waveform [T_wav24]
        
        This follows the standard prompt + semantic design:
        - Semantic codes (from z_8d) provide CONTENT (what to say)
        - Acoustic prompt codes provide STYLE/VOICE (how to say it)
        
        The S2A models use:
        - cond: semantic codes embedded (global conditioning)
        - prompt: acoustic codes concatenated as prefix (style reference)
        
        Args:
            z_8d: [1, T, 8] 8-D continuous latents (content)
            prompt_acoustic_code: Optional [1, T_prompt, 12] acoustic codes to use as prompt.
                                 Provides voice/style reference. Should be extracted from
                                 reference audio using Metis8dEncoder.wav2acoustic() or
                                 Metis8dEncoder.encode_acoustic_from_path().
        
        Returns:
            waveform: [T_wav] numpy array at 24kHz
        """
        semantic_code = self._latent8d_to_semantic_ids(z_8d)
        acoustic_code = self._semantic2acoustic(semantic_code, prompt_acoustic_code)
        return self.audio_tok.code2wav(acoustic_code)

#!/usr/bin/env python3
"""
Dummy/test script for extracting 8-D semantic audio embeddings.

This is a lightweight version for testing with a small number of files (20-30).
Use this to verify the pipeline works before running the full extraction.
"""

import os
import json
import argparse
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
import torch
import numpy as np
from tqdm import tqdm

# Add project root to path
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.util import load_config
from models.tts.metis.audio_tokenizer import AudioTokenizer
from models.tts.metis.semantic_8d_wrappers import Metis8dEncoder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
AUDIO_BASE_DIR = "/data/common/versaface/audios"
JSON_OUTPUT_DIR = "/data/common/versaface/jsons"
SEMANTIC_EMBEDDINGS_JSON = "semantic_audio_embeddings_dummy.json"


def check_gpu_utilization(device_id: int = 0) -> Optional[Dict[str, str]]:
    """Check GPU utilization using nvidia-smi."""
    try:
        result = subprocess.run(
            [
                'nvidia-smi',
                f'--id={device_id}',
                '--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu',
                '--format=csv,noheader,nounits'
            ],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            parts = result.stdout.strip().split(', ')
            if len(parts) >= 7:
                return {
                    'index': parts[0],
                    'name': parts[1],
                    'gpu_util': parts[2] + '%',
                    'mem_util': parts[3] + '%',
                    'mem_used': parts[4] + ' MB',
                    'mem_total': parts[5] + ' MB',
                    'temp': parts[6] + '°C'
                }
    except Exception as e:
        logger.debug(f"nvidia-smi check failed: {e}")
    
    return None


def log_gpu_status(device_id: int = 0, context: str = ""):
    """Log current GPU utilization status."""
    gpu_stats = check_gpu_utilization(device_id)
    if gpu_stats:
        context_str = f" [{context}]" if context else ""
        logger.info(
            f"GPU Status{context_str}: "
            f"Util={gpu_stats['gpu_util']}, "
            f"Mem={gpu_stats['mem_util']} ({gpu_stats['mem_used']}/{gpu_stats['mem_total']}), "
            f"Temp={gpu_stats['temp']}"
        )


def find_sample_audio_files(audio_dir: str, num_files: int = 30) -> List[str]:
    """
    Find a small sample of .wav files for testing.
    
    Args:
        audio_dir: Base directory containing audio files
        num_files: Number of files to find (default: 30)
        
    Returns:
        List of absolute paths to audio files
    """
    audio_files = []
    audio_path = Path(audio_dir)
    
    if not audio_path.exists():
        logger.error(f"Audio directory does not exist: {audio_dir}")
        return audio_files
    
    # Find all .wav files recursively, but limit to num_files
    for wav_file in audio_path.rglob("*.wav"):
        audio_files.append(str(wav_file.absolute()))
        if len(audio_files) >= num_files:
            break
    
    logger.info(f"Found {len(audio_files)} sample audio files in {audio_dir}")
    return sorted(audio_files)


def extract_embeddings_iterative(
    encoder: Metis8dEncoder,
    audio_files: List[str],
    batch_size: int = 10,
    device: str = "cuda"
) -> Dict[str, List[float]]:
    """Extract 8-D semantic embeddings from audio files iteratively in batches."""
    embeddings_dict = {}
    failed_files = []
    
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available!")
    
    if device.startswith("cuda"):
        device_id = int(device.split(":")[1]) if ":" in device else 0
        logger.info(f"Using GPU: {torch.cuda.get_device_name(device_id)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(device_id).total_memory / 1e9:.2f} GB")
    else:
        device_id = 0
    
    total_files = len(audio_files)
    num_batches = (total_files + batch_size - 1) // batch_size
    
    logger.info(f"Processing {total_files} audio files in {num_batches} batches (batch_size={batch_size})")
    logger.info(f"Device: {device}")
    
    if device.startswith("cuda"):
        log_gpu_status(device_id, "Before processing")
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_files)
        batch_files = audio_files[start_idx:end_idx]
        
        logger.info(f"\n--- Batch {batch_idx + 1}/{num_batches} (files {start_idx+1}-{end_idx}) ---")
        
        for audio_path in tqdm(batch_files, desc=f"Batch {batch_idx+1}/{num_batches}"):
            try:
                # Extract 8-D semantic latents
                _, z_8d = encoder.encode_from_path(audio_path)
                
                # Convert to numpy and then to list
                z_8d_np = z_8d.squeeze(0).cpu().numpy()  # (T, 8)
                embedding_list = z_8d_np.tolist()
                
                # Use relative path from AUDIO_BASE_DIR as key
                rel_path = os.path.relpath(audio_path, AUDIO_BASE_DIR)
                embeddings_dict[rel_path] = embedding_list
                
                # Clear GPU memory
                del z_8d, z_8d_np
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
                
            except Exception as e:
                logger.error(f"Failed to process {audio_path}: {str(e)}")
                failed_files.append(audio_path)
                continue
        
        logger.info(f"Batch {batch_idx + 1} complete: {len(embeddings_dict)}/{total_files} files processed")
        
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
            log_gpu_status(device_id, f"After batch {batch_idx + 1}")
    
    if failed_files:
        logger.warning(f"Failed to process {len(failed_files)} files")
        if len(failed_files) <= 20:
            logger.warning(f"Failed files: {failed_files}")
        else:
            logger.warning(f"Failed files (first 20): {failed_files[:20]}")
    
    return embeddings_dict


def save_embeddings_json(
    embeddings_dict: Dict[str, List[List[float]]],
    output_path: str
):
    """Save embeddings dictionary to JSON file."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving {len(embeddings_dict)} embeddings to {output_path}")
    
    with open(output_path, 'w') as f:
        json.dump(embeddings_dict, f, indent=2)
    
    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    logger.info(f"Saved embeddings JSON ({file_size_mb:.2f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description="Dummy/test script: Extract 8-D semantic audio embeddings (small sample)"
    )
    parser.add_argument(
        "--audio-dir",
        type=str,
        default=AUDIO_BASE_DIR,
        help=f"Directory containing audio files (default: {AUDIO_BASE_DIR})"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=JSON_OUTPUT_DIR,
        help=f"Directory to save JSON output (default: {JSON_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=SEMANTIC_EMBEDDINGS_JSON,
        help=f"Output JSON filename (default: {SEMANTIC_EMBEDDINGS_JSON})"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on (default: cuda:0)"
    )
    parser.add_argument(
        "--num-files",
        type=int,
        default=30,
        help="Number of files to process (default: 30)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of files per batch (default: 10)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="models/tts/metis/config/base.json",
        help="Path to config file (default: models/tts/metis/config/base.json)"
    )
    
    args = parser.parse_args()
    
    # Validate GPU
    if not torch.cuda.is_available():
        logger.error("CUDA/GPU is not available! This script requires GPU.")
        raise RuntimeError("GPU required but not available.")
    
    device = args.device if args.device.startswith("cuda") else "cuda:0"
    
    if device.startswith("cuda"):
        device_id = int(device.split(":")[1]) if ":" in device else 0
        torch.cuda.set_device(device_id)
        logger.info(f"Using GPU: {torch.cuda.get_device_name(device_id)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(device_id).total_memory / 1e9:.2f} GB")
        log_gpu_status(device_id, "Initial")
    
    # Load config
    config_path = args.config
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    logger.info(f"Loading configuration from {config_path}...")
    cfg = load_config(config_path)
    
    # Initialize models
    logger.info("\n" + "=" * 70)
    logger.info("INITIALIZING MODELS ON GPU (DUMMY TEST)")
    logger.info("=" * 70)
    
    try:
        logger.info("Loading AudioTokenizer...")
        audio_tokenizer = AudioTokenizer(cfg, device)
        logger.info("✓ AudioTokenizer loaded")
        
        logger.info("Loading Metis8dEncoder...")
        encoder = Metis8dEncoder(audio_tokenizer)
        logger.info("✓ Encoder loaded")
        
        if device.startswith("cuda"):
            encoder_device = next(encoder.semantic_model.parameters()).device
            logger.info(f"Encoder device: {encoder_device}")
            log_gpu_status(device_id, "After model loading")
        
    except Exception as e:
        logger.error(f"Failed to initialize models: {str(e)}")
        raise
    
    # Find sample audio files
    logger.info(f"\nFinding {args.num_files} sample audio files...")
    audio_files = find_sample_audio_files(args.audio_dir, args.num_files)
    
    if not audio_files:
        logger.error("No audio files found!")
        return
    
    logger.info(f"Will process {len(audio_files)} files")
    
    # Extract embeddings
    embeddings_dict = extract_embeddings_iterative(
        encoder,
        audio_files,
        batch_size=args.batch_size,
        device=device
    )
    
    # Save to JSON
    output_path = os.path.join(args.output_dir, args.output_file)
    save_embeddings_json(embeddings_dict, output_path)
    
    if device.startswith("cuda"):
        log_gpu_status(device_id, "Final")
    
    logger.info("=" * 70)
    logger.info("✅ DUMMY EXTRACTION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Processed: {len(embeddings_dict)} files")
    logger.info(f"Output: {output_path}")
    logger.info("\n💡 To view the embeddings, see instructions below:")
    logger.info("   python3 -c \"import json; data=json.load(open('{}')); print(list(data.keys())[:5])\"".format(output_path))
    logger.info("=" * 70)


if __name__ == "__main__":
    main()


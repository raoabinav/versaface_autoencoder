#!/usr/bin/env python3
"""
Extract 8-D semantic audio embeddings from versaface audio files.

This script processes all audio files in /data/common/versaface/audios/
and extracts 8-D semantic embeddings using the Metis encoder.
The embeddings are saved as JSON files in /data/common/versaface/jsons/
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
SEMANTIC_EMBEDDINGS_JSON = "semantic_audio_embeddings.json"


def check_gpu_utilization(device_id: int = 0) -> Optional[Dict[str, str]]:
    """
    Check GPU utilization using nvidia-smi.
    
    Args:
        device_id: GPU device ID (default: 0)
        
    Returns:
        Dictionary with GPU stats, or None if nvidia-smi fails
    """
    try:
        # Run nvidia-smi query
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
            # Parse output: index,name,gpu_util,mem_util,mem_used,mem_total,temp
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
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError) as e:
        logger.debug(f"nvidia-smi check failed: {e}")
    
    return None


def log_gpu_status(device_id: int = 0, context: str = ""):
    """
    Log current GPU utilization status.
    
    Args:
        device_id: GPU device ID
        context: Context string to include in log message
    """
    gpu_stats = check_gpu_utilization(device_id)
    if gpu_stats:
        context_str = f" [{context}]" if context else ""
        logger.info(
            f"GPU Status{context_str}: "
            f"Util={gpu_stats['gpu_util']}, "
            f"Mem={gpu_stats['mem_util']} ({gpu_stats['mem_used']}/{gpu_stats['mem_total']}), "
            f"Temp={gpu_stats['temp']}"
        )
    else:
        logger.debug("Could not retrieve GPU stats (nvidia-smi may not be available)")


def find_all_audio_files(audio_dir: str) -> List[str]:
    """
    Find all .wav files in the audio directory.
    
    Args:
        audio_dir: Base directory containing audio files
        
    Returns:
        List of absolute paths to audio files
    """
    audio_files = []
    audio_path = Path(audio_dir)
    
    if not audio_path.exists():
        logger.error(f"Audio directory does not exist: {audio_dir}")
        return audio_files
    
    # Find all .wav files recursively
    for wav_file in audio_path.rglob("*.wav"):
        audio_files.append(str(wav_file.absolute()))
    
    logger.info(f"Found {len(audio_files)} audio files in {audio_dir}")
    return sorted(audio_files)


def extract_embeddings_iterative(
    encoder: Metis8dEncoder,
    audio_files: List[str],
    batch_size: int = 100,
    device: str = "cuda"
) -> Dict[str, List[float]]:
    """
    Extract 8-D semantic embeddings from audio files iteratively in batches.
    
    Args:
        encoder: Metis8dEncoder instance
        audio_files: List of audio file paths
        batch_size: Number of files to process before saving intermediate results
        device: Device to run inference on (should be GPU)
        
    Returns:
        Dictionary mapping audio file paths to 8-D embedding arrays
    """
    embeddings_dict = {}
    failed_files = []
    
    # Ensure GPU is being used
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available!")
    
    if device.startswith("cuda"):
        device_id = int(device.split(":")[1]) if ":" in device else 0
        logger.info(f"Using GPU: {torch.cuda.get_device_name(device_id)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(device_id).total_memory / 1e9:.2f} GB")
    
    total_files = len(audio_files)
    num_batches = (total_files + batch_size - 1) // batch_size
    
    logger.info(f"Processing {total_files} audio files in {num_batches} batches (batch_size={batch_size})")
    logger.info(f"Device: {device}")
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_files)
        batch_files = audio_files[start_idx:end_idx]
        
        logger.info(f"\n--- Batch {batch_idx + 1}/{num_batches} (files {start_idx+1}-{end_idx}) ---")
        
        for audio_path in tqdm(batch_files, desc=f"Batch {batch_idx+1}/{num_batches}"):
            try:
                # Note: Device is already set during initialization
                # No need to set again here
                
                # Extract 8-D semantic latents (this should use GPU internally)
                _, z_8d = encoder.encode_from_path(audio_path)
                
                # Move to CPU for numpy conversion (keep on GPU until needed)
                z_8d_np = z_8d.squeeze(0).cpu().numpy()  # (T, 8)
                
                # Store as list of lists: [[d1, d2, ..., d8], ...] for each time step
                embedding_list = z_8d_np.tolist()
                
                # Use relative path from AUDIO_BASE_DIR as key
                rel_path = os.path.relpath(audio_path, AUDIO_BASE_DIR)
                embeddings_dict[rel_path] = embedding_list
                
                # Clear GPU memory immediately
                del z_8d, z_8d_np
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
                
            except Exception as e:
                logger.error(f"Failed to process {audio_path}: {str(e)}")
                failed_files.append(audio_path)
                continue
        
        # Log batch progress
        logger.info(f"Batch {batch_idx + 1} complete: {len(embeddings_dict)}/{total_files} files processed")
        
        # Clear GPU cache after each batch
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
            allocated_gb = torch.cuda.memory_allocated(device_id) / 1e9
            logger.debug(f"GPU memory cleared. Allocated: {allocated_gb:.2f} GB")
            
            # Log GPU utilization after batch
            log_gpu_status(device_id, f"After batch {batch_idx + 1}")
    
    if failed_files:
        logger.warning(f"Failed to process {len(failed_files)} files")
        if len(failed_files) <= 20:
            logger.warning(f"Failed files: {failed_files}")
        else:
            logger.warning(f"Failed files (first 20): {failed_files[:20]}")
            logger.warning(f"... and {len(failed_files) - 20} more")
    
    return embeddings_dict


def save_embeddings_json(
    embeddings_dict: Dict[str, List[List[float]]],
    output_path: str
):
    """
    Save embeddings dictionary to JSON file.
    
    Args:
        embeddings_dict: Dictionary mapping file paths to embeddings
        output_path: Path to save JSON file
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving {len(embeddings_dict)} embeddings to {output_path}")
    
    with open(output_path, 'w') as f:
        json.dump(embeddings_dict, f, indent=2)
    
    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    logger.info(f"Saved embeddings JSON ({file_size_mb:.2f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description="Extract 8-D semantic audio embeddings from versaface audio files"
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
        help="Device to run inference on (default: cuda:0 if available, else cpu). Use cuda:0, cuda:1, etc."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of files to process in each batch before clearing memory (default: 100)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="models/tts/metis/config/base.json",
        help="Path to config file (default: models/tts/metis/config/base.json)"
    )
    parser.add_argument(
        "--part",
        type=str,
        default=None,
        help="Process only specific part directory (e.g., 'part_003'). If None, process all."
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of files to process (for testing). If None, process all."
    )
    
    args = parser.parse_args()
    
    # Validate and setup device - FORCE GPU
    if not torch.cuda.is_available():
        logger.error("CUDA/GPU is not available! This script requires GPU.")
        raise RuntimeError("GPU required but not available. Please use a machine with CUDA support.")
    
    # Use GPU
    device = args.device if args.device.startswith("cuda") else "cuda:0"
    
    # Set default CUDA device
    if device.startswith("cuda"):
        device_id = int(device.split(":")[1]) if ":" in device else 0
        torch.cuda.set_device(device_id)
        logger.info(f"Using GPU: {torch.cuda.get_device_name(device_id)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(device_id).total_memory / 1e9:.2f} GB")
        
        # Log initial GPU status
        log_gpu_status(device_id, "Initial")
    
    # Load config
    config_path = args.config
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    logger.info(f"Loading configuration from {config_path}...")
    cfg = load_config(config_path)
    
    # Initialize models - ensure they're on GPU
    logger.info("\n" + "=" * 70)
    logger.info("INITIALIZING MODELS ON GPU")
    logger.info("=" * 70)
    logger.info("(This may take a few minutes on first run as models are downloaded)")
    
    try:
        logger.info("Loading AudioTokenizer...")
        audio_tokenizer = AudioTokenizer(cfg, device)
        logger.info("✓ AudioTokenizer loaded on GPU")
        
        logger.info("Loading Metis8dEncoder...")
        encoder = Metis8dEncoder(audio_tokenizer)
        logger.info("✓ Encoder loaded on GPU")
        
        # Verify models are on GPU
        if device.startswith("cuda"):
            # Check if encoder components are on GPU
            encoder_device = next(encoder.semantic_model.parameters()).device
            logger.info(f"Encoder device: {encoder_device}")
            if str(encoder_device) != device:
                logger.warning(f"Encoder may not be on expected device {device}, found {encoder_device}")
            
            # Log GPU status after model loading
            log_gpu_status(device_id, "After model loading")
        
    except Exception as e:
        logger.error(f"Failed to initialize models: {str(e)}")
        raise
    
    # Find audio files
    audio_dir = args.audio_dir
    if args.part:
        audio_dir = os.path.join(args.audio_dir, args.part)
        logger.info(f"Processing only: {args.part}")
    
    audio_files = find_all_audio_files(audio_dir)
    
    if not audio_files:
        logger.error("No audio files found!")
        return
    
    # Limit files if specified
    if args.max_files:
        audio_files = audio_files[:args.max_files]
        logger.info(f"Limited to {len(audio_files)} files for testing")
    
    # Extract embeddings iteratively in batches
    embeddings_dict = extract_embeddings_iterative(
        encoder,
        audio_files,
        batch_size=args.batch_size,
        device=device
    )
    
    # Save to JSON
    output_path = os.path.join(args.output_dir, args.output_file)
    save_embeddings_json(embeddings_dict, output_path)
    
    # Log final GPU status
    if device.startswith("cuda"):
        device_id = int(device.split(":")[1]) if ":" in device else 0
        log_gpu_status(device_id, "Final")
    
    logger.info("=" * 70)
    logger.info("✅ EMBEDDING EXTRACTION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Processed: {len(embeddings_dict)} files")
    logger.info(f"Output: {output_path}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()


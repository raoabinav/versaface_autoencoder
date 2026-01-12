#!/usr/bin/env python3
"""
Extract 8-D semantic audio embeddings from audio files.

This script processes audio files and:
1. Saves embeddings as .npy files (mirroring the folder structure: .wav → .npy)
2. Tracks successfully processed paths in a JSON list for resumability

Each .npy file contains a float32 array of shape [T, 8] where:
- T = number of time frames (approximately 50 per second of audio)
- 8 = dimension of the semantic latent space

Usage:
    # Process all files
    python extract_audio_embeddings.py

    # Process specific part
    python extract_audio_embeddings.py --part part_003

    # Test with limited files
    python extract_audio_embeddings.py --max-files 100

    # Custom directories
    python extract_audio_embeddings.py \\
        --audio-dir /path/to/audios \\
        --embedding-dir /path/to/output/embeddings \\
        --json-dir /path/to/output/jsons
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import List, Set
import numpy as np
import torch
from tqdm import tqdm

# Add project root to path
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
EMBEDDING_BASE_DIR = "/data/common/versaface/audio_embeddings"
JSON_OUTPUT_DIR = "/data/common/versaface/jsons"
SUCCESS_JSON = "success_total_audio.json"


def find_all_audio_files(audio_dir: str, part: str = None) -> List[str]:
    """
    Find all .wav files in the audio directory.
    
    Args:
        audio_dir: Base directory containing audio files
        part: Optional part filter (e.g., 'part_003')
        
    Returns:
        List of absolute paths to audio files
    """
    audio_files = []
    search_dir = Path(audio_dir)
    
    if part:
        search_dir = search_dir / part
        if not search_dir.exists():
            logger.error(f"Part directory does not exist: {search_dir}")
            return audio_files
    
    if not search_dir.exists():
        logger.error(f"Audio directory does not exist: {search_dir}")
        return audio_files
    
    # Find all .wav files recursively
    for wav_file in search_dir.rglob("*.wav"):
        audio_files.append(str(wav_file.absolute()))
    
    logger.info(f"Found {len(audio_files)} audio files in {search_dir}")
    return sorted(audio_files)


def get_embedding_path(audio_path: str, audio_base_dir: str, embedding_base_dir: str) -> str:
    """
    Convert audio path to embedding path.
    
    /data/common/versaface/audios/part_XXX/XX/XX/hash.wav
    → /data/common/versaface/audio_embeddings/part_XXX/XX/XX/hash.npy
    """
    # Get relative path from audio base dir
    rel_path = os.path.relpath(audio_path, audio_base_dir)
    # Change extension to .npy
    rel_path_npy = os.path.splitext(rel_path)[0] + ".npy"
    # Join with embedding base dir
    return os.path.join(embedding_base_dir, rel_path_npy)


def load_existing_success(json_path: str) -> Set[str]:
    """Load existing successful paths from JSON."""
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                return set(json.load(f))
        except Exception as e:
            logger.warning(f"Could not load existing success JSON: {e}")
    return set()


def save_success_json(success_paths: List[str], json_path: str):
    """Save successful paths to JSON."""
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump(sorted(success_paths), f, indent=4)
    logger.info(f"Saved {len(success_paths)} successful paths to {json_path}")


def process_audio_files(
    encoder: Metis8dEncoder,
    audio_files: List[str],
    existing_success: Set[str],
    audio_base_dir: str,
    embedding_base_dir: str,
    json_output_dir: str,
    batch_size: int = 100,
    device: str = "cuda"
) -> List[str]:
    """
    Process audio files and save embeddings as NPY files.
    
    Args:
        encoder: Metis8dEncoder instance
        audio_files: List of audio file paths
        existing_success: Set of already processed paths
        audio_base_dir: Base directory for audio files
        embedding_base_dir: Base directory for embedding files
        json_output_dir: Directory for JSON output
        batch_size: Files to process before saving progress
        device: Device for inference
        
    Returns:
        List of successfully processed audio paths
    """
    success_paths = list(existing_success)
    failed_files = []
    skipped = 0
    
    # Filter out already processed files
    files_to_process = []
    for audio_path in audio_files:
        if audio_path in existing_success:
            skipped += 1
            continue
        # Also check if NPY file exists
        npy_path = get_embedding_path(audio_path, audio_base_dir, embedding_base_dir)
        if os.path.exists(npy_path):
            success_paths.append(audio_path)
            skipped += 1
            continue
        files_to_process.append(audio_path)
    
    if skipped > 0:
        logger.info(f"Skipping {skipped} already processed files")
    
    if not files_to_process:
        logger.info("No new files to process!")
        return success_paths
    
    logger.info(f"Processing {len(files_to_process)} audio files")
    
    # Process files
    for i, audio_path in enumerate(tqdm(files_to_process, desc="Extracting embeddings")):
        try:
            # Extract 8-D embeddings
            _, z_8d = encoder.encode_from_path(audio_path)
            
            # Convert to numpy: [1, T, 8] → [T, 8]
            z_8d_np = z_8d.squeeze(0).cpu().numpy()
            
            # Get output path and create directories
            npy_path = get_embedding_path(audio_path, audio_base_dir, embedding_base_dir)
            os.makedirs(os.path.dirname(npy_path), exist_ok=True)
            
            # Save as NPY
            np.save(npy_path, z_8d_np)
            
            # Track success
            success_paths.append(audio_path)
            
            # Clear GPU memory
            del z_8d, z_8d_np
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"Failed to process {audio_path}: {str(e)}")
            failed_files.append(audio_path)
            continue
        
        # Save progress periodically
        if (i + 1) % batch_size == 0:
            save_success_json(success_paths, os.path.join(json_output_dir, SUCCESS_JSON))
            logger.info(f"Progress: {i + 1}/{len(files_to_process)} files processed")
    
    # Log failures
    if failed_files:
        logger.warning(f"Failed to process {len(failed_files)} files")
        if len(failed_files) <= 20:
            for f in failed_files:
                logger.warning(f"  - {f}")
    
    return success_paths


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
        "--embedding-dir",
        type=str,
        default=EMBEDDING_BASE_DIR,
        help=f"Directory to save NPY embeddings (default: {EMBEDDING_BASE_DIR})"
    )
    parser.add_argument(
        "--json-dir",
        type=str,
        default=JSON_OUTPUT_DIR,
        help=f"Directory to save success JSON (default: {JSON_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device for inference (default: cuda:0)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Save progress every N files (default: 100)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="models/tts/metis/config/base.json",
        help="Path to config file"
    )
    parser.add_argument(
        "--part",
        type=str,
        default=None,
        help="Process only specific part (e.g., 'part_003')"
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum files to process (for testing)"
    )
    
    args = parser.parse_args()
    
    # Use args for paths (don't modify globals)
    audio_base_dir = args.audio_dir
    embedding_base_dir = args.embedding_dir
    json_output_dir = args.json_dir
    
    # Validate GPU
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        logger.error("CUDA not available! Use --device cpu or run on GPU machine.")
        sys.exit(1)
    
    device = args.device
    logger.info(f"Using device: {device}")
    
    # Load config
    if not os.path.exists(args.config):
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)
    
    logger.info(f"Loading configuration from {args.config}")
    cfg = load_config(args.config)
    
    # Initialize models
    logger.info("=" * 70)
    logger.info("INITIALIZING MODELS")
    logger.info("=" * 70)
    
    logger.info("Loading AudioTokenizer...")
    audio_tokenizer = AudioTokenizer(cfg, device)
    logger.info("✓ AudioTokenizer loaded")
    
    logger.info("Loading Metis8dEncoder...")
    encoder = Metis8dEncoder(audio_tokenizer)
    logger.info("✓ Encoder loaded")
    
    # Find audio files
    audio_files = find_all_audio_files(audio_base_dir, args.part)
    
    if not audio_files:
        logger.error("No audio files found!")
        sys.exit(1)
    
    # Limit files if specified
    if args.max_files:
        audio_files = audio_files[:args.max_files]
        logger.info(f"Limited to {len(audio_files)} files for testing")
    
    # Load existing success
    success_json_path = os.path.join(json_output_dir, SUCCESS_JSON)
    existing_success = load_existing_success(success_json_path)
    logger.info(f"Found {len(existing_success)} already processed files")
    
    # Process files
    logger.info("=" * 70)
    logger.info("EXTRACTING EMBEDDINGS")
    logger.info("=" * 70)
    
    success_paths = process_audio_files(
        encoder,
        audio_files,
        existing_success,
        audio_base_dir=audio_base_dir,
        embedding_base_dir=embedding_base_dir,
        json_output_dir=json_output_dir,
        batch_size=args.batch_size,
        device=device
    )
    
    # Save final success JSON
    save_success_json(success_paths, success_json_path)
    
    # Summary
    logger.info("=" * 70)
    logger.info("✅ EXTRACTION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total successful: {len(success_paths)}")
    logger.info(f"Embeddings saved to: {embedding_base_dir}")
    logger.info(f"Success JSON: {success_json_path}")


if __name__ == "__main__":
    main()

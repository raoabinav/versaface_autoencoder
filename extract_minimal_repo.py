#!/usr/bin/env python3
"""
Script to extract minimal repo for semantic 8-D wrappers.
This will identify and optionally delete non-essential files.
"""

import os
import shutil
from pathlib import Path

# Essential files to KEEP
ESSENTIAL_FILES = {
    # Core semantic wrapper
    "models/tts/metis/semantic_8d_wrappers.py",
    "models/tts/metis/audio_tokenizer.py",
    "models/tts/metis/config/base.json",
    
    # MaskGCT dependencies
    "models/tts/maskgct/maskgct_utils.py",
    "models/tts/maskgct/maskgct_s2a.py",
    "models/tts/maskgct/llama_nar.py",
    "models/tts/maskgct/ckpt/wav2vec2bert_stats.pt",
    
    # Codec dependencies
    "models/codec/kmeans/repcodec_model.py",
    "models/codec/kmeans/vocos.py",
    "models/codec/amphion_codec/codec.py",
    "models/codec/amphion_codec/vocos.py",
    "models/codec/amphion_codec/quantize/__init__.py",
    "models/codec/amphion_codec/quantize/residual_vq.py",
    "models/codec/amphion_codec/quantize/factorized_vector_quantize.py",
    "models/codec/amphion_codec/quantize/lookup_free_quantize.py",
    "models/codec/amphion_codec/quantize/vector_quantize.py",
    
    # Utils
    "utils/util.py",
    "utils/hparam.py",
    
    # Keep __init__.py files for imports to work
    "models/__init__.py",
    "models/tts/__init__.py",
    "models/tts/metis/__init__.py",
    "models/tts/maskgct/__init__.py",
    "models/codec/__init__.py",
    "models/codec/kmeans/__init__.py",
    "models/codec/amphion_codec/__init__.py",
    "models/codec/amphion_codec/quantize/__init__.py",
    "utils/__init__.py",
}

# Directories to KEEP (but may clean files inside)
KEEP_DIRECTORIES = {
    "models/tts/metis",
    "models/tts/maskgct",
    "models/codec/kmeans",
    "models/codec/amphion_codec",
    "utils",
}

# Directories to DELETE entirely
DELETE_DIRECTORIES = {
    "bins",
    "egs",
    "evaluation",
    "modules",
    "optimizer",
    "preprocessors",
    "processors",
    "schedulers",
    "text",
    "visualization",
    "pretrained",
    "imgs",
    "config",  # Keep only metis/config/base.json
    "models/tts/vits",
    "models/tts/valle",
    "models/tts/naturalspeech2",
    "models/tts/jets",
    "models/tts/fastspeech2",
    "models/tts/debatts",
    "models/svc",
    "models/vc",
    "models/tta",
    "models/codec/ns3_codec",
    "models/codec/dualcodec",
    "models/codec/amphion_codec/loss.py",  # Not needed for inference
}

def find_all_files(root_dir="."):
    """Find all Python, JSON, and other relevant files."""
    all_files = []
    for root, dirs, files in os.walk(root_dir):
        # Skip hidden directories and common ignore patterns
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules']]
        
        for file in files:
            if file.startswith('.'):
                continue
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, root_dir)
            all_files.append(rel_path)
    return all_files

def should_keep_file(file_path):
    """Check if file should be kept."""
    # Normalize path separators
    file_path = file_path.replace('\\', '/')
    
    # Check exact matches
    if file_path in ESSENTIAL_FILES:
        return True
    
    # Check if in keep directories
    for keep_dir in KEEP_DIRECTORIES:
        if file_path.startswith(keep_dir + '/'):
            # Check if it's a Python file in these directories
            if file_path.endswith('.py') or file_path.endswith('.json') or file_path.endswith('.pt'):
                # Allow it for now, we'll filter more specifically
                return True
    
    # Keep README and LICENSE files
    if file_path.endswith('README.md') or file_path == 'LICENSE':
        return True
    
    # Keep requirements files
    if 'requirements.txt' in file_path or 'setup.py' in file_path:
        return True
    
    return False

def should_delete_directory(dir_path):
    """Check if directory should be deleted entirely."""
    dir_path = dir_path.replace('\\', '/')
    for delete_dir in DELETE_DIRECTORIES:
        if dir_path.startswith(delete_dir):
            return True
    return False

def main():
    print("=" * 70)
    print("MINIMAL REPO EXTRACTION - FILE ANALYSIS")
    print("=" * 70)
    
    root_dir = "."
    all_files = find_all_files(root_dir)
    
    files_to_keep = []
    files_to_delete = []
    
    for file_path in all_files:
        if should_keep_file(file_path):
            files_to_keep.append(file_path)
        else:
            files_to_delete.append(file_path)
    
    print(f"\n[STATISTICS]")
    print(f"   Total files: {len(all_files)}")
    print(f"   Files to KEEP: {len(files_to_keep)}")
    print(f"   Files to DELETE: {len(files_to_delete)}")
    
    print(f"\n[FILES TO KEEP] ({len(files_to_keep)}):")
    for f in sorted(files_to_keep)[:20]:  # Show first 20
        print(f"   {f}")
    if len(files_to_keep) > 20:
        print(f"   ... and {len(files_to_keep) - 20} more")
    
    print(f"\n[FILES TO DELETE] ({len(files_to_delete)}):")
    for f in sorted(files_to_delete)[:30]:  # Show first 30
        print(f"   {f}")
    if len(files_to_delete) > 30:
        print(f"   ... and {len(files_to_delete) - 30} more")
    
    # Ask for confirmation
    print("\n" + "=" * 70)
    response = input("Delete non-essential files? (yes/no): ").strip().lower()
    
    if response == 'yes':
        print("\n[DELETING FILES...]")
        deleted_count = 0
        for file_path in files_to_delete:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    deleted_count += 1
            except Exception as e:
                print(f"   WARNING: Could not delete {file_path}: {e}")
        
        # Delete empty directories
        print("\n[CLEANING UP EMPTY DIRECTORIES...]")
        for root, dirs, files in os.walk(root_dir, topdown=False):
            try:
                if not dirs and not files and root != root_dir:
                    os.rmdir(root)
            except:
                pass
        
        print(f"\n[SUCCESS] Deleted {deleted_count} files")
    else:
        print("\n[CANCELLED] Deletion cancelled")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()


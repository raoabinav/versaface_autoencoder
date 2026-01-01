#!/usr/bin/env python3
"""
Delete non-essential files for semantic 8-D wrapper minimal repo.
"""

import os
import shutil

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
    
    # Keep __init__.py files
    "models/__init__.py",
    "models/tts/__init__.py",
    "models/tts/metis/__init__.py",
    "models/tts/maskgct/__init__.py",
    "models/codec/__init__.py",
    "models/codec/kmeans/__init__.py",
    "models/codec/amphion_codec/__init__.py",
    "models/codec/amphion_codec/quantize/__init__.py",
    "utils/__init__.py",
    
    # Keep root files
    "LICENSE",
    "README.md",
}

# Directories to DELETE entirely
DELETE_DIRS = [
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
    "maskgct_env",
    "ideas",  # Keep ideas folder for now, user might want it
]

# Files in keep directories that should be deleted
DELETE_PATTERNS = [
    "models/tts/metis/metis.py",
    "models/tts/metis/metis_model.py",
    "models/tts/metis/metis_infer_vc.py",
    "models/tts/metis/notebook.ipynb",  # Keep this? User might want it
    "models/tts/maskgct/g2p/",
    "models/tts/maskgct/maskgct_t2s.py",
    "models/tts/maskgct/gradio_demo.py",
    "models/tts/maskgct/maskgct_inference.py",
    "models/tts/maskgct/maskgct_demo.ipynb",
    "models/codec/amphion_codec/loss.py",
]

def normalize_path(path):
    """Normalize path separators."""
    return path.replace('\\', '/')

def should_keep(file_path):
    """Check if file should be kept."""
    file_path = normalize_path(file_path)
    
    # Check exact matches
    if file_path in ESSENTIAL_FILES:
        return True
    
    # Check if in delete patterns
    for pattern in DELETE_PATTERNS:
        if file_path.startswith(pattern):
            return False
    
    # Keep Python files in essential directories (but filter later)
    essential_dirs = [
        "models/tts/metis/",
        "models/tts/maskgct/",
        "models/codec/kmeans/",
        "models/codec/amphion_codec/",
        "utils/",
    ]
    
    for dir_path in essential_dirs:
        if file_path.startswith(dir_path):
            # Only keep .py, .json, .pt files
            if file_path.endswith(('.py', '.json', '.pt', '.txt')):
                # But exclude specific files
                if any(exclude in file_path for exclude in [
                    'metis.py', 'metis_model.py', 'metis_infer_vc.py',
                    'maskgct_t2s.py', 'gradio_demo.py', 'maskgct_inference.py',
                    'loss.py', 'g2p'
                ]):
                    return False
                return True
    
    return False

def delete_directory(dir_path):
    """Delete directory and all contents."""
    if os.path.exists(dir_path):
        try:
            shutil.rmtree(dir_path)
            print(f"   Deleted directory: {dir_path}")
            return True
        except Exception as e:
            print(f"   ERROR deleting {dir_path}: {e}")
            return False
    return False

def main():
    print("=" * 70)
    print("DELETING NON-ESSENTIAL FILES")
    print("=" * 70)
    
    # First, delete entire directories
    print("\n[STEP 1] Deleting entire directories...")
    for dir_path in DELETE_DIRS:
        delete_directory(dir_path)
    
    # Delete config directory except metis/config/base.json
    print("\n[STEP 2] Cleaning config directory...")
    config_dir = "config"
    if os.path.exists(config_dir):
        for item in os.listdir(config_dir):
            item_path = os.path.join(config_dir, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
                print(f"   Deleted: {item_path}")
            elif item != "base.json":  # Keep base.json if it's here, but it should be in metis/config
                os.remove(item_path)
                print(f"   Deleted: {item_path}")
    
    # Now delete individual files
    print("\n[STEP 3] Deleting non-essential files...")
    deleted_count = 0
    
    # Walk through remaining directories
    for root, dirs, files in os.walk("."):
        # Skip hidden and cache directories
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules']]
        
        for file in files:
            if file.startswith('.'):
                continue
            
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, ".")
            
            if not should_keep(rel_path):
                try:
                    os.remove(file_path)
                    deleted_count += 1
                    if deleted_count % 100 == 0:
                        print(f"   Deleted {deleted_count} files...")
                except Exception as e:
                    print(f"   ERROR: Could not delete {rel_path}: {e}")
    
    print(f"\n[SUCCESS] Deleted {deleted_count} files")
    
    # Clean up empty directories
    print("\n[STEP 4] Cleaning up empty directories...")
    for root, dirs, files in os.walk(".", topdown=False):
        try:
            if root != "." and not os.listdir(root):
                os.rmdir(root)
        except:
            pass
    
    print("\n[COMPLETE] Cleanup finished!")
    print("=" * 70)

if __name__ == "__main__":
    main()


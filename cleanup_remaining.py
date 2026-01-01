#!/usr/bin/env python3
"""Clean up remaining non-essential files."""

import os
import shutil

# Files to delete
FILES_TO_DELETE = [
    # Metis inference files (not needed for semantic wrapper)
    "models/tts/metis/metis_infer_omni.py",
    "models/tts/metis/metis_infer_se.py",
    "models/tts/metis/metis_infer_tse.py",
    "models/tts/metis/metis_infer_tts.py",
    
    # Extra config files (only need base.json)
    "models/tts/metis/config/ft.json",
    "models/tts/metis/config/l2s.json",
    "models/tts/metis/config/se.json",
    "models/tts/metis/config/tse.json",
    "models/tts/metis/config/tts.json",
    "models/tts/metis/config/vc.json",
    
    # G2P directory (not needed for audio encoding/decoding)
    "models/tts/maskgct/g2p",
    
    # Extra config
    "models/tts/maskgct/config/maskgct.json",
    "models/tts/maskgct/requirements.txt",
    
    # Empty nested directories
    "models/tts/metis/models",
]

# Utils files that might not be needed (keep for now, but list them)
# Most utils are probably not needed, but util.py and hparam.py are essential

def main():
    print("=" * 70)
    print("CLEANING UP REMAINING NON-ESSENTIAL FILES")
    print("=" * 70)
    
    deleted = 0
    
    for item in FILES_TO_DELETE:
        if os.path.exists(item):
            try:
                if os.path.isdir(item):
                    shutil.rmtree(item)
                    print(f"   Deleted directory: {item}")
                else:
                    os.remove(item)
                    print(f"   Deleted file: {item}")
                deleted += 1
            except Exception as e:
                print(f"   ERROR deleting {item}: {e}")
    
    # Clean up empty __pycache__ directories
    print("\n[Cleaning __pycache__ directories...]")
    for root, dirs, files in os.walk("."):
        if "__pycache__" in dirs:
            pycache_path = os.path.join(root, "__pycache__")
            try:
                shutil.rmtree(pycache_path)
                print(f"   Deleted: {pycache_path}")
            except:
                pass
    
    print(f"\n[SUCCESS] Cleaned up {deleted} items")
    print("=" * 70)

if __name__ == "__main__":
    main()


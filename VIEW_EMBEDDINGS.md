# How to View JSON Embeddings

This guide shows you how to inspect and visualize the extracted 8-D semantic embeddings.

## Quick View Commands

### 1. **View File Structure**
```bash
# See how many files are in the JSON
python3 -c "import json; data=json.load(open('/data/common/versaface/jsons/semantic_audio_embeddings_dummy.json')); print(f'Total files: {len(data)}')"

# List first 5 file paths
python3 -c "import json; data=json.load(open('/data/common/versaface/jsons/semantic_audio_embeddings_dummy.json')); print('First 5 files:'); [print(f'  {k}') for k in list(data.keys())[:5]]"
```

### 2. **View Embedding Shape**
```bash
# Check shape of embeddings for first file
python3 -c "
import json
data = json.load(open('/data/common/versaface/jsons/semantic_audio_embeddings_dummy.json'))
first_key = list(data.keys())[0]
emb = data[first_key]
print(f'File: {first_key}')
print(f'Number of time steps: {len(emb)}')
print(f'Dimensions per time step: {len(emb[0])} (should be 8)')
print(f'First 3 time steps:')
for i, vec in enumerate(emb[:3]):
    print(f'  Step {i}: {vec}')
"
```

### 3. **View Statistics**
```bash
python3 << 'EOF'
import json
import numpy as np

with open('/data/common/versaface/jsons/semantic_audio_embeddings_dummy.json', 'r') as f:
    data = json.load(f)

print("=" * 70)
print("EMBEDDING STATISTICS")
print("=" * 70)
print(f"Total files: {len(data)}")

# Collect all embeddings
all_embeddings = []
for key, emb in data.items():
    all_embeddings.extend(emb)

all_embeddings = np.array(all_embeddings)
print(f"\nTotal time steps: {len(all_embeddings)}")
print(f"Embedding dimension: {all_embeddings.shape[1]}")

print(f"\nStatistics per dimension:")
for dim in range(8):
    print(f"  Dim {dim}: min={all_embeddings[:, dim].min():.4f}, "
          f"max={all_embeddings[:, dim].max():.4f}, "
          f"mean={all_embeddings[:, dim].mean():.4f}, "
          f"std={all_embeddings[:, dim].std():.4f}")

print(f"\nPer-file statistics:")
for i, (key, emb) in enumerate(list(data.items())[:5]):
    emb_arr = np.array(emb)
    print(f"  {key}:")
    print(f"    Time steps: {len(emb)}")
    print(f"    Duration (approx): {len(emb)/50:.2f} seconds")
    print(f"    Mean values: {emb_arr.mean(axis=0)}")
EOF
```

## Python Script for Visualization

Create a simple visualization script:

```python
#!/usr/bin/env python3
"""Visualize 8-D semantic embeddings"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load embeddings
json_path = "/data/common/versaface/jsons/semantic_audio_embeddings_dummy.json"
with open(json_path, 'r') as f:
    data = json.load(f)

# Get first file's embeddings
first_key = list(data.keys())[0]
embeddings = np.array(data[first_key])

print(f"Visualizing: {first_key}")
print(f"Shape: {embeddings.shape} (time_steps, 8_dims)")

# Plot all 8 dimensions over time
fig, axes = plt.subplots(4, 2, figsize=(12, 10))
fig.suptitle(f'8-D Semantic Embeddings: {Path(first_key).name}', fontsize=14)

for dim in range(8):
    row = dim // 2
    col = dim % 2
    axes[row, col].plot(embeddings[:, dim])
    axes[row, col].set_title(f'Dimension {dim}')
    axes[row, col].set_xlabel('Time Step')
    axes[row, col].set_ylabel('Value')
    axes[row, col].grid(True)

plt.tight_layout()
plt.savefig('embeddings_visualization.png', dpi=150)
print("Saved visualization to: embeddings_visualization.png")
```

## Jupyter Notebook for Interactive Exploration

```python
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load embeddings
with open('/data/common/versaface/jsons/semantic_audio_embeddings_dummy.json', 'r') as f:
    embeddings_data = json.load(f)

# Explore structure
print(f"Total files: {len(embeddings_data)}")
print(f"\nFirst file: {list(embeddings_data.keys())[0]}")

# Convert to numpy for analysis
first_file = list(embeddings_data.keys())[0]
emb = np.array(embeddings_data[first_file])

print(f"\nEmbedding shape: {emb.shape}")
print(f"Time steps: {emb.shape[0]}")
print(f"Dimensions: {emb.shape[1]}")

# Plot
plt.figure(figsize=(12, 6))
for dim in range(8):
    plt.plot(emb[:, dim], label=f'Dim {dim}', alpha=0.7)
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.title(f'8-D Embeddings Over Time: {Path(first_file).name}')
plt.legend()
plt.grid(True)
plt.show()
```

## Command-Line Tools

### View specific file's embeddings:
```bash
python3 << 'EOF'
import json
import sys

json_path = sys.argv[1] if len(sys.argv) > 1 else '/data/common/versaface/jsons/semantic_audio_embeddings_dummy.json'
file_key = sys.argv[2] if len(sys.argv) > 2 else None

with open(json_path, 'r') as f:
    data = json.load(f)

if file_key:
    if file_key in data:
        emb = data[file_key]
        print(f"File: {file_key}")
        print(f"Time steps: {len(emb)}")
        print(f"First 5 time steps:")
        for i, vec in enumerate(emb[:5]):
            print(f"  {i}: {vec}")
    else:
        print(f"File not found: {file_key}")
        print(f"Available files (first 10):")
        for k in list(data.keys())[:10]:
            print(f"  {k}")
else:
    print(f"Total files: {len(data)}")
    print(f"\nFirst file: {list(data.keys())[0]}")
    print(f"Time steps: {len(data[list(data.keys())[0]])}")
EOF
```

## JSON Structure Reference

The JSON file has this structure:
```json
{
  "part_003/e8/7b/file.wav": [
    [d1, d2, d3, d4, d5, d6, d7, d8],  // Time step 1
    [d1, d2, d3, d4, d5, d6, d7, d8],  // Time step 2
    ...
  ],
  ...
}
```

- **Keys**: Relative paths from `/data/common/versaface/audios/`
- **Values**: List of 8-D vectors (one per time step)
- **Time step rate**: ~50 Hz (one vector per 20ms)
- **Duration**: `num_time_steps / 50` seconds

## Quick Tips

1. **Check file size**: `ls -lh /data/common/versaface/jsons/semantic_audio_embeddings_dummy.json`
2. **Count files**: `python3 -c "import json; print(len(json.load(open('...json'))))"`
3. **View first file**: Use the Python commands above
4. **Plot embeddings**: Use matplotlib to visualize the 8 dimensions over time


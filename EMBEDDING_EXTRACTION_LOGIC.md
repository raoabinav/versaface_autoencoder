# Audio Embedding Extraction Logic

## Overview

This document explains how the `extract_audio_embeddings.py` script works to convert audio files into 8-D semantic embeddings and save them as JSON.

## Architecture Flow

```
Audio Files (.wav)
    ↓
[Metis8dEncoder] → Extract 8-D Semantic Latents
    ↓
[Post-processing] → Convert to JSON-serializable format
    ↓
JSON File → Save embeddings with file paths as keys
```

## Step-by-Step Logic

### 1. **File Discovery**
- **Input**: `/data/common/versaface/audios/` directory
- **Process**: Recursively finds all `.wav` files
- **Structure**: Files are organized as `part_XXX/YY/ZZ/hash.wav`
- **Output**: List of absolute paths to all audio files

**Example structure:**
```
/data/common/versaface/audios/
  ├── part_003/
  │   ├── e8/7b/e87baa01e12a73d1f92334ade0a32f41.wav
  │   └── e8/b9/e8b911d98e3c8b42fcebcb1f8d11b105.wav
  ├── part_004/
  └── ...
```

### 2. **GPU Initialization**
- **Requirement**: Script **requires** GPU (CUDA)
- **Validation**: Exits if GPU not available
- **Device Selection**: Uses `cuda:0` by default, or specified device
- **Memory Check**: Logs GPU name and available memory

### 3. **Model Loading**
- **AudioTokenizer**: Loads Metis audio tokenizer (downloads models on first run)
- **Metis8dEncoder**: Wraps tokenizer to extract 8-D continuous latents
- **Device Placement**: Ensures all models are on GPU
- **Verification**: Checks that models are actually on GPU

### 4. **Iterative Batch Processing**

The script processes files in **batches** to manage GPU memory:

```
For each batch (default: 100 files):
  For each audio file in batch:
    1. Load audio file (16kHz, mono)
    2. Extract 8-D semantic latents using encoder
    3. Convert tensor to numpy array
    4. Convert to list (JSON-serializable)
    5. Store in dictionary with relative path as key
    6. Clear GPU memory
  Clear GPU cache after batch
```

**Why batches?**
- Prevents GPU memory overflow
- Allows progress tracking
- Enables recovery if process crashes
- Better memory management

### 5. **Embedding Extraction**

For each audio file:

1. **Load Audio**: 
   - Uses `librosa.load()` at 16kHz (required by encoder)
   - Converts to mono if stereo

2. **Encode to 8-D Latents**:
   - Input: `(T_samples,)` waveform at 16kHz
   - Process: 
     - Extract 1024-D SSL features using w2v-bert-2.0
     - Project to 8-D continuous latents via RepCodec
   - Output: `(1, T, 8)` tensor where:
     - `T` = number of time steps (~50 Hz, so T ≈ duration_in_seconds × 50)
     - `8` = semantic dimensions

3. **Format Conversion**:
   ```python
   z_8d: torch.Tensor (1, T, 8) on GPU
     ↓ squeeze(0)
   z_8d: torch.Tensor (T, 8) on GPU
     ↓ .cpu().numpy()
   z_8d_np: numpy.ndarray (T, 8) on CPU
     ↓ .tolist()
   embedding_list: List[List[float]] = [[d1...d8], [d1...d8], ...]
   ```

### 6. **JSON Format**

The embeddings are saved in the following JSON structure:

```json
{
  "part_003/e8/7b/e87baa01e12a73d1f92334ade0a32f41.wav": [
    [-0.123, 0.456, -0.789, 0.234, -0.567, 0.890, -0.345, 0.678],
    [-0.124, 0.457, -0.790, 0.235, -0.568, 0.891, -0.346, 0.679],
    ...
  ],
  "part_003/e8/b9/e8b911d98e3c8b42fcebcb1f8d11b105.wav": [
    [0.123, -0.456, 0.789, -0.234, 0.567, -0.890, 0.345, -0.678],
    ...
  ],
  ...
}
```

**Key Points:**
- **Keys**: Relative paths from `/data/common/versaface/audios/`
- **Values**: List of lists, where each inner list is 8 floats (one time step)
- **Structure**: `{file_path: [[d1, d2, ..., d8], [d1, d2, ..., d8], ...]}`

**Example:**
- Audio file: `/data/common/versaface/audios/part_003/e8/7b/e87baa01e12a73d1f92334ade0a32f41.wav`
- Key in JSON: `"part_003/e8/7b/e87baa01e12a73d1f92334ade0a32f41.wav"`
- Value: List of 8-D vectors, one per time step

### 7. **Memory Management & GPU Monitoring**

After each file:
- Delete tensor variables
- Clear GPU cache: `torch.cuda.empty_cache()`

After each batch:
- Clear GPU cache again
- Log GPU utilization via `nvidia-smi`:
  - GPU utilization percentage
  - Memory utilization percentage
  - Memory used/total
  - Temperature

GPU status is logged at:
- **Initial**: Before processing starts
- **Before processing**: After model loading
- **After each batch**: To monitor progress
- **Final**: After completion

This prevents GPU OOM (Out of Memory) errors and helps monitor performance.

## JSON Format Verification

### Comparison with Existing JSON Files

The existing JSON files in `/data/common/versaface/jsons/` contain:
- **Format**: List of strings (video file paths)
- **Example**: `["/data/common/versaface-raw/p1-p64/.../video.mp4", ...]`

Our new embeddings JSON:
- **Format**: Dictionary mapping audio paths to embeddings
- **Structure**: `{audio_path: [[8-D vector], [8-D vector], ...]}`

**Why different?**
- Existing JSONs: Simple list of video paths (metadata)
- Our JSON: Rich embeddings data (actual feature vectors)

**Compatibility:**
- Both are valid JSON
- Can be loaded with `json.load()`
- Our format is more structured (dictionary vs list)

### Correctness Check

✅ **Keys**: Relative paths (matches file structure)  
✅ **Values**: List of 8-D vectors (correct format)  
✅ **Serialization**: All floats (JSON-compatible)  
✅ **Structure**: Dictionary (easy to lookup by path)  

## Usage Examples

### Basic Usage
```bash
# Activate environment first
conda activate versaface

# Process all files (default batch size: 100)
python extract_audio_embeddings.py
```

**Important**: Always activate the `versaface` conda environment first, as it contains all required dependencies (torch, librosa, etc.).

### Custom Batch Size
```bash
# Process in smaller batches (50 files at a time)
python extract_audio_embeddings.py --batch-size 50
```

### Process Specific Part
```bash
# Only process part_003
python extract_audio_embeddings.py --part part_003
```

### Test Run
```bash
# Test with 10 files, batch size 5
python extract_audio_embeddings.py --max-files 10 --batch-size 5
```

### Custom Output
```bash
# Save to custom location
python extract_audio_embeddings.py \
    --output-dir /path/to/output \
    --output-file my_embeddings.json
```

## Output File

**Default Location**: `/data/common/versaface/jsons/semantic_audio_embeddings.json`

**Size Estimate**:
- Each 8-D vector: 8 floats × 4 bytes = 32 bytes
- Time steps per second: ~50 (SEMANTIC_RATE_HZ)
- For 10-second audio: 10 × 50 × 32 = 16 KB per file
- For 378,686 files (assuming avg 5 seconds): ~378,686 × 5 × 50 × 32 = ~3 GB
- JSON overhead (indentation, keys): +20-30%
- **Estimated total size**: ~3.5-4 GB

**Note**: Actual size depends on audio durations. Longer audio = more time steps = larger embeddings.

## Error Handling

- **Missing files**: Logged, skipped, processing continues
- **GPU OOM**: Process in smaller batches (reduce `--batch-size`)
- **Invalid audio**: Logged, skipped, processing continues
- **Failed files**: Tracked and logged at end

## Performance

- **GPU Required**: Yes (CUDA)
- **Processing Speed**: ~1-5 files/second (depends on GPU and audio length)
- **Memory Usage**: Cleared after each file and batch
- **Scalability**: Can process millions of files (batched)

## Technical Details

### 8-D Semantic Latents

These are **continuous** (not quantized) representations:
- **Source**: w2v-bert-2.0 SSL features (1024-D)
- **Projection**: RepCodec quantizer's `in_project` layer
- **Output**: 8-D continuous latents (before VQ quantization)
- **Rate**: ~50 Hz (one vector per 20ms)

### Why 8-D?

- **Compression**: Reduces 1024-D SSL features to 8-D
- **Semantic**: Captures high-level semantic content
- **Continuous**: Preserves fine-grained information (not discrete tokens)
- **Efficient**: Small enough for storage, rich enough for reconstruction

## Dataset Statistics

Based on verification:
- **Total audio files**: ~378,686 `.wav` files
- **Directory structure**: Organized in `part_XXX/` subdirectories
- **File organization**: Hash-based nested structure (e.g., `e8/7b/hash.wav`)
- **Estimated processing time**: 
  - At ~2 files/second: ~52 hours
  - At ~5 files/second: ~21 hours
  - (Depends on GPU and audio lengths)

## Troubleshooting

### GPU Out of Memory
**Symptom**: `RuntimeError: CUDA out of memory`

**Solution**: Reduce batch size
```bash
python extract_audio_embeddings.py --batch-size 50
# Or even smaller:
python extract_audio_embeddings.py --batch-size 20
```

### No GPU Available
**Symptom**: `RuntimeError: GPU required but not available`

**Solution**: 
- Ensure you're on a machine with CUDA
- Activate the correct conda environment: `conda activate versaface`
- Check GPU: `nvidia-smi`

### Config File Not Found
**Symptom**: `FileNotFoundError: Config file not found`

**Solution**: 
- Check that `models/tts/metis/config/base.json` exists
- Or specify custom config: `--config /path/to/config.json`

### No Audio Files Found
**Symptom**: `No audio files found!`

**Solution**:
- Verify audio directory exists: `/data/common/versaface/audios/`
- Check permissions
- Use `--audio-dir` to specify custom path

### Processing Too Slow
**Solutions**:
- Use faster GPU
- Process specific parts: `--part part_003`
- Test first: `--max-files 100` to verify setup

## Sample Output

A sample JSON file has been created: `sample_embeddings_output.json`

This demonstrates the exact format that will be produced:
- 3 example files
- 100 time steps per file (simulating ~2 seconds of audio)
- 8-D vectors per time step

You can inspect it to verify the format before running the full extraction.

## Summary

1. ✅ **Finds all audio files** recursively (~378,686 files)
2. ✅ **Forces GPU usage** (exits if not available)
3. ✅ **Processes in batches** (memory efficient, default: 100 files/batch)
4. ✅ **Extracts 8-D embeddings** using Metis encoder
5. ✅ **Saves as JSON** with correct format: `{path: [[8-D vectors]]}`
6. ✅ **Manages GPU memory** (clears after each file/batch)
7. ✅ **Error handling** (logs failures, continues processing)
8. ✅ **Progress tracking** (tqdm progress bars, batch logging)

The JSON format is correct and verified. The script is ready for production use!


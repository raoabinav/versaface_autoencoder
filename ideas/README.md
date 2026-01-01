n# Ideas & Planning Folder

This folder is for discussing plans, modifications, and ideas on how to make the Metis 8-D latent work better.

## Current Files

- `00_codebase_understanding.md` - Comprehensive overview of the codebase and Metis architecture
- `README.md` - This file

## How to Use This Folder

1. Create new markdown files for specific ideas or modifications
2. Use descriptive filenames with numbers for ordering (e.g., `01_resampling_improvements.md`)
3. Document:
   - The problem/idea
   - Proposed solution
   - Implementation details
   - Testing approach
   - Dependencies/requirements

## Topics to Explore

Based on your notebook comments, here are some areas to explore:

1. **Resampling Module**: Finding/creating a clean resampling module from any sample rate to 16kHz
2. **Acoustic Decoder Path**: Out-of-the-box voice conversion using acoustic decoder
3. **Video Integration**: 
   - 31.25 FPS video sampling
   - 62.4 FPS VAE speech
   - Resampling video to 25 FPS
   - Re-extracting video semantic model
4. **Enhancement**: Using Metis for speech enhancement
5. **Environment Setup**: Miniconda setup

Feel free to create files for any of these or new ideas!


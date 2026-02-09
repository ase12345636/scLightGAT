#!/bin/bash
# ============================================================================
# scLightGAT Data Downloader
# ============================================================================
# This script downloads the scLightGAT dataset from Hugging Face
# and places it in the correct directory format for the project.
#
# Usage:
#   chmod +x download_hf_data.sh
#   ./download_hf_data.sh
# ============================================================================

# Define the repository ID
REPO_ID="Alfiechuang/scLightGAT"

# Define the target directory relative to this script
# Assuming script is run from scLightGAT.main or project root
# We want data to end up in ../data/scLightGAT_data if running from scLightGAT.main
# Or ./data/scLightGAT_data if running from root

# Detect where we are
CURRENT_DIR=$(pwd)
DIR_NAME=$(basename "$CURRENT_DIR")

if [ "$DIR_NAME" == "scLightGAT" ]; then
    TARGET_ROOT="./scLightGAT_data"
else
    # Default to creating scLightGAT_data/ folder in current dir if not inside main repo
    TARGET_ROOT="./scLightGAT_data"
fi

echo "========================================================"
echo "scLightGAT Data Downloader"
echo "Target Repository: https://huggingface.co/datasets/$REPO_ID"
echo "Download Location: $TARGET_ROOT"
echo "========================================================"

# Check for Python and huggingface_hub
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found."
    exit 1
fi

# Create target directory
mkdir -p "$TARGET_ROOT"

echo "Starting download..."
# Using Python directly to download from Hugging Face
python3 << 'PYEOF'
import sys
import os
from huggingface_hub import snapshot_download

try:
    repo_id = os.environ.get('REPO_ID', 'Alfiechuang/scLightGAT')
    target_root = os.environ.get('TARGET_ROOT', './scLightGAT_data')
    
    print(f"Downloading {repo_id} to {target_root}...")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=target_root,
        local_dir_use_symlinks=False,
        ignore_patterns=[".gitattributes", "README.md"]
    )
    print("Download successful!")
except Exception as e:
    print(f"Error during download: {e}")
    sys.exit(1)
PYEOF

echo ""
echo "========================================================"
echo "Download Complete!"
echo "Data is located at: $TARGET_ROOT"
echo "You can now run the training scripts."
echo "========================================================"

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

# Check for huggingface-cli
if ! command -v huggingface-cli &> /dev/null; then
    echo "Error: huggingface-cli not found."
    echo "Please install it using: pip install huggingface_hub"
    exit 1
fi

# Create target directory
mkdir -p "$TARGET_ROOT"

echo "Starting download..."
# specific-files download is cleaner than cloning the whole repo if .git is not needed
# But downloading the whole snapshot is easiest for directory structure preservation.
# using --repo-type dataset

huggingface-cli download "$REPO_ID" \
    --repo-type dataset \
    --local-dir "$TARGET_ROOT" \
    --local-dir-use-symlinks False \
    --exclude ".gitattributes" "README.md"

echo ""
echo "========================================================"
echo "Download Complete!"
echo "Data is located at: $TARGET_ROOT"
echo "You can now run the training scripts."
echo "========================================================"

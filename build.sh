#!/usr/bin/env bash

# Exit immediately on error
set -e

# Update system packages and install dependencies
apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    poppler-utils

# Optional cleanup (removes cached apt data to reduce image size)
apt-get clean
rm -rf /var/lib/apt/lists/*

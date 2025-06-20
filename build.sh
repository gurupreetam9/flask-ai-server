#!/usr/bin/env bash

# Update system packages and install dependencies
apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    poppler-utils

# Optional: clean up to reduce image size
apt-get clean

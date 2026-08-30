#!/usr/bin/env bash
# Render build script — runs during deploy
set -o errexit

# Install system dependencies
apt-get update && apt-get install -y ffmpeg libgl1-mesa-glx libglib2.0-0

# Install Python dependencies
pip install --upgrade pip
pip install -r CrimeVision-backend/requirements.txt

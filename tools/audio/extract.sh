#!/usr/bin/env bash
# Extract the source audio track (native 48 kHz, all channels, float32) into
# out/audio/ (gitignored, never commit).
#   out/audio/full.wav  - whole track, sample 0 = container time 0
#                         (aresample first_pts=0 keeps container timing)
set -euo pipefail
cd "$(dirname "$0")/../.."
SRC=$(ls *.mkv | head -1)
mkdir -p out/audio
ffmpeg -v error -y -i "$SRC" -map 0:a:0 -vn -af "aresample=async=1:first_pts=0" \
  -c:a pcm_f32le out/audio/full.wav
ffprobe -v error -show_entries stream=sample_rate,channels,channel_layout,duration -of compact out/audio/full.wav

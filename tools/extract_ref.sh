#!/usr/bin/env bash
# Extract reference frames for 0:12-0:21 into ref/frames (gitignored).
# Frames are scaled to the 960x540 scoring resolution.
set -euo pipefail
cd "$(dirname "$0")/.."
SRC=$(ls *.mkv | head -1)
mkdir -p ref/frames
ffmpeg -v error -y -ss 12 -i "$SRC" -t 9 -vf "scale=960:540:flags=area" -start_number 0 ref/frames/%04d.png
ls ref/frames | wc -l

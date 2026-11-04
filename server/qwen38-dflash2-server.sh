#!/usr/bin/env bash
set -euo pipefail

exec /home/asad/llama-dflash2/build/bin/llama-server \
  -m /home/asad/models/qwen38-27b-q4/Qwen3.8-27B-Q4_K_M.gguf \
  -md /home/asad/models/qwen38-27b-dflash2/Qwen3.8-27B-DFlash2-Q4_K_M.gguf \
  -a Qwen3.8-27B-Q4_K_M \
  --host 127.0.0.1 \
  --port 18088 \
  --ctx-size 16384 \
  --n-gpu-layers all \
  --flash-attn on \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --batch-size 2048 \
  --ubatch-size 512 \
  --parallel 1 \
  --jinja \
  --reasoning on \
  --reasoning-effort high \
  --spec-type draft-dflash \
  --spec-draft-n-max 4

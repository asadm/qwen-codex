#!/usr/bin/env bash
set -euo pipefail

exec /home/asad/llama-dflash2/build/bin/llama-server \
  -m /home/asad/models/qwen38-27b-iq4xs/Qwen3.8-27B-IQ4_XS.gguf \
  -md /home/asad/models/qwen38-27b-dflash2/Qwen3.8-27B-DFlash2-Q4_K_M.gguf \
  -a Qwen3.8-27B-IQ4_XS \
  --host 127.0.0.1 \
  --port 18089 \
  --ctx-size 262144 \
  --n-gpu-layers all \
  --flash-attn on \
  --fit off \
  --cache-type-k q4_0 \
  --cache-type-v q4_0 \
  --batch-size 512 \
  --ubatch-size 128 \
  --parallel 1 \
  --jinja \
  --chat-template-file /home/asad/models/qwen38-codex-chat-template.jinja \
  --reasoning on \
  --reasoning-effort high \
  --spec-type draft-dflash \
  --spec-draft-n-max 4

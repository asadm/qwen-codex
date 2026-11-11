# qwen-codex

Run [Codex CLI](https://github.com/openai/codex) threads against a **local
Qwen3.8-27B** model, served by a patched llama.cpp on a GPU box over SSH.
No cloud API; full local speed (speculative decoding, long context,
multi-instance).

```
codex CLI  ──HTTP /v1/responses──>  qwen-codex-bridge.py  (adapter, loopback)
   (provider: qwen_tailscale)            │ strips OpenAI-private compaction items
                                         │ truncates history to fit the context window
                                         ▼
  ssh tunnel (started by codex-qwen) ──────────>  llama-server  (Qwen3.8-27B, patched)
```

## What you need

- A GPU machine that runs a Qwen3.8-27B GGUF (this setup: a llama.cpp fork
  on a Tailscale host; one 27B model fits in VRAM at a time)
- `codex` CLI on the machine you type on (tested with Codex 0.166.x)
- Python 3 (stdlib only — the bridge has no dependencies)
- SSH from the typing machine to the GPU box

## Setup

### 1. GPU box — patched llama-server

```bash
git clone https://github.com/ggml-org/llama.cpp ~/llama-dflash2
cd ~/llama-dflash2 && git checkout 7071ba3    # base this patch was made against
git apply /path/to/this/repo/server/llama-cpp-codex-compat.patch
cmake -B build                               # + your backend, e.g. -DGGML_CUDA=on
cmake --build build --target llama-server -j
```

The patch makes the Responses→chat conversion skip items only OpenAI's
backend can consume (`compaction` blobs, `additional_tools`, empty
`reasoning`) and turns Codex `custom_tool_call` items into JSON-argument
function calls. Without it, resuming any OpenAI-compacted thread fails with
400 "Cannot determine type of 'item'" (openai/codex#45393).

Put your GGUFs somewhere (this setup: `~/models/...`) and point the server
scripts at them:

- `Qwen3.8-27B-IQ4_XS.gguf` — full mode (262k context)
- `Qwen3.8-27B-Q4_K_M.gguf` + `Qwen3.8-27B-DFlash2-Q4_K_M.gguf` — fast mode
  (16k context). The draft model needs the fork's `--spec-type draft-dflash`;
  on plain upstream llama.cpp drop the `--spec-*` flags (you lose spec speed).

Copy `server/qwen38-full-server.sh` and `server/qwen38-dflash2-server.sh` to
`~/bin/` and edit the model paths. `--parallel 1` is deliberate (single
slot; the bridge queues for it).

Both scripts load the Qwen3.8-27B vision projector from
`~/models/qwen38-27b-q4/mmproj-F16.gguf` with `--mmproj`, enabling Codex image
attachments. Restart a running server after installing an updated script.
The full-context 4090 setup leaves limited GPU memory for large image requests.

### 2. Typing machine — launcher + bridge

```bash
git clone https://github.com/asadm/qwen-codex ~/qwen-codex
ln -s ~/qwen-codex/codex-qwen ~/bin/codex-qwen   # or add ~/qwen-codex to PATH
```

The launcher finds the bridge next to itself (or `$HOME/qwen-codex-bridge.py`,
or `$QWEN_BRIDGE`). Edit the `QWEN_REMOTE` / `QWEN_SSH_KEY` defaults at the
top of `codex-qwen` to point at your box.

### 3. Run

```bash
codex-qwen                      # picker: new conversation / fork an OpenAI thread
codex-qwen --fast               # small-context mode
codex-qwen exec resume <id> "..."   # any codex subcommand works
```

Multiple concurrent instances are fine: the launcher walks to the next free
local port per instance (tunnel + bridge pair); the remote server is shared.

## Ports

| mode             | local tunnel | bridge (tunnel+100) | remote | ctx    |
|------------------|--------------|---------------------|--------|--------|
| full (default)   | 18189        | 18289               | 18089  | 262144 |
| `--fast`         | 18188        | 18288               | 18088  | 16384  |

Standalone bridge (what the launcher starts per instance):

```bash
python3 qwen-codex-bridge.py --port 18299 --upstream 18189 [--max-tokens 240000]
# env: QWEN_PROXY_PORT, QWEN_PROXY_UPSTREAM, QWEN_PROXY_MAX_TOKENS
```

## Why the bridge

Three things a single-slot local llama.cpp cannot do, and codex needs:

1. **Compaction items** — a thread auto-compacted on OpenAI carries
   `{"type":"compaction", ...encrypted_content}` items in its persisted
   history; codex replays them verbatim on resume and third-party endpoints
   reject them. The bridge strips them (the server patch does too).
2. **Context overflow** — big OpenAI threads replay far more tokens than a
   local window holds. The bridge truncates the replay to `ctx - 20000` at a
   user/developer message boundary, keeps the newest history, and marks the
   cut with a `[bridge: …]` notice.
3. **Single slot** — with `--parallel 1`, the bridge single-flights requests
   instead of letting them race into the busy response.

It is a streaming pass-through (no per-token overhead).

## Troubleshooting

- **"We're currently experiencing high demand…"** — the server's busy
  response; codex retries. Several concurrent instances just queue.
- **400 "Cannot determine type of 'item'"** — server missing the patch, or
  codex pointed at the tunnel (not the bridge) on a compacted thread.
- **Port in use** — a stale earlier instance (its ssh tunnel holds the
  port); kill its process group. New launches also auto-walk around it.
- **Truncation on big resums** — expected: only the most recent history is
  kept; the `[bridge: …]` message marks the cut. Older content stays in the
  OpenAI thread.

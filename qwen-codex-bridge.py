#!/usr/bin/env python3
"""Codex -> local llama.cpp adapter bridge (loopback only).

Sits between codex (wire_api="responses") and the SSH tunnel to the
patched llama-server. It is a streaming pass-through (no per-token
overhead) that adds the three things a single-slot local server cannot:

  1. Strip OpenAI-private history items a local model cannot consume:
     `compaction` / `context_compaction` blobs (encrypted_content),
     `additional_tools`, and empty (encrypted-only) `reasoning` items.
     Without this, resuming an OpenAI-compacted thread 400s with
     "Cannot determine type of 'item'" (openai/codex#45393).
  2. Truncate over-long replayed history to fit the local context
     window, cutting at the oldest user/developer message boundary whose
     suffix fits, so tool-call pairs stay intact and the most recent
     history is kept. Without this, large OpenAI threads (e.g. ~1.5M
     token replays) can never resume on a 262k-context local model.
  3. Serialize requests to the single-slot upstream (`--parallel 1`)
     instead of letting them race into llama.cpp's busy response.

Usage:
  qwen-codex-bridge.py [--port 18299] [--upstream 18189] [--max-tokens N]

Env overrides: QWEN_PROXY_PORT, QWEN_PROXY_UPSTREAM, QWEN_PROXY_MAX_TOKENS.
Keep N below the model's context window, with headroom for output
(~/codex-qwen passes context - 20000).
"""
import argparse
import http.client
import json
import os
import socket
import sys
import threading
import time

DEFAULT_MAX_TOKENS = 240_000
CHUNK = 262_144
CHARS_PER_TOKEN = 3.5

UPSTREAM = ("127.0.0.1", 18189)
MAX_TOKENS = DEFAULT_MAX_TOKENS
SINGLE_FLIGHT = threading.Lock()
STATS = {"requests": 0, "stripped": 0, "truncated": 0, "queued_wait": 0.0}
STATS_LOCK = threading.Lock()

FORWARD_SKIP = ("host", "content-length", "transfer-encoding",
                "connection", "accept-encoding", "expect")


def log(msg):
    print(f"bridge: {msg}", file=sys.stderr, flush=True)


def sanitize_input(items):
    """Drop items a local model cannot consume. Returns (items, stripped)."""
    kept, stripped = [], 0
    for item in items:
        if isinstance(item, dict):
            itype = item.get("type")
            if itype in ("compaction", "context_compaction",
                         "additional_tools"):
                stripped += 1
                continue
            if itype == "reasoning":
                content = item.get("content")
                if not isinstance(content, list) or not content:
                    stripped += 1
                    continue
        kept.append(item)
    return kept, stripped


def fit_budget(items, budget_tokens):
    """Keep the most recent history that fits the token budget.

    Cuts at the oldest user/developer message boundary whose suffix
    fits the budget, so tool-call pairs stay intact. Returns
    (items, truncated)."""
    budget_chars = int(budget_tokens * CHARS_PER_TOKEN)
    lens = [len(json.dumps(item)) for item in items]
    total = sum(lens)
    if total <= budget_chars:
        return items, False
    cut = None
    prefix = 0
    for index, item in enumerate(items):
        suffix = total - prefix
        if suffix <= budget_chars and isinstance(item, dict) \
                and item.get("type") == "message" \
                and item.get("role") in ("user", "developer"):
            cut = index
            break
        prefix += lens[index]
    if cut is None:
        acc = 0
        cut = len(items)
        for index in range(len(items) - 1, -1, -1):
            acc += lens[index]
            if acc > budget_chars:
                break
            cut = index
        if cut == len(items):
            cut = len(items) - 1
    while cut < len(items) and isinstance(items[cut], dict) \
            and items[cut].get("type") == "function_call_output":
        cut += 1
    notice = {
        "type": "message",
        "role": "user",
        "content": [{
            "type": "input_text",
            "text": ("[bridge: the earliest part of this conversation was "
                     "truncated to fit the local model's context window; "
                     "the most recent history below is retained verbatim]"),
        }],
    }
    return [notice] + items[cut:], True


def read_head(rfile):
    line = rfile.readline(65537)
    if not line or line in (b"\r\n", b"\n"):
        return None
    parts = line.decode("latin1").split()
    if len(parts) < 2:
        return None
    method, path = parts[0], parts[1]
    headers = {}
    while True:
        line = rfile.readline()
        if line in (b"\r\n", b"\n", b""):
            break
        if b":" in line:
            key, value = line.split(b":", 1)
            headers[key.strip().lower().decode("latin1")] = \
                value.strip().decode("latin1")
    return method, path, headers


def read_body(rfile, headers):
    if "chunked" in headers.get("transfer-encoding", ""):
        body = b""
        while True:
            size_line = rfile.readline().split(b";")[0].strip()
            size = int(size_line or b"0", 16)
            if size == 0:
                rfile.readline()
                return body
            body += rfile.read(size)
            rfile.readline()
    return rfile.read(int(headers.get("content-length", "0")))


def upstream_request(method, path, body, headers):
    conn = http.client.HTTPConnection(UPSTREAM[0], UPSTREAM[1])
    fwd = {k: v for k, v in headers.items() if k not in FORWARD_SKIP}
    conn.request(method, path, body=body or None, headers=fwd)
    return conn, conn.getresponse()


def write_full(wfile, resp, data):
    wfile.write(f"HTTP/1.1 {resp.status} {resp.reason}\r\n".encode())
    wfile.write(
        f"Content-Type: "
        f"{resp.getheader('Content-Type', 'application/json')}\r\n"
        .encode())
    wfile.write(f"Content-Length: {len(data)}\r\n".encode())
    wfile.write(b"Connection: close\r\n\r\n")
    wfile.write(data)
    wfile.flush()


def handle(conn):
    rfile = conn.makefile("rb")
    wfile = conn.makefile("wb")
    parsed = read_head(rfile)
    if not parsed:
        return
    method, path, headers = parsed
    if headers.get("expect", "").lower() == "100-continue":
        wfile.write(b"HTTP/1.1 100 Continue\r\n\r\n")
        wfile.flush()
    body = read_body(rfile, headers)

    if method == "POST" and path.startswith("/v1/responses"):
        req = json.loads(body)
        items = req.get("input")
        stripped = truncated = 0
        if isinstance(items, list):
            items, stripped = sanitize_input(items)
            items, truncated = fit_budget(items, MAX_TOKENS)
            req["input"] = items
            body = json.dumps(req).encode()
        with STATS_LOCK:
            STATS["requests"] += 1
            STATS["stripped"] += stripped
            STATS["truncated"] += truncated
        log(f"responses: items={len(items)} stripped={stripped} "
            f"truncated={truncated} body={len(body) // 1024}KB")

        acquired = SINGLE_FLIGHT.acquire(blocking=False)
        if not acquired:
            start = time.monotonic()
            SINGLE_FLIGHT.acquire()
            waited = time.monotonic() - start
            with STATS_LOCK:
                STATS["queued_wait"] += waited
            log(f"waited {waited:.1f}s for the single-flight slot")
        try:
            upstream, resp = upstream_request("POST", path, body, headers)
            try:
                if resp.status == 200:
                    wfile.write(b"HTTP/1.1 200 OK\r\n")
                    wfile.write(
                        f"Content-Type: "
                        f"{resp.getheader('Content-Type', 'text/event-stream')}"
                        "\r\n".encode())
                    wfile.write(
                        b"Transfer-Encoding: chunked\r\nConnection: close\r\n\r\n")
                    wfile.flush()
                    while True:
                        data = resp.read1(CHUNK)
                        if not data:
                            break
                        wfile.write(f"{len(data):x}\r\n".encode())
                        wfile.write(data)
                        wfile.write(b"\r\n")
                        wfile.flush()
                    wfile.write(b"0\r\n\r\n")
                    wfile.flush()
                else:
                    data = resp.read()
                    write_full(wfile, resp, data)
                    log(f"upstream {resp.status} for {path}: {data[:400]!r}")
            finally:
                upstream.close()
        finally:
            SINGLE_FLIGHT.release()
        return

    upstream, resp = upstream_request(method, path, body, headers)
    try:
        write_full(wfile, resp, resp.read())
        if not path.startswith("/health"):
            log(f"passthrough {method} {path} -> {resp.status}")
    finally:
        upstream.close()


def serve_conn(conn):
    try:
        handle(conn)
    except (BrokenPipeError, ConnectionResetError):
        pass
    except Exception as exc:
        log(f"error: {exc!r}")
    finally:
        try:
            conn.close()
        except Exception:
            pass


def main():
    global UPSTREAM, MAX_TOKENS
    parser = argparse.ArgumentParser(description="Codex -> llama.cpp adapter bridge")
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get("QWEN_PROXY_PORT", "18299")))
    parser.add_argument("--upstream", type=int,
                        default=int(os.environ.get("QWEN_PROXY_UPSTREAM", "18189")))
    parser.add_argument("--max-tokens", type=int,
                        default=int(os.environ.get("QWEN_PROXY_MAX_TOKENS",
                                                   str(DEFAULT_MAX_TOKENS))))
    args = parser.parse_args()
    UPSTREAM = ("127.0.0.1", args.upstream)
    MAX_TOKENS = args.max_tokens
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", args.port))
    server.listen(16)
    log(f"listening on 127.0.0.1:{args.port} -> 127.0.0.1:{args.upstream} "
        f"(history budget {args.max_tokens} tokens)")
    while True:
        conn, _ = server.accept()
        threading.Thread(target=serve_conn, args=(conn,), daemon=True).start()


if __name__ == "__main__":
    main()

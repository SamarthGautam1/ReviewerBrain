"""Minimal client for a LOCAL LLM server (Ollama native /api/chat).

Egress policy: this harness may only talk to a loopback endpoint. The
project runs fully locally (no cloud APIs, no data egress), so the URL is
strictly validated before every request — scheme http/https, no
credentials, and the host must be a loopback name that resolves to a
loopback address. Remote/private-network hosts other than loopback are
rejected outright (DNS rebinding is covered by resolving and checking
every returned address).
"""
import ipaddress
import json
import socket
import time
from urllib.parse import urlsplit

import requests

DEFAULT_ENDPOINT = "http://127.0.0.1:11434"
DEFAULT_TIMEOUT_S = 300

_ALLOWED_HOSTNAMES = {"localhost"}
_SCHEMES = {"http", "https"}


class LLMError(RuntimeError):
    pass


def _is_allowed_host(host):
    """localhost (resolved + checked) or a literal loopback address."""
    if host.lower() in _ALLOWED_HOSTNAMES:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def validate_endpoint(endpoint):
    """Return (scheme, host, port) after enforcing the loopback-only policy.
    Raises LLMError for anything that is not a plain http(s) URL pointing at
    the local machine."""
    parts = urlsplit(endpoint)
    if parts.scheme.lower() not in _SCHEMES:
        raise LLMError(f"endpoint scheme must be http/https: {endpoint!r}")
    if parts.username or parts.password:
        raise LLMError("endpoint must not carry credentials")
    host = parts.hostname
    if not host or not _is_allowed_host(host):
        raise LLMError(f"endpoint host must be localhost or a loopback "
                       f"address (local inference only): got {host!r}")
    try:
        infos = socket.getaddrinfo(host, parts.port or 0,
                                   type=socket.SOCK_STREAM)
    except OSError as e:
        raise LLMError(f"cannot resolve endpoint host {host!r}: {e}") from e
    for info in infos:
        addr = ipaddress.ip_address(info[4][0])
        if not addr.is_loopback:
            raise LLMError(f"endpoint host {host!r} resolves to "
                           f"non-loopback address {addr}; refusing")
    return parts.scheme, host, parts.port


def chat(endpoint, model, messages, options=None, timeout_s=DEFAULT_TIMEOUT_S):
    """One chat completion. Returns (content, meta) where meta carries
    latency and the token counters Ollama reports (for the run manifest)."""
    validate_endpoint(endpoint)
    url = endpoint.rstrip("/") + "/api/chat"
    payload = {"model": model, "messages": messages, "stream": False,
               "options": options or {}}
    t0 = time.perf_counter()
    try:
        resp = requests.post(url, json=payload, timeout=timeout_s,
                             allow_redirects=False)
    except requests.ConnectionError as e:
        raise LLMError(f"cannot reach local LLM server at {url} "
                       f"(is it running?) — {e}") from e
    latency = time.perf_counter() - t0
    if resp.status_code != 200:
        raise LLMError(f"{url} returned HTTP {resp.status_code}: "
                       f"{resp.text[:300]}")
    body = resp.json()
    content = body.get("message", {}).get("content", "")
    if not content.strip():
        raise LLMError(f"empty completion from {model}: {json.dumps(body)[:300]}")
    meta = {"latency_s": round(latency, 2),
            "prompt_eval_count": body.get("prompt_eval_count"),
            "eval_count": body.get("eval_count"),
            "done_reason": body.get("done_reason")}
    return content, meta

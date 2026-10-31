# src/cloud/gemini_client.py
"""Google Gemini API client used by app2.py instead of the local Ollama model."""
import os
import re
import json
import time
import requests

API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

def _headers():
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set")
    return {"x-goog-api-key": key, "Content-Type": "application/json"}

def _retry_delay(response):
    """Seconds to wait from Gemini's RetryInfo (e.g. "retryDelay": "23s"), if present"""
    match = re.search(r'"retryDelay":\s*"(\d+)', response.text or "")
    return int(match.group(1)) + 1 if match else None

def _post(url, payload, stream=False, retries=3, max_wait=20):
    """POST with a few retries on short rate limits (429) and temporary server errors.
    Long waits (e.g. the daily quota is used up) are not retried; the error is shown instead."""
    for attempt in range(retries):
        response = requests.post(url, json=payload, headers=_headers(), stream=stream, timeout=120)
        if response.status_code in (429, 500, 503) and attempt < retries - 1:
            wait = _retry_delay(response) or 2 ** attempt * 2
            if wait > max_wait:
                return response
            print(f"Gemini API {response.status_code}, retrying in {wait}s...", flush=True)
            time.sleep(wait)
            continue
        return response
    return response

def _error_message(response):
    try:
        message = response.json()["error"]["message"]
    except Exception:
        message = response.text[:200]
    if response.status_code == 429:
        message = "The Gemini API quota is used up for now. Please try again later. " + message.split("\n")[0]
    return f"Error: Gemini API {response.status_code}: {message}"

def _generate_payload(prompt):
    return {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.7, "maxOutputTokens": 500}
    }

def _candidate_text(data):
    candidates = data.get("candidates") or []
    if not candidates:
        return ""
    parts = (candidates[0].get("content") or {}).get("parts") or []
    return "".join(p.get("text", "") for p in parts)

def generate(prompt, model=GEMINI_MODEL):
    """Generate a full response"""
    try:
        response = _post(f"{API_BASE}/{model}:generateContent", _generate_payload(prompt))
        if response.status_code != 200:
            return _error_message(response)
        return _candidate_text(response.json()) or "Error: Gemini returned an empty response."
    except Exception as e:
        print(f"Gemini error: {e}")
        return f"Error: {str(e)}"

def generate_stream(prompt, model=GEMINI_MODEL):
    """Generate a streaming response, yielding text pieces as they arrive"""
    try:
        response = _post(f"{API_BASE}/{model}:streamGenerateContent?alt=sse",
                         _generate_payload(prompt), stream=True)
        if response.status_code != 200:
            yield _error_message(response)
            return

        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            try:
                text = _candidate_text(json.loads(line[6:]))
            except json.JSONDecodeError:
                continue
            if text:
                yield text
    except Exception as e:
        print(f"Gemini streaming error: {e}")
        yield f"Error: {str(e)}"

_availability = {"checked_at": 0, "ok": False}

def check_gemini_available(model=GEMINI_MODEL, cache_seconds=300):
    """True if the API key works and the model exists (cached so /health stays cheap)"""
    if time.time() - _availability["checked_at"] < cache_seconds:
        return _availability["ok"]
    try:
        response = requests.get(f"{API_BASE}/{model}", headers=_headers(), timeout=10)
        ok = response.status_code == 200
    except Exception:
        ok = False
    _availability.update(checked_at=time.time(), ok=ok)
    return ok

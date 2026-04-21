"""Test script for LLM connection."""
import json
import urllib.request

url = "http://localhost:8000/v1/chat/completions"
data = {
    "model": "gemma-4-31b-it@q4_k_xl",
    "messages": [{"role": "user", "content": "What is DSD in SDMX? Answer briefly."}],
    "max_tokens": 100,
    "temperature": 0.1
}

req = urllib.request.Request(
    url,
    data=json.dumps(data).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)

with urllib.request.urlopen(req) as resp:
    result = json.loads(resp.read().decode())
    msg = result["choices"][0]["message"]
    print(f"content: {repr(msg.get('content', '')[:200])}")
    print(f"reasoning: {repr(msg.get('reasoning_content', '')[:100] if msg.get('reasoning_content') else '')}")
    print(f"finish: {result['choices'][0].get('finish_reason')}")
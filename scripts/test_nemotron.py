"""H1: Verify Nemotron Super function calling via NVIDIA NIM API."""
import os
import json
import httpx

API_KEY = os.environ.get("NVIDIA_API_KEY", "")
BASE_URL = "https://integrate.api.nvidia.com/v1"
MODEL = "nvidia/nemotron-3-super-120b-a12b"

# One dummy tool
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"}
                },
                "required": ["city"]
            }
        }
    }
]

payload = {
    "model": MODEL,
    "messages": [
        {"role": "system", "content": "You are a helpful assistant. Use tools when appropriate."},
        {"role": "user", "content": "What's the weather in San Francisco?"}
    ],
    "tools": tools,
    "tool_choice": "auto",
    "max_tokens": 256,
    "temperature": 0.1
}

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

print(f"Model: {MODEL}")
print(f"API key set: {bool(API_KEY)}")
print(f"API key prefix: {API_KEY[:12]}..." if API_KEY else "NO KEY")
print()

try:
    resp = httpx.post(f"{BASE_URL}/chat/completions", json=payload, headers=headers, timeout=30)
    print(f"Status: {resp.status_code}")
    
    if resp.status_code == 200:
        data = resp.json()
        msg = data["choices"][0]["message"]
        print(f"Finish reason: {data['choices'][0]['finish_reason']}")
        print(f"Role: {msg['role']}")
        
        if msg.get("tool_calls"):
            tc = msg["tool_calls"][0]
            print(f"\n✅ TOOL CALL RECEIVED")
            print(f"  Tool ID:   {tc['id']}")
            print(f"  Type:      {tc['type']}")
            print(f"  Function:  {tc['function']['name']}")
            print(f"  Arguments: {tc['function']['arguments']}")
            
            # Verify arguments parse as JSON
            args = json.loads(tc["function"]["arguments"])
            print(f"  Parsed:    {args}")
            print(f"\n✅ H1 PASS: Nemotron function calling works. OpenAI-compatible format confirmed.")
        else:
            print(f"\n⚠️  No tool_calls in response. Content: {msg.get('content', '')[:200]}")
            print("Try with tool_choice='required' or check model support.")
    else:
        print(f"Error body: {resp.text[:500]}")
        
except Exception as e:
    print(f"❌ Request failed: {e}")

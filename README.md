# mbs-python — Python SDK for MBS Workbench

Official async Python client for [MBS Workbench](https://github.com/MBSTech-sol/MBS-Workbench).
Connects to a running `mbsd` daemon or any OpenAI-compatible server.

## Installation

```bash
pip install mbs-python
# with NumPy support:
pip install "mbs-python[numpy]"
# with Pandas support:
pip install "mbs-python[pandas]"
# everything:
pip install "mbs-python[all]"
```

## Quick Start

```python
import asyncio
from mbs import MbsClient

async def main():
    async with MbsClient(base_url="http://127.0.0.1:3030") as client:
        # Chat completion
        resp = await client.chat([
            {"role": "user", "content": "Explain Rust ownership in 2 sentences."}
        ])
        print(resp.choices[0].message.content)

        # Streaming
        async for delta in await client.chat_stream([
            {"role": "user", "content": "Write a haiku about async."}
        ]):
            print(delta, end="", flush=True)
        print()

        # Embeddings
        resp = await client.embed("Hello, world!")
        print(len(resp.data[0].embedding))  # vector dimension

asyncio.run(main())
```

## API Reference

### `MbsClient`

```python
client = MbsClient(
    base_url="http://127.0.0.1:3030",  # default
    api_key=None,                        # optional bearer token
    timeout=120.0,                       # seconds
    max_retries=3,                       # on 429/5xx
    retry_base_delay_ms=500,
)
```

| Method | Description |
|--------|-------------|
| `await client.models()` | List available models → `ModelsResponse` |
| `await client.load_model(req)` | Load a `.gguf` into VRAM → `ModelLoadResponse` |
| `await client.unload_model()` | Unload current model → `ModelUnloadResponse` |
| `await client.chat(messages, *, model, temperature, max_tokens, stop)` | Chat completion → `ChatCompletionResponse` |
| `await client.chat_stream(messages, ...)` | Async iterator yielding text deltas |
| `await client.complete(prompt, ...)` | Text completion → `CompletionResponse` |
| `await client.embed(input, *, model)` | Embeddings → `EmbeddingResponse` |
| `await client.embed_numpy(input, *, model)` | Embeddings as NumPy array (requires `[numpy]`) |
| `await client.generate_image(prompt, *, n, size)` | Image generation → `ImageGenerationResponse` |
| `await client.run_agent(task, *, model, max_iterations)` | ReAct agent → `AgentRunResponse` |
| `await client.list_tools()` | List MCP tools → `McpToolsResponse` |
| `await client.invoke_tool(tool_id, arguments)` | Invoke MCP tool → `McpInvokeResponse` |
| `await client.anthropic_messages(req)` | Anthropic-compatible → `AnthropicMessagesResponse` |
| `await client.batch_chat(requests, *, concurrency)` | Batch chat → `BatchSummary` |
| `await client.batch_embed(inputs, *, model, concurrency)` | Batch embeddings → `BatchSummary` |
| `await client.batch_embed_dataframe(series, ...)` | Add embedding column to Pandas Series (requires `[pandas]`) |
| `await client.ping()` | Health check → `bool` |

## Streaming

```python
async for delta in await client.chat_stream([
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user",   "content": "Tell me about Python 3.12."},
]):
    print(delta, end="", flush=True)
```

## Batch Processing

```python
from mbs import ChatCompletionRequest, ChatMessage

requests = [
    ChatCompletionRequest(messages=[ChatMessage(role="user", content=f"What is {i}^2?")])
    for i in range(10)
]

summary = await client.batch_chat(requests, concurrency=5)
print(f"{summary.succeeded}/{len(summary.results)} succeeded")

for item in summary.results:
    if item.ok:
        data = item.value  # dict from ChatCompletionResponse.model_dump()
        print(data["choices"][0]["message"]["content"])
    else:
        print(f"Error: {item.error}")
```

## NumPy / Pandas Integration

```python
import numpy as np
import pandas as pd

# Get embeddings as a numpy array — shape (n, dim)
vectors = await client.embed_numpy(["hello", "world", "rust"])
print(vectors.shape)  # e.g. (3, 4096)

# Add embedding column to a DataFrame
df = pd.DataFrame({"text": ["foo", "bar", "baz"]})
df["embedding"] = await client.batch_embed_dataframe(
    df["text"], concurrency=3
)
```

## Error Handling

```python
from mbs import MbsError

try:
    resp = await client.chat([{"role": "user", "content": "Hi"}])
except MbsError as e:
    print(f"HTTP {e.status_code}: {e}")
```

Transient errors (429, 500–504) are automatically retried with exponential backoff + jitter.

## Context Manager

```python
async with MbsClient() as client:
    print(await client.ping())
# connection pool is closed automatically
```

## Running Tests

```bash
cd sdk/python
pip install -e ".[dev]"
pytest tests/ -v
```

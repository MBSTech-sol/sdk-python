"""
mbs-python — MBS Workbench Python SDK
Async HTTP client with auto-retry, cancellation, and batch processing.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncIterator, Dict, List, Optional, Sequence, TypeVar

import httpx
from httpx_sse import aconnect_sse

from .types import (
    AgentRunRequest,
    AgentRunResponse,
    AnalyticsSummary,
    AnthropicMessagesRequest,
    AnthropicMessagesResponse,
    BatchResultItem,
    BatchSummary,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    CompletionRequest,
    CompletionResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    HealthResponse,
    ImageGenerationRequest,
    ImageGenerationResponse,
    MbsError,
    McpInvokeRequest,
    McpInvokeResponse,
    McpToolsResponse,
    ModelLoadRequest,
    ModelLoadResponse,
    ModelUnloadResponse,
    ModelsResponse,
    PoolAddRequest,
    PoolListResponse,
    ModelSwitchRequest,
    QuotaConfigUpdate,
    QueueStatus,
    StreamChunk,
    WebhookAddRequest,
    WebhookItem,
    WebhookListResponse,
)

logger = logging.getLogger("mbs")

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_T = TypeVar("_T")


async def _sleep_jitter(attempt: int, base_delay_ms: int) -> None:
    """Exponential backoff with jitter."""
    import random
    delay_s = (base_delay_ms * (2 ** attempt) + random.randint(0, 100)) / 1000.0
    await asyncio.sleep(delay_s)


class MbsClient:
    """
    Async MBS Workbench client.

    Parameters
    ----------
    base_url:
        URL of the MBS server. Default: ``http://127.0.0.1:3030``.
    api_key:
        Bearer token for authenticated servers.
    timeout:
        Request timeout in seconds (default: 120).
    max_retries:
        Maximum retry attempts on transient failures (default: 3).
    retry_base_delay_ms:
        Base delay for exponential backoff in milliseconds (default: 500).

    Examples
    --------
    >>> async with MbsClient() as client:
    ...     resp = await client.chat(messages=[{"role": "user", "content": "Hello"}])
    ...     print(resp.choices[0].message.content)
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:3030",
        *,
        api_key: Optional[str] = None,
        timeout: float = 120.0,
        max_retries: int = 3,
        retry_base_delay_ms: int = 500,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._max_retries = max_retries
        self._retry_base_delay_ms = retry_base_delay_ms

        headers: Dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers=headers,
            timeout=httpx.Timeout(timeout),
        )

    async def __aenter__(self) -> "MbsClient":
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    async def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        await self._client.aclose()

    # ── Low-level request with retry ─────────────────────────────────────────

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: Optional[Dict[str, Any]] = None,
        response_model: type[_T],
    ) -> _T:
        last_exc: Exception = RuntimeError("Unreachable")
        for attempt in range(self._max_retries + 1):
            try:
                if method == "GET":
                    resp = await self._client.get(path)
                else:
                    resp = await self._client.post(path, json=json_body or {})

                if resp.is_success:
                    return response_model.model_validate(resp.json())  # type: ignore[attr-defined]

                # Non-2xx
                body: Any
                try:
                    body = resp.json()
                except Exception:
                    body = resp.text

                exc = MbsError(
                    f"MBS API error {resp.status_code}: {str(body)[:200]}",
                    resp.status_code,
                    body,
                )
                if resp.status_code not in _RETRYABLE_STATUS or attempt == self._max_retries:
                    raise exc
                last_exc = exc

            except (httpx.TimeoutException, httpx.NetworkError) as e:
                last_exc = e
                if attempt == self._max_retries:
                    raise

            await _sleep_jitter(attempt, self._retry_base_delay_ms)

        raise last_exc  # should not reach here

    # ── Models ────────────────────────────────────────────────────────────────

    async def models(self) -> ModelsResponse:
        """List available models."""
        return await self._request("GET", "/v1/models", response_model=ModelsResponse)

    async def load_model(self, req: ModelLoadRequest) -> ModelLoadResponse:
        """Load a .gguf model into VRAM."""
        return await self._request(
            "POST", "/v1/models/load",
            json_body=req.model_dump(exclude_none=True),
            response_model=ModelLoadResponse,
        )

    async def unload_model(self) -> ModelUnloadResponse:
        """Unload the current model from VRAM."""
        return await self._request(
            "POST", "/v1/models/unload",
            json_body={},
            response_model=ModelUnloadResponse,
        )

    # ── Chat completions ──────────────────────────────────────────────────────

    async def chat(
        self,
        messages: Sequence[Dict[str, str]] | Sequence[ChatMessage],
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
    ) -> ChatCompletionResponse:
        """
        Send a chat completion request (non-streaming).

        Parameters
        ----------
        messages:
            List of ``{"role": ..., "content": ...}`` dicts or ChatMessage objects.
        """
        # Normalise messages to dicts
        msgs = [
            m.model_dump() if isinstance(m, ChatMessage) else dict(m)
            for m in messages
        ]
        body: Dict[str, Any] = {"messages": msgs, "stream": False}
        if model is not None:
            body["model"] = model
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if stop is not None:
            body["stop"] = stop

        return await self._request(
            "POST", "/v1/chat/completions",
            json_body=body,
            response_model=ChatCompletionResponse,
        )

    async def chat_stream(
        self,
        messages: Sequence[Dict[str, str]] | Sequence[ChatMessage],
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[str]:
        """
        Streaming chat completion. Yields text deltas as they arrive.

        Examples
        --------
        >>> async for delta in client.chat_stream([{"role": "user", "content": "Hi"}]):
        ...     print(delta, end="", flush=True)
        """
        msgs = [
            m.model_dump() if isinstance(m, ChatMessage) else dict(m)
            for m in messages
        ]
        # Use /v1/stream endpoint (SSE)
        prompt = "\n".join(f"{m['role']}: {m['content']}" for m in msgs)
        body: Dict[str, Any] = {"prompt": prompt}
        if model:
            body["model"] = model
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_tokens"] = max_tokens

        return self._sse_stream("/v1/stream", body)

    async def _sse_stream(
        self, path: str, body: Dict[str, Any]
    ) -> AsyncIterator[str]:
        """Internal: POST to an SSE endpoint and yield content deltas."""
        async with aconnect_sse(self._client, "POST", path, json=body) as event_source:
            async for event in event_source.aiter_sse():
                if event.data == "[DONE]":
                    return
                try:
                    chunk = StreamChunk.model_validate_json(event.data)
                    delta = chunk.choices[0].delta.content if chunk.choices else None
                    if delta:
                        yield delta
                except Exception:
                    continue  # skip malformed chunks

    # ── Text completions ──────────────────────────────────────────────────────

    async def complete(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> CompletionResponse:
        """Text completion (non-streaming, /v1/completions)."""
        req = CompletionRequest(
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return await self._request(
            "POST", "/v1/completions",
            json_body=req.model_dump(exclude_none=True),
            response_model=CompletionResponse,
        )

    # ── Embeddings ────────────────────────────────────────────────────────────

    async def embed(
        self,
        input: str | List[str],
        *,
        model: Optional[str] = None,
    ) -> EmbeddingResponse:
        """Generate embeddings for one or more strings."""
        req = EmbeddingRequest(input=input, model=model)
        return await self._request(
            "POST", "/v1/embeddings",
            json_body=req.model_dump(exclude_none=True),
            response_model=EmbeddingResponse,
        )

    # ── Images ────────────────────────────────────────────────────────────────

    async def generate_image(
        self,
        prompt: str,
        *,
        n: int = 1,
        size: str = "512x512",
    ) -> ImageGenerationResponse:
        """Generate an image from a text prompt."""
        req = ImageGenerationRequest(prompt=prompt, n=n, size=size)
        return await self._request(
            "POST", "/v1/images/generations",
            json_body=req.model_dump(exclude_none=True),
            response_model=ImageGenerationResponse,
        )

    # ── Agents ────────────────────────────────────────────────────────────────

    async def run_agent(
        self,
        task: str,
        *,
        model: Optional[str] = None,
        max_iterations: int = 20,
    ) -> AgentRunResponse:
        """Run a ReAct-style agent task and return the result."""
        req = AgentRunRequest(task=task, model=model, max_iterations=max_iterations)
        return await self._request(
            "POST", "/v1/agents/run",
            json_body=req.model_dump(exclude_none=True),
            response_model=AgentRunResponse,
        )

    # ── MCP Tools ─────────────────────────────────────────────────────────────

    async def list_tools(self) -> McpToolsResponse:
        """List registered MCP tools."""
        return await self._request("GET", "/v1/mcp/tools", response_model=McpToolsResponse)

    async def invoke_tool(
        self,
        tool_id: str,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> McpInvokeResponse:
        """Invoke an MCP tool by ID."""
        req = McpInvokeRequest(tool_id=tool_id, arguments=arguments or {})
        return await self._request(
            "POST", "/v1/mcp/tools/invoke",
            json_body=req.model_dump(exclude_none=True),
            response_model=McpInvokeResponse,
        )

    # ── Anthropic pass-through ─────────────────────────────────────────────────

    async def anthropic_messages(
        self, req: AnthropicMessagesRequest
    ) -> AnthropicMessagesResponse:
        """Anthropic-compatible messages endpoint (proxied to local LLM)."""
        return await self._request(
            "POST", "/v1/messages",
            json_body=req.model_dump(exclude_none=True),
            response_model=AnthropicMessagesResponse,
        )

    # ── Pool management ───────────────────────────────────────────────────────

    async def list_pool_models(self) -> PoolListResponse:
        """List all models in the pool."""
        return await self._request("GET", "/v1/pool", response_model=PoolListResponse)

    async def add_pool_model(self, req: PoolAddRequest) -> Dict[str, Any]:
        """Add a model to the pool."""
        resp = await self._client.post("/v1/pool/add", json=req.model_dump(exclude_none=True))
        return resp.json()

    async def remove_pool_model(self, name: str) -> Dict[str, Any]:
        """Remove a model from the pool by name."""
        resp = await self._client.post("/v1/pool/remove", json={"model_name": name})
        return resp.json()

    async def set_pool_strategy(self, strategy: str) -> Dict[str, Any]:
        """Set pool load-balancing strategy (round_robin, least_loaded, random)."""
        resp = await self._client.post("/v1/pool/strategy", json={"strategy": strategy})
        return resp.json()

    async def switch_model(self, req: ModelSwitchRequest) -> Dict[str, Any]:
        """Switch the active model."""
        resp = await self._client.post("/v1/models/switch", json=req.model_dump(exclude_none=True))
        return resp.json()

    async def cpu_fallback(self, model_name: str) -> Dict[str, Any]:
        """Trigger CPU fallback for a model."""
        resp = await self._client.post("/v1/models/fallback", json={"model_name": model_name})
        return resp.json()

    # ── Analytics ──────────────────────────────────────────────────────────────

    async def get_analytics(self) -> AnalyticsSummary:
        """Get analytics summary."""
        return await self._request("GET", "/v1/analytics", response_model=AnalyticsSummary)

    async def reset_analytics(self) -> Dict[str, Any]:
        """Reset analytics counters."""
        resp = await self._client.post("/v1/analytics/reset", json={})
        return resp.json()

    async def set_key_quota(self, api_key: str, daily_tokens: int) -> Dict[str, Any]:
        """Set a per-key daily token quota."""
        resp = await self._client.post(
            "/v1/analytics/quota",
            json={"api_key": api_key, "daily_limit": daily_tokens},
        )
        return resp.json()

    async def set_quota_config(self, config: QuotaConfigUpdate) -> Dict[str, Any]:
        """Update quota configuration."""
        resp = await self._client.post(
            "/v1/analytics/config",
            json=config.model_dump(exclude_none=True),
        )
        return resp.json()

    # ── Webhooks ───────────────────────────────────────────────────────────────

    async def list_webhooks(self) -> WebhookListResponse:
        """List registered webhooks."""
        return await self._request("GET", "/v1/webhooks", response_model=WebhookListResponse)

    async def add_webhook(self, req: WebhookAddRequest) -> Dict[str, Any]:
        """Register a new webhook."""
        resp = await self._client.post("/v1/webhooks", json=req.model_dump())
        return resp.json()

    async def remove_webhook(self, id: str) -> Dict[str, Any]:
        """Remove a webhook by ID."""
        resp = await self._client.post("/v1/webhooks", json={"action": "delete", "id": id})
        return resp.json()

    # ── Health & Queue ─────────────────────────────────────────────────────────

    async def get_health(self) -> HealthResponse:
        """Get server health status."""
        return await self._request("GET", "/v1/health", response_model=HealthResponse)

    async def get_queue_status(self) -> QueueStatus:
        """Get request queue status."""
        return await self._request("GET", "/v1/queue/status", response_model=QueueStatus)

    # ── Batch processing ──────────────────────────────────────────────────────

    async def _run_batch(
        self,
        items: List[_T],
        fn: Any,
        concurrency: int,
    ) -> BatchSummary:
        """Run `fn(item)` for each item, concurrency-limited."""
        sem = asyncio.Semaphore(concurrency)
        results: List[BatchResultItem] = []
        succeeded = 0
        failed = 0

        async def _run_one(item: _T) -> BatchResultItem:
            async with sem:
                try:
                    value = await fn(item)
                    return BatchResultItem(
                        ok=True,
                        value=value.model_dump() if hasattr(value, "model_dump") else value,
                    )
                except Exception as e:
                    return BatchResultItem(ok=False, error=str(e))

        tasks = [asyncio.create_task(_run_one(item)) for item in items]
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            if result.ok:
                succeeded += 1
            else:
                failed += 1

        return BatchSummary(results=results, succeeded=succeeded, failed=failed)

    async def batch_chat(
        self,
        requests: List[ChatCompletionRequest] | List[Dict[str, Any]],
        *,
        concurrency: int = 5,
    ) -> BatchSummary:
        """
        Run multiple chat requests concurrently. Failed requests are captured
        in the result rather than raised.

        Examples
        --------
        >>> summary = await client.batch_chat([
        ...     ChatCompletionRequest(messages=[{"role":"user","content":"Hi"}]),
        ...     ChatCompletionRequest(messages=[{"role":"user","content":"Bye"}]),
        ... ])
        >>> print(f"{summary.succeeded}/{len(summary.results)} succeeded")
        """
        normalized = [
            r if isinstance(r, ChatCompletionRequest) else ChatCompletionRequest(**r)
            for r in requests
        ]

        async def _fn(req: ChatCompletionRequest) -> ChatCompletionResponse:
            return await self.chat(
                req.messages,
                model=req.model,
                temperature=req.temperature,
                max_tokens=req.max_tokens,
            )

        return await self._run_batch(normalized, _fn, concurrency)

    async def batch_embed(
        self,
        inputs: List[str],
        *,
        model: Optional[str] = None,
        concurrency: int = 5,
    ) -> BatchSummary:
        """
        Generate embeddings for multiple strings concurrently.

        Returns a BatchSummary where each result's ``value`` contains the
        embedding list.
        """
        async def _fn(text: str) -> EmbeddingResponse:
            return await self.embed(text, model=model)

        return await self._run_batch(inputs, _fn, concurrency)

    # ── numpy / pandas helpers (optional) ─────────────────────────────────────

    async def embed_numpy(
        self,
        input: str | List[str],
        *,
        model: Optional[str] = None,
    ) -> "Any":
        """
        Like ``embed()`` but returns a NumPy array of shape (n, dim).
        Requires ``pip install mbs-python[numpy]``.
        """
        try:
            import numpy as np
        except ImportError as e:
            raise ImportError(
                "numpy is required: pip install mbs-python[numpy]"
            ) from e

        resp = await self.embed(input, model=model)
        return np.array([obj.embedding for obj in resp.data], dtype=np.float32)

    async def batch_embed_dataframe(
        self,
        texts: "Any",  # pd.Series
        *,
        model: Optional[str] = None,
        concurrency: int = 5,
        column: str = "embedding",
    ) -> "Any":
        """
        Add an ``embedding`` column to a Pandas DataFrame or Series.

        Requires ``pip install mbs-python[pandas]``.

        Parameters
        ----------
        texts:
            A ``pd.Series`` of strings.

        Returns
        -------
        pd.Series of embedding lists (same index as input).
        """
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "pandas is required: pip install mbs-python[pandas]"
            ) from e

        text_list: List[str] = list(texts)
        summary = await self.batch_embed(text_list, model=model, concurrency=concurrency)

        embeddings = []
        for item in summary.results:
            if item.ok and item.value:
                # value is a dict from EmbeddingResponse.model_dump()
                data = item.value.get("data", [])
                emb = data[0]["embedding"] if data else []
                embeddings.append(emb)
            else:
                embeddings.append(None)

        return pd.Series(embeddings, index=texts.index, name=column)

    # ── Health ────────────────────────────────────────────────────────────────

    async def ping(self) -> bool:
        """Returns True if the server is reachable."""
        try:
            await self.models()
            return True
        except Exception:
            return False

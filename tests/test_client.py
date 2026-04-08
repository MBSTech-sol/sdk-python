"""
Unit tests for mbs-python SDK — uses pytest-httpx to mock HTTP responses.
Run: pytest sdk/python/tests/ -v
"""

from __future__ import annotations

import json
from typing import Any, Dict

import pytest
import pytest_httpx
from pytest_httpx import HTTPXMock

from mbs import (
    AgentRunRequest,
    BatchSummary,
    ChatCompletionRequest,
    ChatMessage,
    MbsClient,
    MbsError,
    ModelLoadRequest,
)


BASE_URL = "http://127.0.0.1:3030"


@pytest.fixture
def client() -> MbsClient:
    return MbsClient(base_url=BASE_URL, max_retries=0)


# ── Helpers ───────────────────────────────────────────────────────────────────

def models_payload() -> Dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {"id": "test-model", "object": "model", "created": 1000, "owned_by": "local"}
        ],
    }


def chat_payload(content: str = "Hello!") -> Dict[str, Any]:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1000,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
    }


# ── tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_models(client: MbsClient, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(url=f"{BASE_URL}/v1/models", json=models_payload())
    resp = await client.models()
    assert len(resp.data) == 1
    assert resp.data[0].id == "test-model"


@pytest.mark.asyncio
async def test_chat(client: MbsClient, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url=f"{BASE_URL}/v1/chat/completions",
        json=chat_payload("World"),
    )
    resp = await client.chat([{"role": "user", "content": "Hello"}])
    assert resp.choices[0].message.content == "World"
    assert resp.usage.total_tokens == 15


@pytest.mark.asyncio
async def test_chat_with_chat_message(client: MbsClient, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(url=f"{BASE_URL}/v1/chat/completions", json=chat_payload())
    msg = ChatMessage(role="user", content="Hi")
    resp = await client.chat([msg])
    assert resp.choices[0].finish_reason == "stop"


@pytest.mark.asyncio
async def test_load_model(client: MbsClient, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url=f"{BASE_URL}/v1/models/load",
        json={"success": True, "model_name": "llama-3", "message": "Loaded"},
    )
    req = ModelLoadRequest(path="/models/llama-3.gguf", name="llama-3")
    resp = await client.load_model(req)
    assert resp.success is True
    assert resp.model_name == "llama-3"


@pytest.mark.asyncio
async def test_unload_model(client: MbsClient, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url=f"{BASE_URL}/v1/models/unload",
        json={"success": True, "message": "Unloaded"},
    )
    resp = await client.unload_model()
    assert resp.success is True


@pytest.mark.asyncio
async def test_embed(client: MbsClient, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url=f"{BASE_URL}/v1/embeddings",
        json={
            "object": "list",
            "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]}],
            "model": "test-model",
            "usage": {"prompt_tokens": 3, "completion_tokens": 0, "total_tokens": 3},
        },
    )
    resp = await client.embed("test input")
    assert len(resp.data[0].embedding) == 3
    assert resp.data[0].embedding[0] == pytest.approx(0.1)


@pytest.mark.asyncio
async def test_generate_image(client: MbsClient, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url=f"{BASE_URL}/v1/images/generations",
        json={
            "created": 1000,
            "data": [{"url": "http://example.com/img.png"}],
            "revised_prompt": "A photo of a cat",
        },
    )
    resp = await client.generate_image("A cat")
    assert resp.revised_prompt == "A photo of a cat"
    assert resp.data[0].url is not None


@pytest.mark.asyncio
async def test_run_agent(client: MbsClient, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url=f"{BASE_URL}/v1/agents/run",
        json={
            "success": True,
            "result": "Task done",
            "iterations": 3,
            "reasoning_steps": ["step1", "step2", "step3"],
        },
    )
    resp = await client.run_agent("Do something")
    assert resp.success is True
    assert resp.iterations == 3
    assert len(resp.reasoning_steps) == 3


@pytest.mark.asyncio
async def test_list_tools(client: MbsClient, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url=f"{BASE_URL}/v1/mcp/tools",
        json={
            "tools": [
                {"id": "tool1", "name": "Tool One", "description": "Does stuff"},
            ]
        },
    )
    resp = await client.list_tools()
    assert len(resp.tools) == 1
    assert resp.tools[0].id == "tool1"


@pytest.mark.asyncio
async def test_invoke_tool(client: MbsClient, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url=f"{BASE_URL}/v1/mcp/tools/invoke",
        json={"success": True, "result": {"answer": 42}},
    )
    resp = await client.invoke_tool("tool1", {"x": 1})
    assert resp.success is True
    assert resp.result == {"answer": 42}


@pytest.mark.asyncio
async def test_mbs_error_on_4xx(client: MbsClient, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url=f"{BASE_URL}/v1/models",
        status_code=401,
        json={"error": {"message": "Unauthorized", "type": "auth_error"}},
    )
    with pytest.raises(MbsError) as exc_info:
        await client.models()
    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_batch_chat(client: MbsClient, httpx_mock: HTTPXMock) -> None:
    # Mock two identical responses
    for _ in range(2):
        httpx_mock.add_response(
            url=f"{BASE_URL}/v1/chat/completions",
            json=chat_payload("ok"),
        )
    reqs = [
        ChatCompletionRequest(
            messages=[ChatMessage(role="user", content=f"q{i}")]
        )
        for i in range(2)
    ]
    summary: BatchSummary = await client.batch_chat(reqs, concurrency=2)
    assert summary.succeeded == 2
    assert summary.failed == 0


@pytest.mark.asyncio
async def test_ping_true(client: MbsClient, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(url=f"{BASE_URL}/v1/models", json=models_payload())
    assert await client.ping() is True


@pytest.mark.asyncio
async def test_ping_false(client: MbsClient, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url=f"{BASE_URL}/v1/models", status_code=500, json={}
    )
    assert await client.ping() is False

"""
mbs-python — MBS Workbench Python SDK
Type definitions (Pydantic v2 models matching the server wire format)
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


# ── Models ─────────────────────────────────────────────────────────────────────


class Model(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str


class ModelsResponse(BaseModel):
    object: Literal["list"] = "list"
    data: List[Model]


class ModelLoadRequest(BaseModel):
    path: str
    name: Optional[str] = None
    gpu_layers: Optional[int] = None


class ModelLoadResponse(BaseModel):
    success: bool
    model_name: str
    message: str


class ModelUnloadResponse(BaseModel):
    success: bool
    message: str


# ── Chat ───────────────────────────────────────────────────────────────────────

MessageRole = Literal["system", "user", "assistant"]


class ChatMessage(BaseModel):
    role: MessageRole
    content: str


class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessage]
    model: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: Optional[bool] = None
    stop: Optional[List[str]] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[ChatChoice]
    usage: UsageInfo


# ── Completions ────────────────────────────────────────────────────────────────


class CompletionRequest(BaseModel):
    prompt: str
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: Optional[bool] = None
    stop: Optional[List[str]] = None


class CompletionChoice(BaseModel):
    index: int
    text: str
    finish_reason: str


class CompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: UsageInfo


# ── Embeddings ─────────────────────────────────────────────────────────────────


class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    model: Optional[str] = None


class EmbeddingObject(BaseModel):
    object: Literal["embedding"] = "embedding"
    index: int
    embedding: List[float]


class EmbeddingResponse(BaseModel):
    object: Literal["list"] = "list"
    data: List[EmbeddingObject]
    model: str
    usage: UsageInfo


# ── Streaming ──────────────────────────────────────────────────────────────────


class StreamDelta(BaseModel):
    content: Optional[str] = None
    role: Optional[str] = None


class StreamChoice(BaseModel):
    index: int
    delta: StreamDelta
    finish_reason: Optional[str] = None


class StreamChunk(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[StreamChoice]


# ── Images ─────────────────────────────────────────────────────────────────────


class ImageGenerationRequest(BaseModel):
    prompt: str
    n: Optional[int] = Field(default=1, ge=1, le=10)
    size: Optional[str] = "512x512"


class ImageData(BaseModel):
    url: Optional[str] = None
    b64_json: Optional[str] = None


class ImageGenerationResponse(BaseModel):
    created: int
    data: List[ImageData]
    revised_prompt: Optional[str] = None


# ── Agents ─────────────────────────────────────────────────────────────────────


class AgentRunRequest(BaseModel):
    task: str
    model: Optional[str] = None
    max_iterations: Optional[int] = Field(default=20, ge=1, le=50)


class AgentRunResponse(BaseModel):
    success: bool
    result: str
    iterations: int
    reasoning_steps: List[str] = Field(default_factory=list)


# ── MCP Tools ──────────────────────────────────────────────────────────────────


class McpTool(BaseModel):
    id: str
    name: str
    description: str
    parameters: Optional[Dict[str, Any]] = None


class McpToolsResponse(BaseModel):
    tools: List[McpTool]


class McpInvokeRequest(BaseModel):
    tool_id: str
    arguments: Optional[Dict[str, Any]] = None


class McpInvokeResponse(BaseModel):
    success: bool
    result: Any
    error: Optional[str] = None


# ── Anthropic ──────────────────────────────────────────────────────────────────


class AnthropicMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class AnthropicMessagesRequest(BaseModel):
    model: str
    max_tokens: int
    messages: List[AnthropicMessage]
    system: Optional[str] = None
    temperature: Optional[float] = None


class AnthropicContentBlock(BaseModel):
    type: Literal["text"]
    text: str


class AnthropicUsage(BaseModel):
    input_tokens: int
    output_tokens: int


class AnthropicMessagesResponse(BaseModel):
    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    content: List[AnthropicContentBlock]
    model: str
    stop_reason: str
    usage: AnthropicUsage


# ── Error ───────────────────────────────────────────────────────────────────────


class MbsError(Exception):
    """Raised when the MBS server returns a non-2xx response."""

    def __init__(self, message: str, status_code: int, body: Any) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body

        # Extract structured error fields
        self.error_type: Optional[str] = None
        self.retry_after_secs: Optional[int] = None
        if isinstance(body, dict) and "error" in body:
            err = body["error"]
            if isinstance(err, dict):
                self.error_type = err.get("type")
                self.retry_after_secs = err.get("retry_after_secs")

    def __str__(self) -> str:
        return f"MbsError(status={self.status_code}, type={self.error_type}): {super().__str__()}"


# ── Batch types ────────────────────────────────────────────────────────────────


class BatchResultItem(BaseModel):
    ok: bool
    value: Optional[Any] = None
    error: Optional[str] = None


class BatchSummary(BaseModel):
    results: List[BatchResultItem]
    succeeded: int
    failed: int


# ── Pool ───────────────────────────────────────────────────────────────────────


class PoolModel(BaseModel):
    name: str
    path: str
    loaded: bool
    requests_total: int = 0
    active_requests: int = 0
    errors_total: int = 0


class PoolAddRequest(BaseModel):
    name: str
    path: str
    gpu_layers: Optional[int] = None


class ModelSwitchRequest(BaseModel):
    model_name: str
    gpu_layers: Optional[int] = None


class PoolListResponse(BaseModel):
    models: List[PoolModel]


# ── Analytics ──────────────────────────────────────────────────────────────────


class KeyUsage(BaseModel):
    key: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    requests: int
    daily_limit: Optional[int] = None


class RouteStats(BaseModel):
    route: str
    requests: int
    avg_latency_ms: Optional[float] = None


class AnalyticsSummary(BaseModel):
    prompt_tokens_total: int
    completion_tokens_total: int
    cost_usd_total: float
    active_keys: int
    keys: List[KeyUsage] = Field(default_factory=list)
    routes: List[RouteStats] = Field(default_factory=list)


class QuotaConfigUpdate(BaseModel):
    cost_per_prompt_token: Optional[float] = None
    cost_per_completion_token: Optional[float] = None
    default_daily_limit: Optional[int] = None


# ── Webhooks ───────────────────────────────────────────────────────────────────


class WebhookItem(BaseModel):
    id: str
    url: str
    events: List[str]
    deliveries_total: int = 0
    failures_total: int = 0


class WebhookListResponse(BaseModel):
    webhooks: List[WebhookItem]


class WebhookAddRequest(BaseModel):
    url: str
    events: List[str]


# ── Health & Queue ─────────────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float
    model_loaded: bool
    connected_clients: int


class QueueStatus(BaseModel):
    depth: int
    processed_total: int
    max_size: int
    default_priority: int

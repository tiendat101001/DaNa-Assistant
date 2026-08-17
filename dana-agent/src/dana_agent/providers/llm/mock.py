"""Deterministic mock LLM, shaped to exercise the brain's agent/tools loop:

- if the most recent message is a TOOL result, answer with final text (this is
  what lets the loop terminate after a tool runs);
- else if a tool is offered whose name appears in the last user message, emit a
  single tool call;
- otherwise stream a text reply echoing the user.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from dana_agent.config import LLMConfig
from dana_agent.providers.base import Capabilities, LLMProvider
from dana_agent.providers.registry import register
from dana_agent.types import LLMChunk, Message, Role, Tool, ToolCall


@register("llm", "mock")
class MockLLM(LLMProvider):
    def __init__(self, cfg: LLMConfig) -> None:
        self._cfg = cfg

    async def stream(
        self, messages: list[Message], tools: list[Tool] | None = None
    ) -> AsyncIterator[LLMChunk]:
        last_user = next((m.content for m in reversed(messages) if m.role is Role.USER), "")
        last_is_tool_result = bool(messages) and messages[-1].role is Role.TOOL
        if tools and not last_is_tool_result:
            hit = next((t for t in tools if t.name in last_user.lower()), None)
            if hit is not None:
                yield LLMChunk(
                    tool_calls=(ToolCall(id="call_1", name=hit.name, arguments={}),),
                )
                yield LLMChunk(done=True)
                return
        reply = f"mock reply to: {last_user}".strip()
        for word in reply.split():
            yield LLMChunk(text=word + " ")
        yield LLMChunk(done=True)

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(
            streaming=True,
            languages=frozenset({"en", "vi"}),
            self_hosted=True,
            tool_calling=True,
            latency_class="fast",
        )

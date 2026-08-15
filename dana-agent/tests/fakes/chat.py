"""ScriptedChatModel — a fake LangChain chat model for offline brain tests.

Returns a scripted sequence of turns, each either a text answer or a single tool
call, and streams them (word chunks / a tool-call chunk) so LangGraph's
``astream_events`` emits ``on_chat_model_stream`` just like a real model. The
last turn repeats, so a single ``{"tool": ...}`` turn drives a runaway loop for
the step-limit test.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from pydantic import PrivateAttr


class ScriptedChatModel(BaseChatModel):  # type: ignore[misc]  # BaseChatModel is Any under strict
    turns: list[dict[str, Any]]
    _pos: int = PrivateAttr(default=0)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _next(self) -> AIMessage:
        turn = self.turns[min(self._pos, len(self.turns) - 1)]
        self._pos += 1
        if "tool" in turn:
            return AIMessage(
                content="",
                tool_calls=[
                    {"name": turn["tool"], "args": turn.get("args", {}), "id": f"call_{self._pos}"}
                ],
            )
        return AIMessage(content=turn["text"])

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._next())])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        msg = self._next()
        if msg.tool_calls:
            call = msg.tool_calls[0]
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content="",
                    tool_call_chunks=[
                        {
                            "name": call["name"],
                            "args": json.dumps(call["args"]),
                            "id": call["id"],
                            "index": 0,
                        }
                    ],
                )
            )
        else:
            for word in str(msg.content).split():
                yield ChatGenerationChunk(message=AIMessageChunk(content=word + " "))

    def bind_tools(
        self,
        tools: Sequence[Any],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        return self  # scripted model ignores binding

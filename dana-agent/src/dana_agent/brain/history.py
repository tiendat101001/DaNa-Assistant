"""Long-conversation memory strategy (langgraph-agent-patterns skill).

Keeps the model's context bounded so a long session doesn't grow unbounded:

- ``trim``      — send only the last N messages to the model (stored history is
  untouched; this only bounds what each turn sees).
- ``summarize`` — once history crosses a threshold, condense the older messages
  into a running summary and prune them from state (RemoveMessage), injecting the
  summary into the system prompt.
- ``none``      — send the whole history every turn.

Pure helpers here; the graph nodes in ``graph.py`` apply them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.messages import BaseMessage, SystemMessage, trim_messages

Strategy = Literal["none", "trim", "summarize"]


@dataclass(frozen=True, slots=True)
class HistoryPolicy:
    strategy: Strategy = "trim"
    trigger_messages: int = 20
    keep_last: int = 8


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            p if isinstance(p, str) else str(p.get("text", ""))
            for p in content
            if isinstance(p, str | dict)
        )
    return str(content)


def select_conversation(messages: list[BaseMessage], policy: HistoryPolicy) -> list[BaseMessage]:
    """The conversation messages to send to the model this turn. Only the
    ``trim`` strategy bounds them; the others send everything (summarize has
    already pruned state)."""
    if policy.strategy == "trim" and len(messages) > policy.keep_last:
        trimmed: list[BaseMessage] = trim_messages(
            messages,
            strategy="last",
            token_counter=len,  # count messages, not tokens
            max_tokens=policy.keep_last,
            include_system=False,
            start_on="human",
        )
        return trimmed
    return messages


def build_system(system_prompt: str, summary: str) -> SystemMessage:
    """System message, with the running summary appended when present."""
    if summary:
        return SystemMessage(content=f"{system_prompt}\n\nConversation summary so far:\n{summary}")
    return SystemMessage(content=system_prompt)


def messages_to_prune(messages: list[BaseMessage], keep_last: int) -> list[BaseMessage]:
    """The prefix of old messages to summarize and drop, keeping the last
    ``keep_last`` — and never leaving a kept window that opens on a tool result
    (which would be an orphaned ToolMessage)."""
    if len(messages) <= keep_last:
        return []
    cut = len(messages) - keep_last
    while cut < len(messages) and getattr(messages[cut], "type", "") == "tool":
        cut += 1
    return messages[:cut]


def summary_prompt(existing: str, old: list[BaseMessage]) -> str:
    convo = "\n".join(f"{getattr(m, 'type', '?')}: {_text(m.content)}" for m in old)
    base = f"Existing summary:\n{existing}\n\n" if existing else ""
    return (
        base
        + "Update the running summary to incorporate the conversation below. "
        + "Keep it brief and factual; return only the updated summary.\n\n"
        + convo
    )

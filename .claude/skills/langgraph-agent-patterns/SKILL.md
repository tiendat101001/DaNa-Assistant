---
name: langgraph-agent-patterns
description: Idiomatic patterns for building LLM agents with LangGraph and LangChain — typed graph state, tool nodes, the tool-calling loop, checkpointing/persistence for memory, streaming, human-in-the-loop interrupts, and safe tool design. Use this skill whenever the user is building an agent, "brain", or orchestration layer with LangGraph or LangChain; whenever they describe a stateful multi-step LLM workflow with tools, memory, or conditional routing; and whenever you are about to design the core reasoning/tool-execution loop of an assistant. Consult it before writing the agent graph so it is a proper state machine with persistence and safe tool handling rather than an ad-hoc chain.
---

# LangGraph Agent Patterns

LangGraph models an agent as a **graph with typed state** — nodes transform state, edges (including conditional ones) route between them. This is the right mental model for an assistant's brain: it makes control flow explicit, inspectable, resumable, and streamable, unlike an opaque chain. This skill is the idiomatic shape; check the current LangGraph docs for exact API names, which move.

## Design the state first

The `State` is the single object flowing through the graph. Design it before nodes.

```python
from typing import Annotated, TypedDict
from langgraph.graph import add_messages

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]   # reducer appends, not overwrites
    # add only what nodes actually read/write: pending action, retrieved docs, flags
```

- Use **reducers** (like `add_messages`) for accumulating fields so parallel/successive nodes merge instead of clobbering.
- Keep state **minimal and typed** — every field should be read or written by some node. Bloated state is the main source of confusing agent bugs.

## The core agent loop

The canonical agent is two nodes and a conditional edge:

```
        ┌────────────┐
        │   agent    │  (LLM decides: answer or call tools)
        └─────┬──────┘
       tools? │ else
        ┌─────▼──────┐
        │   tools    │──► back to agent
        └────────────┘
```

- The **agent node** calls the model (with tools bound) and appends its message.
- A **conditional edge** inspects the last message: if it contains tool calls, route to the tools node; otherwise end.
- The **tools node** executes the requested tools and appends results, then routes back to the agent.
- Prefer the prebuilt tool-execution node / prebuilt ReAct agent to start; drop to a hand-built graph when you need custom routing, guardrails, or extra nodes (retrieval, validation, human approval).

## Tools: typed, documented, and safe

Tools are where an agent touches the world — design them carefully (and see the `safe-computer-use` skill for anything that acts on a browser/OS).

- Define tools with clear names, typed arguments, and **docstrings that read as instructions to the model** — the description is the model's only guide to when/how to use it.
- Keep each tool **narrow** and validate its arguments before doing anything. Return structured results (including structured errors) so the agent can recover.
- Gate side-effecting tools: read-only tools run freely; destructive tools go through confirmation / human-in-the-loop (below).
- Do not expose one generic "run anything" tool; expose specific capabilities.

## Persistence = memory (checkpointing)

Attach a **checkpointer** so graph state is saved after every step, keyed by a thread/conversation id.

```python
graph = builder.compile(checkpointer=checkpointer)
graph.invoke(input, config={"configurable": {"thread_id": conversation_id}})
```

This gives you, for free: conversation memory across turns, resume-after-crash, and time-travel debugging. Use an in-memory checkpointer for dev and a durable one (SQLite/Postgres/etc.) for anything real. For long conversations, **summarize or trim** old messages in a node rather than letting the context grow unbounded — decide a memory strategy explicitly.

Distinguish **short-term** (thread state via checkpointer) from **long-term** memory (facts persisted across threads in a store you query and inject) and implement them separately.

## Streaming (required for voice/UX)

Stream, don't block. LangGraph can stream token-by-token from the LLM and emit state updates per node.

- Stream **LLM tokens** so a UI or TTS can start immediately (essential when the brain feeds a voice pipeline).
- Stream **node/state updates** so a UI can show what the agent is doing ("searching…", "calling tool…").
- Surface intermediate steps to the user; a silent agent that thinks for 8 seconds feels broken.

## Human-in-the-loop (interrupts)

For approvals, corrections, or destructive actions, use LangGraph's **interrupt** capability to pause the graph, surface the pending action, and resume with a human decision.

```
agent → [proposes destructive tool] → INTERRUPT → human approves/edits → tools → agent
```

Because state is checkpointed, the graph can pause indefinitely and resume exactly where it stopped. This is the clean way to implement the confirmation gates from `safe-computer-use`.

## Structure and observability

- **Compose, don't sprawl.** For complex assistants use subgraphs / multi-agent handoff (a router that delegates to specialist graphs) instead of one giant graph. Keep each node single-purpose.
- Put nodes in modules and keep the graph wiring in one place (a `build_graph()` function) — the graph definition should read like a diagram.
- Add tracing/observability (LangSmith or your own structured logs) from day one: capture each node's input/output, tool calls, and latency. Agents are hard to debug blind.
- Set **recursion/step limits** so a looping agent stops instead of running forever; handle hitting the limit gracefully.

## Testing agents

- Unit-test **tools** independently of the LLM (they're just functions).
- Test **routing logic** with a fake/stub model that returns canned tool calls, so you verify the graph's control flow deterministically without real model calls.
- Test **checkpoint/resume**: run partway, reload from the checkpointer, continue, and assert state continuity.
- Keep the model provider swappable (see `provider-plugin-architecture`) so tests use a fake and prod uses a real LLM.

## Review checklist

- [ ] State is typed, minimal, and uses reducers for accumulating fields.
- [ ] Core loop is an explicit agent/tools graph with a conditional edge, not an ad-hoc chain.
- [ ] Tools are narrow, typed, documented for the model, and validate their inputs.
- [ ] A checkpointer provides memory + resumability; a memory/trim strategy is chosen.
- [ ] LLM tokens and node updates stream to the UI/TTS.
- [ ] Destructive actions pause via human-in-the-loop interrupts.
- [ ] Recursion/step limits set; tracing/observability wired in.
- [ ] Tools, routing, and checkpoint/resume are tested with a fake model.

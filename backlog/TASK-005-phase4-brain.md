# TASK-005 — Phase 4: LangGraph brain

- **Status:** TODO
- **Priority:** P2
- **Blocked by:** TASK-004 (needs a conversation loop to feed).
- **Source:** docs/PLAN.md §3 (Phase 4), langgraph-agent-patterns skill.

## Goal

The agent brain as a LangGraph state graph: typed state, agent/tools loop with a
conditional edge, checkpointing for memory, token streaming into TTS, and
interrupt-based confirmation gates for destructive tools.

## Acceptance criteria

- [ ] Typed `AgentState` (messages via reducer); minimal fields.
- [ ] Agent/tools graph with conditional edge (not an ad-hoc chain).
- [ ] SQLite checkpointer keyed by conversation session (thread_id); resume test.
- [ ] Message trim/summarize strategy for long conversations.
- [ ] LLM tokens stream to the voice pipeline; no full-completion waits.
- [ ] Destructive tools pause via `interrupt`; resume on human/UI approval.
- [ ] Recursion/step limits set; tracing/observability wired.
- [ ] Routing tested with a fake model returning canned tool calls.

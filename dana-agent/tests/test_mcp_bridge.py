"""MCP bridge with a fake ClientSession (no mcp SDK / subprocess needed)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from dana_agent.actions.audit import AuditLog
from dana_agent.actions.gate import ActionGate, GateDecision
from dana_agent.brain.tools import build_gated_tools
from dana_agent.config import MCPServerConfig
from dana_agent.providers.actions.mcp import MAX_OUTPUT_CHARS, MCPActions
from dana_agent.types import Action, Reversibility


def tool(name: str, schema: dict[str, Any] | None = None, **ann: Any) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        description=f"{name} desc",
        inputSchema=schema or {"type": "object", "properties": {}},
        annotations=SimpleNamespace(**ann) if ann is not None and ann else None,
    )


class FakeSession:
    def __init__(self, text: str = "ok", is_error: bool = False) -> None:
        self.text = text
        self.is_error = is_error
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def call_tool(self, name: str, arguments: dict[str, object]) -> Any:
        self.calls.append((name, arguments))
        return SimpleNamespace(
            content=[SimpleNamespace(text=self.text), SimpleNamespace(data="img")],
            isError=self.is_error,
        )


def server(**kw: Any) -> MCPServerConfig:
    return MCPServerConfig(name=kw.pop("name", "my-srv"), command="x", **kw)


def make(tools: list[Any], session: FakeSession | None = None, **kw: Any) -> MCPActions:
    m = MCPActions([])
    m.register_tools(server(**kw), session or FakeSession(), tools)
    return m


def cls(m: MCPActions, name: str) -> Reversibility:
    return m.classify(Action(tool=name, arguments={}))


def test_namespacing_and_sanitize() -> None:
    m = make([tool("get.thing"), tool("x" * 100)])
    names = [s.name for s in m.tool_specs()]
    assert names[0] == "mcp_my_srv_get_thing"
    assert len(names[1]) == 64
    assert all(n.replace("_", "").isalnum() for n in names)


def test_name_collision_deduped() -> None:
    m = make([tool("a.b"), tool("a_b")])
    names = [s.name for s in m.tool_specs()]
    assert len(set(names)) == 2


def test_classification_matrix() -> None:
    m = make(
        [
            tool("ro", readOnlyHint=True),
            tool("rev", destructiveHint=False),
            tool("dflt", readOnlyHint=False),
            tool("none"),
            tool("both", readOnlyHint=True, destructiveHint=True),
        ]
    )
    assert cls(m, "mcp_my_srv_ro") is Reversibility.READ_ONLY
    assert cls(m, "mcp_my_srv_rev") is Reversibility.REVERSIBLE
    assert cls(m, "mcp_my_srv_dflt") is Reversibility.DESTRUCTIVE
    assert cls(m, "mcp_my_srv_none") is Reversibility.DESTRUCTIVE
    assert cls(m, "mcp_my_srv_both") is Reversibility.READ_ONLY
    assert cls(m, "mcp_other_x") is Reversibility.OUT_OF_SCOPE


def test_confirm_all_overrides_hints() -> None:
    m = make([tool("ro", readOnlyHint=True)], trust="confirm_all")
    assert cls(m, "mcp_my_srv_ro") is Reversibility.DESTRUCTIVE


def test_allowed_tools_filter() -> None:
    m = make([tool("a", readOnlyHint=True), tool("b", readOnlyHint=True)], allowed_tools=["a"])
    assert [s.name for s in m.tool_specs()] == ["mcp_my_srv_a"]
    assert cls(m, "mcp_my_srv_b") is Reversibility.OUT_OF_SCOPE


def test_schema_to_toolspec() -> None:
    schema = {
        "type": "object",
        "properties": {
            "q": {"type": "string", "description": "query"},
            "n": {"type": "integer"},
            "x": {"type": "number"},
            "flag": {"type": "boolean"},
            "opts": {"type": "object", "description": "options"},
            "tags": {"type": "array"},
        },
        "required": ["q"],
    }
    (spec,) = make([tool("search", schema)]).tool_specs()
    a = {x.name: x for x in spec.args}
    assert a["q"].annotation is str and a["q"].required and a["q"].description == "query"
    assert a["n"].annotation is int and not a["n"].required
    assert a["x"].annotation is float
    assert a["flag"].annotation is bool
    assert a["opts"].annotation is str and "JSON" in a["opts"].description
    assert a["tags"].annotation is str and "array" in a["tags"].description
    assert spec.description.startswith("[MCP:my-srv]")


async def test_execute_ok_and_json_arg_decoding() -> None:
    sess = FakeSession("hello")
    schema = {"type": "object", "properties": {"opts": {"type": "object"}, "q": {"type": "string"}}}
    m = make([tool("t", schema, readOnlyHint=True)], sess)
    res = await m.execute(Action("mcp_my_srv_t", {"opts": '{"a": 1}', "q": "{not json"}))
    assert res.ok and "hello" in res.output and "untrusted" in res.output
    assert sess.calls == [("t", {"opts": {"a": 1}, "q": "{not json"})]


async def test_execute_error_flag() -> None:
    m = make([tool("t")], FakeSession("boom", is_error=True))
    res = await m.execute(Action("mcp_my_srv_t", {}))
    assert not res.ok and res.error == "boom"


async def test_execute_exception_and_unknown() -> None:
    class Bad:
        async def call_tool(self, *a: Any) -> Any:
            raise RuntimeError("pipe closed")

    m = MCPActions([])
    m.register_tools(server(), Bad(), [tool("t")])
    assert not (await m.execute(Action("mcp_my_srv_t", {}))).ok
    assert not (await m.execute(Action("nope", {}))).ok


async def test_truncation() -> None:
    m = make([tool("t")], FakeSession("a" * 10_000))
    res = await m.execute(Action("mcp_my_srv_t", {}))
    assert len(res.output) < MAX_OUTPUT_CHARS + 300
    assert "truncated" in res.output


async def test_through_gate_and_tools(tmp_path: Any) -> None:
    sess = FakeSession("done")
    m = make([tool("read", readOnlyHint=True), tool("wipe")], sess)
    gate = ActionGate(m, policy="standard", audit=AuditLog())
    assert gate.decide(Action("mcp_my_srv_read", {}))[0] is GateDecision.ALLOW
    assert gate.decide(Action("mcp_my_srv_wipe", {}))[0] is GateDecision.CONFIRM
    assert gate.decide(Action("mcp_zzz", {}))[0] is GateDecision.REFUSE
    tools = build_gated_tools(m.tool_specs(), gate)
    assert {t.name for t in tools} == {"mcp_my_srv_read", "mcp_my_srv_wipe"}
    read = next(t for t in tools if t.name == "mcp_my_srv_read")
    out = await read.ainvoke({})
    assert "done" in out


async def test_disabled_server_and_aclose() -> None:
    m = MCPActions([server(enabled=False)])
    await m.connect()  # nothing enabled -> no mcp import, no error
    assert m.tool_specs() == []
    await m.aclose()


async def test_connect_failure_isolated(monkeypatch: pytest.MonkeyPatch) -> None:
    m = MCPActions([server()])

    async def boom(self: Any, s: Any) -> Any:
        raise OSError("no such command")

    monkeypatch.setattr(MCPActions, "_open", boom)
    await m.connect()
    assert m.tool_specs() == []


def test_snake_case_annotations_mcp2() -> None:
    def t2(name: str, **ann: Any) -> SimpleNamespace:
        return SimpleNamespace(
            name=name, description="d", inputSchema={}, annotations=SimpleNamespace(**ann)
        )

    m = make(
        [
            t2("ro", read_only_hint=True),
            t2("rev", destructive_hint=False),
            t2("dflt", read_only_hint=False, destructive_hint=None),
        ]
    )
    assert cls(m, "mcp_my_srv_ro") is Reversibility.READ_ONLY
    assert cls(m, "mcp_my_srv_rev") is Reversibility.REVERSIBLE
    assert cls(m, "mcp_my_srv_dflt") is Reversibility.DESTRUCTIVE

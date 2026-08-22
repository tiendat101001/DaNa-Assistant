"""Composite action provider: browser + Windows OS in one gated surface.

Real voice sessions need both at once — "pause the video" (browser tab) and
"volume 30" (system mixer) in the same conversation. The registry selects ONE
action provider, so this class composes the two bounded providers and
dispatches by tool-name prefix (``browser_*`` / ``os_*``). It adds no tools
and no logic of its own: each delegate keeps its own allowlists, validation
and honest classification; the gate still sees a single provider.
"""

from __future__ import annotations

from typing import Any

from dana_agent.config import ActionsConfig
from dana_agent.providers.actions.browser import BrowserActions
from dana_agent.providers.actions.os_win import WindowsOSActions
from dana_agent.providers.base import ActionProvider, Capabilities
from dana_agent.providers.registry import register
from dana_agent.types import Action, ActionResult, Reversibility


@register("actions", "browser_os")
class BrowserOSActions(ActionProvider):
    """Browser tools + (explicitly enabled) OS tools, one dispatch table."""

    def __init__(self, cfg: ActionsConfig) -> None:
        self._cfg = cfg
        self._browser = BrowserActions(cfg)
        self._os = WindowsOSActions(cfg)

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(
            streaming=False,
            languages=frozenset({"en", "vi"}),
            latency_class="fast",  # OS part is instant; browser part is batch
        )

    # Delegated surface used by tests / the tray to reach the browser part.
    @property
    def browser(self) -> BrowserActions:
        return self._browser

    def classify(self, action: Action) -> Reversibility:
        if action.tool.startswith("os_"):
            return self._os.classify(action)
        return self._browser.classify(action)

    async def execute(self, action: Action) -> ActionResult:
        if action.tool.startswith("os_"):
            return await self._os.execute(action)
        return await self._browser.execute(action)

    async def aclose(self) -> None:
        await self._browser.aclose()

    def __getattr__(self, name: str) -> Any:
        # Transparent passthrough for browser internals used by the control
        # plane (e.g. storage-state export on shutdown).
        return getattr(self._browser, name)

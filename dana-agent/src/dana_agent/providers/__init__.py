"""Provider packages. Importing this package self-registers every built-in
provider (import side effects), so the registry is populated before any factory
call. Heavy/vendor SDKs are imported lazily inside each real provider module, so
a missing extra breaks only that one provider, not the whole app.
"""

from dana_agent.providers import (  # noqa: F401  (registration side effects)
    actions,
    llm,
    s2s,
    stt,
    tts,
    vad,
    wake,
)

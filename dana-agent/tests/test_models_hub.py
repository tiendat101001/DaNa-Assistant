"""HubClient against a mocked HuggingFace API."""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from dana_agent.config import ModelsConfig
from dana_agent.models import HubAuthError, HubClient, HubUnavailableError, classify_task
from dana_agent.models.errors import ModelNotFoundError
from dana_agent.models.hub import COMPANIONS, companion_files, recommended_files

HUB = "https://hub.test"


def _f(path: str, size: int, lfs: bool = True) -> dict[str, Any]:
    d: dict[str, Any] = {"type": "file", "path": path, "size": 130 if lfs else size}
    if lfs:
        d["lfs"] = {"size": size}
    return d


WHISPER_TREE = [
    _f("README.md", 13651, lfs=False),
    _f("whisper_tiny_30s_f32.tflite", 150_979_184),
    _f("whisper_tiny_30s_f32_MediaTek_MT6877.tflite", 158_197_360),
    _f("whisper_tiny_30s_f32_Qualcomm_SM8650.tflite", 119_859_536),
    _f("whisper_tiny_30s_i8.litertlm", 41_804_416),
    _f("whisper_tiny_30s_i8.tflite", 41_116_288),
]
KITTEN_TREE = [
    _f("README.md", 14724, lfs=False),
    _f("bench_inputs.npz", 1_340_256),
    _f("g2p/dp_g2p_matcha_fp16.tflite", 25_785_872),
    _f("g2p/g2p_dict.txt.gz", 1_762_038),
    _f("g2p/g2p_meta.json", 1904, lfs=False),
    _f("g2p/symbols.json", 1613, lfs=False),
    _f("kitten_predictor.tflite", 33_819_808),
    _f("kitten_predictor_fp16.tflite", 16_990_756),
    _f("kitten_prosody_fp16.tflite", 1_697_636),
    _f("kitten_vocoder_fp16.tflite", 13_367_840),
    _f("kitten_vocoder_static80.tflite", 33_779_816),
    _f("samples/litert_golden.wav", 218_444),
    _f("voices.npz", 3_278_902),
]
KOKORO_TREE = [
    _f("README.md", 32889, lfs=False),
    _f("kokoro_int8_fp16.litertlm", 294_783_298),
    _f("kokoro_int8_gpu_fp16.litertlm", 294_750_530),
    _f("legacy/istft_Wi_f32.bin", 880, lfs=False),
    _f("legacy/istft_Wr_f32.bin", 880, lfs=False),
    _f("legacy/kokoro_82m_fixedlen_fp32.tflite", 337_596_032),
    _f("legacy/kokoro_predictor.tflite", 90_741_756),
    _f("legacy/kokoro_prosody.tflite", 37_393_104),
    _f("legacy/kokoro_vocoder.tflite", 236_130_056),
]


def _paths(tree: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"path": e["path"], "size": e["lfs"]["size"] if "lfs" in e else e["size"]} for e in tree
    ]


def test_recommended_whisper_excludes_npu_and_prefers_i8() -> None:
    rec = recommended_files(_paths(WHISPER_TREE), "cpu")
    assert rec == ["README.md", "whisper_tiny_30s_i8.tflite"]
    assert recommended_files(_paths(WHISPER_TREE), "webgpu")[-1] == "whisper_tiny_30s_i8.litertlm"


def test_recommended_kitten_pulls_all_fp16_stages() -> None:
    rec = set(recommended_files(_paths(KITTEN_TREE), "cpu"))
    assert rec == {
        "README.md",
        "g2p/dp_g2p_matcha_fp16.tflite",
        "g2p/g2p_dict.txt.gz",
        "g2p/g2p_meta.json",
        "g2p/symbols.json",
        "kitten_predictor_fp16.tflite",
        "kitten_prosody_fp16.tflite",
        "kitten_vocoder_fp16.tflite",
        "voices.npz",
    }


def test_recommended_kokoro_cpu_stages_webgpu_litertlm() -> None:
    cpu = recommended_files(_paths(KOKORO_TREE), "cpu")
    assert "legacy/kokoro_predictor.tflite" in cpu and "legacy/istft_Wi_f32.bin" in cpu
    assert "legacy/kokoro_82m_fixedlen_fp32.tflite" not in cpu
    assert not any(p.endswith(".litertlm") for p in cpu)
    web = recommended_files(_paths(KOKORO_TREE), "webgpu")
    assert [p for p in web if p.endswith(".litertlm")] == ["kokoro_int8_gpu_fp16.litertlm"]


def test_companions_for_whisper() -> None:
    assert "litert-community/whisper-large-v3-turbo" in COMPANIONS
    files = companion_files("litert-community/whisper-tiny")
    assert {f["repo_id"] for f in files} == {"openai/whisper-tiny"}
    assert "vocab.json" in {f["path"] for f in files}
    assert companion_files("litert-community/kitten-tts-nano-0.8") == []


@pytest.mark.parametrize(
    ("tag", "name", "expected"),
    [
        ("automatic-speech-recognition", "x", "asr"),
        ("text-to-speech", "x", "tts"),
        ("text-generation", "x", "llm"),
        ("audio-to-audio", "x", "audio"),
        ("voice-activity-detection", "silero", "audio"),
        ("image-classification", "x", "vision"),
        ("feature-extraction", "x", "embedding"),
        (None, "whisper-tiny", "asr"),
        (None, "Kokoro-82M", "tts"),
        (None, "Qwen2.5-0.5B", "llm"),
        (None, "resnet", "other"),
    ],
)
def test_classify(tag: str | None, name: str, expected: str) -> None:
    assert classify_task(tag, [], name) == expected


def _item(i: int, downloads: int, **kw: Any) -> dict[str, Any]:
    return {
        "id": f"litert-community/m{i}",
        "downloads": downloads,
        "likes": 1,
        "tags": [],
        "lastModified": "2026-01-01T00:00:00.000Z",
        **kw,
    }


class Counter:
    def __init__(self) -> None:
        self.calls: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        url = request.url
        if url.path == "/api/models" and url.params.get("page") == "2":
            return httpx.Response(200, json=[_item(3, 500, pipeline_tag="text-to-speech")])
        if url.path == "/api/models":
            assert url.params["author"] == "litert-community"
            nxt = f'<{HUB}/api/models?author=litert-community&page=2>; rel="next"'
            return httpx.Response(
                200,
                json=[
                    _item(1, 10, pipeline_tag="automatic-speech-recognition", gated="manual"),
                    _item(2, 900, pipeline_tag="text-generation", gated=False),
                ],
                headers={"Link": nxt},
            )
        if url.path == "/api/models/litert-community/m1":
            return httpx.Response(401, json={"error": "nope"})
        if url.path == "/api/models/litert-community/unknown":
            return httpx.Response(401)
        if url.path == "/api/models/litert-community/missing":
            return httpx.Response(404)
        if url.path == "/api/models/litert-community/whisper-tiny":
            return httpx.Response(
                200,
                json={
                    "id": "litert-community/whisper-tiny",
                    "pipeline_tag": "automatic-speech-recognition",
                },
            )
        if url.path == "/api/models/litert-community/whisper-tiny/tree/main":
            return httpx.Response(200, json=[*WHISPER_TREE, {"type": "directory", "path": "x"}])
        return httpx.Response(500)


def _client(counter: Counter, **cfg: Any) -> HubClient:
    return HubClient(
        ModelsConfig(hub_endpoint=HUB, **cfg), transport=httpx.MockTransport(counter.handler)
    )


async def test_list_paginates_sorts_and_caches() -> None:
    c = Counter()
    hub = _client(c)
    first = await hub.list_models()
    assert [m["id"] for m in first.models] == [
        "litert-community/m2",
        "litert-community/m3",
        "litert-community/m1",
    ]
    assert not first.cached
    assert first.models[2]["gated"] is True and first.models[0]["gated"] is False
    assert first.models[2]["task"] == "asr"
    n = len(c.calls)
    second = await hub.list_models(task="tts")
    assert second.cached and [m["id"] for m in second.models] == ["litert-community/m3"]
    assert len(c.calls) == n
    await hub.list_models(refresh=True)
    assert len(c.calls) > n


async def test_cache_expires_with_zero_ttl() -> None:
    c = Counter()
    hub = _client(c, catalog_ttl_s=0.0)
    await hub.list_models()
    assert not (await hub.list_models()).cached


async def test_bearer_only_when_token_set() -> None:
    c = Counter()
    await _client(c).list_models()
    assert all("authorization" not in r.headers for r in c.calls)
    c2 = Counter()
    await _client(c2, hf_token=SecretStr("hf_secret")).list_models()
    assert all(r.headers["authorization"] == "Bearer hf_secret" for r in c2.calls)


async def test_detail_uses_lfs_size_and_recommends() -> None:
    d = await _client(Counter()).model_detail("litert-community/whisper-tiny")
    sizes = {f["path"]: f["size"] for f in d["files"]}
    assert sizes["whisper_tiny_30s_i8.tflite"] == 41_116_288
    assert "x" not in sizes
    assert d["recommended_files"] == ["README.md", "whisper_tiny_30s_i8.tflite"]
    assert set(d["formats"]) == {"tflite", "litertlm"}
    assert d["task"] == "asr"
    assert d["companions"][0]["repo_id"] == "openai/whisper-tiny"


async def test_errors_mapped() -> None:
    hub = _client(Counter())
    with pytest.raises(HubAuthError, match="HF_TOKEN"):
        await hub.model_detail("litert-community/m1")  # exists in listing -> gated
    with pytest.raises(ModelNotFoundError):  # anonymous 401 for unknown repo -> 404
        await hub.model_detail("litert-community/unknown")
    with pytest.raises(ModelNotFoundError):
        await hub.model_detail("litert-community/missing")

    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    down = HubClient(ModelsConfig(hub_endpoint=HUB), transport=httpx.MockTransport(boom))
    with pytest.raises(HubUnavailableError):
        await down.list_models()
    err = HubClient(
        ModelsConfig(hub_endpoint=HUB), transport=httpx.MockTransport(lambda r: httpx.Response(503))
    )
    with pytest.raises(HubUnavailableError):
        await err.list_models()


@pytest.mark.parametrize(
    ("repo", "tag", "tags", "expected"),
    [
        ("litert-community/MoGe-2-LiteRT", "depth-estimation", ["litert", "depth"], "vision"),
        ("litert-community/MiDaS-small", "depth-estimation", [], "vision"),
        ("litert-community/RTMPose-s-LiteRT", "keypoint-detection", [], "vision"),
        ("litert-community/SAM2.1-Hiera-Tiny", "mask-generation", [], "vision"),
        ("litert-community/MoViNet-A0", "video-classification", [], "vision"),
        ("litert-community/FLUX.2-klein-4B", "text-to-image", [], "vision"),
        ("litert-community/Kokoro-82M", "text-to-speech", [], "tts"),
        ("litert-community/whisper-tiny", "automatic-speech-recognition", [], "asr"),
        ("litert-community/kitten-tts-nano-0.8", "text-to-speech", [], "tts"),
        ("litert-community/SmolVLM2-500M", "image-text-to-text", [], "llm"),
        ("litert-community/gemma-4-E2B-it-litert-lm", None, ["litert-lm"], "llm"),
        ("litert-community/Qwen3-0.6B", None, [], "llm"),
        ("litert-community/PP-OCRv5-LiteRT", None, [], "other"),
        ("litert-community/Test", None, [], "other"),
    ],
)
def test_classify_real_catalog_entries(
    repo: str, tag: str | None, tags: list[str], expected: str
) -> None:
    from dana_agent.models.hub import hub_model

    item = {"id": repo, "pipeline_tag": tag, "tags": tags}
    assert hub_model(item)["task"] == expected


def test_keyword_fallback_matches_whole_tokens() -> None:
    assert classify_task(None, [], "MoGe-2-LiteRT") == "other"  # not "ge"/"moge" substrings
    assert classify_task(None, [], "graphics-asrx") == "other"
    assert classify_task(None, [], "Phi-3-mini") == "llm"


async def test_foreign_link_next_not_followed() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.host)
        return httpx.Response(
            200,
            json=[_item(1, 1)],
            headers={"Link": '<https://evil.test/api/models?page=2>; rel="next"'},
        )

    hub = HubClient(ModelsConfig(hub_endpoint=HUB), transport=httpx.MockTransport(handler))
    assert len((await hub.list_models()).models) == 1
    assert set(seen) == {"hub.test"}

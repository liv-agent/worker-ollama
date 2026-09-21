"""Unit tests for the boot-time warm-up logic in handler.py.

These need no GPU, no network and no running Ollama, which is the point: the
warm-up only ever runs on real hardware at cold start, so getting its request
shape, bounds or failure handling wrong is otherwise only discovered by a timed
out first request in production.

    python3 -m pytest test_warmup.py -v
"""

import json
import sys
import types

import pytest

# handler.py imports the runpod SDK at module scope; stub it when it isn't
# installed so the pure functions stay testable from a bare checkout.
if "runpod" not in sys.modules:
    try:
        import runpod  # noqa: F401
    except ImportError:
        sys.modules["runpod"] = types.ModuleType("runpod")

import requests

import handler


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=None, path="/api/generate"):
        self.status_code = status_code
        self.ok = status_code < 400
        self._payload = payload
        self.text = text if text is not None else json.dumps(payload or {})
        self.request = types.SimpleNamespace(path_url=path)

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class FakeSession:
    """Records every call so tests can assert on the exact request shape."""

    def __init__(self, post_response, ps_response=None):
        self.post_response = post_response
        self.ps_response = ps_response or FakeResponse(payload={"models": []}, path="/api/ps")
        self.posts = []
        self.gets = []

    def post(self, url, **kwargs):
        self.posts.append((url, kwargs))
        return self.post_response

    def get(self, url, **kwargs):
        self.gets.append((url, kwargs))
        return self.ps_response


def loaded_ps(model, size=1000, size_vram=None):
    return FakeResponse(
        payload={
            "models": [
                {"name": model, "size": size, "size_vram": size if size_vram is None else size_vram}
            ]
        },
        path="/api/ps",
    )


# --- _duration_seconds: Go-style durations, as Ollama's env vars use ------------


@pytest.mark.parametrize(
    "value, expected",
    [
        ("60m", 3600.0),
        ("1h30m", 5400.0),
        ("90s", 90.0),
        ("500ms", 0.5),
        ("1.5h", 5400.0),
        ("-5m", -300.0),
    ],
)
def test_duration_seconds_parses(value, expected):
    assert handler._duration_seconds(value, 123.0) == pytest.approx(expected)


@pytest.mark.parametrize("value", ["", None, "garbage", "10", "5x", "m5", "1h 30m"])
def test_duration_seconds_falls_back(value):
    assert handler._duration_seconds(value, 123.0) == 123.0


# --- _warmup_timeout_seconds: always bounded, tracks OLLAMA_LOAD_TIMEOUT --------


def test_warmup_timeout_tracks_load_timeout(monkeypatch):
    monkeypatch.setenv("OLLAMA_LOAD_TIMEOUT", "10m")
    assert handler._warmup_timeout_seconds() == 600 + handler._WARMUP_MARGIN_S


@pytest.mark.parametrize("value", [None, "0s", "-1s", "not-a-duration"])
def test_warmup_timeout_never_unbounded(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("OLLAMA_LOAD_TIMEOUT", raising=False)
    else:
        monkeypatch.setenv("OLLAMA_LOAD_TIMEOUT", value)
    assert (
        handler._warmup_timeout_seconds()
        == handler._WARMUP_FALLBACK_S + handler._WARMUP_MARGIN_S
    )


# --- warm_model: the load request itself ----------------------------------------


def test_warm_model_sends_empty_prompt_load(monkeypatch):
    monkeypatch.setenv("OLLAMA_LOAD_TIMEOUT", "10m")
    session = FakeSession(
        FakeResponse(payload={"done": True}), ps_response=loaded_ps("llama3.2:3b")
    )
    monkeypatch.setattr(handler, "session", session)

    handler.warm_model("llama3.2:3b")

    url, kwargs = session.posts[0]
    assert url.endswith("/api/generate")
    # An empty prompt is Ollama's "load into memory, generate nothing" request.
    assert kwargs["json"] == {"model": "llama3.2:3b", "prompt": "", "stream": False}
    assert kwargs["timeout"] == 600 + handler._WARMUP_MARGIN_S


def test_warm_model_raises_on_server_error(monkeypatch):
    session = FakeSession(
        FakeResponse(
            status_code=500,
            payload={"error": "timed out waiting for llama runner to start"},
        )
    )
    monkeypatch.setattr(handler, "session", session)

    with pytest.raises(ValueError, match="timed out waiting for llama runner"):
        handler.warm_model("llama3.2:3b")


def test_warm_model_warns_on_cpu_offload(monkeypatch, capsys):
    session = FakeSession(
        FakeResponse(payload={"done": True}),
        ps_response=loaded_ps("big-model:latest", size=1000, size_vram=400),
    )
    monkeypatch.setattr(handler, "session", session)

    handler.warm_model("big-model")

    out = capsys.readouterr().out
    assert "offloaded to CPU" in out


def test_warm_model_reports_full_residency(monkeypatch, capsys):
    session = FakeSession(
        FakeResponse(payload={"done": True}), ps_response=loaded_ps("llama3.2:3b")
    )
    monkeypatch.setattr(handler, "session", session)

    handler.warm_model("llama3.2:3b")

    assert "resident in GPU memory" in capsys.readouterr().out


def test_warm_model_survives_ps_failure(monkeypatch, capsys):
    """A broken /api/ps must not turn a successful load into a failure."""
    session = FakeSession(FakeResponse(payload={"done": True}))

    def broken_get(url, **kwargs):
        raise requests.ConnectionError("ps is down")

    session.get = broken_get
    monkeypatch.setattr(handler, "session", session)

    handler.warm_model("llama3.2:3b")  # must not raise

    assert "could not read /api/ps" in capsys.readouterr().out


# --- warm_default_model: non-fatal, but says what failed -------------------------


def test_warm_default_model_success(monkeypatch):
    monkeypatch.setattr(handler, "resolve_default_model", lambda: "llama3.2:3b")
    warmed = []
    monkeypatch.setattr(handler, "warm_model", warmed.append)

    assert handler.warm_default_model() is True
    assert warmed == ["llama3.2:3b"]


@pytest.mark.parametrize(
    "err",
    [
        requests.ConnectionError("connection refused"),
        requests.Timeout("read timed out"),
        ValueError("HTTP 500 from /api/generate: model requires more system memory"),
    ],
)
def test_warm_default_model_failure_is_nonfatal_but_loud(monkeypatch, capsys, err):
    monkeypatch.setattr(handler, "resolve_default_model", lambda: "big-model")

    def failing_warm(model):
        raise err

    monkeypatch.setattr(handler, "warm_model", failing_warm)

    assert handler.warm_default_model() is False
    out = capsys.readouterr().out
    assert "could not pre-load 'big-model'" in out
    # The cause must be surfaced, not swallowed behind a bare WARN.
    assert str(err) in out

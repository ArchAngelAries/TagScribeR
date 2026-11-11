"""Security of the API path (from a review of the captioning code): the key never reaches a header malformed
and never a message or a log; a redirect never carries the key or the picture to another host; a hostile or
broken server can neither exhaust memory nor hold the worker; what a server says reaches the UI as plain text.

Every test server listens on 127.0.0.1 only, runs on daemon threads, is stopped in a ``finally`` and every
call into it has a hard time limit."""
import gzip
import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from PIL import Image

from inference import openai_api, server_probe, textsafe
from inference.base import CaptionRequest, InferenceError, ProviderSpec

SECRET = "tk-" + "Zq7" * 6            # a made-up key: long enough to be recognisable in any text
GOOD_REPLY = {"choices": [{"index": 0, "finish_reason": "stop",
                           "message": {"role": "assistant", "content": "a red square"}}],
              "usage": {"prompt_tokens": 9, "completion_tokens": 3}}
REQ = CaptionRequest(prompt="describe")


def within(seconds, fn):
    box = {}

    def run():
        try:
            box["value"] = fn()
        except BaseException as e:  # noqa: BLE001
            box["error"] = e

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(seconds)
    assert not t.is_alive(), f"did not finish within {seconds} s"
    if "error" in box:
        raise box["error"]
    return box["value"]


class Server:
    """A small HTTP server whose answer is decided by ``behave(handler)``; it records what it was sent."""

    def __init__(self, behave):
        self.requests = []          # (method, path, authorization header or None, body length)
        self.sent = 0               # body bytes written by endless / slow answers
        self.stop = threading.Event()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def end_headers(self):
                # one request per connection, and say so: the client must not reuse one that is closing
                if not any(b"connection:" in line.lower() for line in getattr(self, "_headers_buffer", [])):
                    self.send_header("Connection", "close")
                super().end_headers()

            def _any(self):
                body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                outer.requests.append((self.command, self.path, self.headers.get("Authorization"), len(body)))
                try:
                    behave(self)
                except OSError:
                    pass                        # the client went away: that is the point of several tests
                self.close_connection = True

            do_GET = do_POST = _any

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        self.root = f"http://127.0.0.1:{self.port}"
        self.base_url = self.root + "/v1"

    def __enter__(self):
        threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.httpd.shutdown()
        self.httpd.server_close()


def send_json(h, payload, status=200, headers=()):
    data = json.dumps(payload).encode("utf-8")
    h.send_response(status)
    h.send_header("Content-Type", "application/json")
    h.send_header("Content-Length", str(len(data)))
    for name, value in headers:
        h.send_header(name, value)
    h.end_headers()
    h.wfile.write(data)


class Sent:
    """Counts what an endless answer wrote, and tells it to stop."""

    def __init__(self):
        self.sent = 0
        self.stop = threading.Event()


def endless(server, status=200, content_type="application/json", piece=b"x" * 8192, pause=0.0, limit=40_000_000):
    """An answer that never ends by itself (until ``limit`` bytes, the client leaving, or the server stopping)."""
    def behave(h):
        h.send_response(status)
        h.send_header("Content-Type", content_type)
        h.send_header("Connection", "close")
        h.end_headers()
        while not server.stop.is_set() and server.sent < limit:
            h.wfile.write(piece)
            h.wfile.flush()
            server.sent += len(piece)
            if pause:
                time.sleep(pause)
    return behave


def provider(base_url, key="", **options):
    options.setdefault("timeout", 10.0)
    prov = openai_api.OpenAICompatible(ProviderSpec.make("openai", "m", base_url=base_url, api_key=key,
                                                         max_retries=0, **options))
    prov.load(lambda m: None)
    return prov


def ask(prov, seconds=20):
    try:
        return within(seconds, lambda: prov.generate([Image.new("RGB", (8, 8))], REQ, threading.Event()))[0]
    finally:
        prov.unload()


# -- 1. the key ------------------------------------------------------------------------------------------------------
def test_key_is_stripped_and_a_key_that_cannot_be_a_header_is_refused():
    assert textsafe.clean_api_key(f"  {SECRET}\n") == SECRET
    assert textsafe.clean_api_key(None) == "" and textsafe.clean_api_key("   ") == ""
    for bad in (f"{SECRET}\nX-Injected: 1", f"{SECRET[:5]} {SECRET[5:]}", SECRET + "\x00", SECRET + "\x7f",
                SECRET + "é", f"{SECRET}\r\nHost: evil"):
        with pytest.raises(textsafe.BadApiKey) as ei:
            textsafe.clean_api_key(bad)
        assert SECRET not in str(ei.value) and SECRET[:5] not in str(ei.value)


def test_load_refuses_a_bad_key_without_repeating_it(monkeypatch):
    import openai
    made = {}
    monkeypatch.setattr(openai, "OpenAI", lambda **kw: made.update(kw) or object())
    prov = openai_api.OpenAICompatible(ProviderSpec.make("openai", "m", base_url="http://127.0.0.1:1/v1",
                                                         api_key=f"{SECRET}\nX-Injected: 1"))
    with pytest.raises(InferenceError) as ei:
        prov.load(lambda m: None)
    assert SECRET not in str(ei.value) and "X-Injected" not in str(ei.value) and not made
    ok = openai_api.OpenAICompatible(ProviderSpec.make("openai", "m", base_url="http://127.0.0.1:1/v1",
                                                       api_key=f" {SECRET} \n"))
    ok.load(lambda m: None)
    assert made["api_key"] == SECRET                              # the trailing blank never reaches a header
    assert made["default_headers"] == {"Accept-Encoding": "identity"}


def test_key_with_surrounding_blanks_is_sent_clean():
    with Server(lambda h: send_json(h, GOOD_REPLY)) as server:
        assert ask(provider(server.base_url, f"  {SECRET} \n")) == "a red square"
        info = within(10, lambda: server_probe.probe(server.base_url, "m", f" {SECRET}\n"))
        assert info.reachable
    auth = {a for _m, _p, a, _n in server.requests}
    assert auth == {"Bearer " + SECRET}


def test_errors_and_logs_never_contain_the_key(caplog):
    from types import SimpleNamespace
    quoted = f"Illegal header value b'Bearer {SECRET} '"

    def explode(**kw):
        raise RuntimeError(quoted)

    prov = openai_api.OpenAICompatible(ProviderSpec.make("openai", "m", base_url="http://127.0.0.1:1/v1",
                                                         api_key=SECRET))
    prov.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=explode)),
                                  close=lambda: None)
    with caplog.at_level(logging.DEBUG):
        out = prov.generate([Image.new("RGB", (8, 8))], REQ)[0]
    assert isinstance(out, InferenceError) and "API call failed: RuntimeError" in str(out)
    assert SECRET not in str(out) and SECRET not in caplog.text and "[key]" in str(out)
    assert caplog.records and all(r.exc_info is None for r in caplog.records)     # no traceback in the log


def test_redact_and_plain():
    assert SECRET not in textsafe.redact(f"token {SECRET} was sent", SECRET)
    assert textsafe.redact("Authorization: Bearer abcdef123456") == "Authorization: Bearer [key]"
    assert textsafe.redact("GET /v1/models?api_key=abcdef&x=1") == "GET /v1/models?api_key=[key]&x=1"
    assert "[key]" in textsafe.redact("the key sk-" + "a1B2" * 5 + " leaked")
    nasty = "bad <img src=x onerror=alert(1)>\r\n<b>request</b>\x00\x1b[31m " + "y" * 900
    out = textsafe.plain(nasty)
    assert "<" not in out and ">" not in out and "\n" not in out and "\x1b" not in out and "\x00" not in out
    assert len(out) <= textsafe.MAX_SERVER_TEXT and out.startswith("bad ")
    login = "user:" + "pw" + "@"          # put together here so no credential-shaped URL stands in the source
    assert textsafe.url_for_display("https://" + login + "api.example.com:8443/v1?api_key=abc#frag") == \
        "https://api.example.com:8443/v1"
    assert textsafe.url_for_display("not a url") == ""
    prov = openai_api.OpenAICompatible(ProviderSpec.make(
        "openai", "m<i>", base_url="http://" + f"user:{SECRET}" + f"@127.0.0.1:1234/v1?api_key={SECRET}"))
    assert SECRET not in prov.display_name and "<" not in prov.display_name
    assert prov.display_name.endswith("@ http://127.0.0.1:1234/v1")


# -- 2. redirects ----------------------------------------------------------------------------------------------------
def test_probe_does_not_follow_a_redirect_with_the_key():
    with Server(lambda h: send_json(h, {"data": [{"id": "stolen"}], "total_slots": 4})) as other:
        def redirect(h):
            h.send_response(307)
            h.send_header("Location", other.root + h.path)
            h.send_header("Content-Length", "0")
            h.end_headers()

        with Server(redirect) as server:
            info = within(15, lambda: server_probe.probe(server.base_url, "m", SECRET))
        assert server.requests and all(a == "Bearer " + SECRET for _m, _p, a, _n in server.requests)
    assert other.requests == []                                   # nothing went to the other host: no key, no request
    assert info.models == [] and info.slots is None


def test_client_does_not_follow_a_redirect_to_another_host():
    with Server(lambda h: send_json(h, GOOD_REPLY)) as other:
        def redirect(h):
            h.send_response(307)
            h.send_header("Location", other.root + h.path)
            h.send_header("Content-Length", "0")
            h.end_headers()

        with Server(redirect) as server:
            out = ask(provider(server.base_url, SECRET))
    assert other.requests == []                                   # neither the key nor the picture left the host
    assert isinstance(out, InferenceError) and "redirected" in str(out) and SECRET not in str(out)


def test_client_follows_a_redirect_on_the_same_host():
    def behave(h):
        if h.path.startswith("/old/"):
            h.send_response(307)
            h.send_header("Location", h.path[len("/old"):])
            h.send_header("Content-Length", "0")
            h.end_headers()
        else:
            send_json(h, GOOD_REPLY)

    with Server(behave) as server:
        out = ask(provider(server.root + "/old/v1", SECRET))
    assert out == "a red square"
    assert [p for _m, p, _a, _n in server.requests] == ["/old/v1/chat/completions", "/v1/chat/completions"]
    assert all(a == "Bearer " + SECRET for _m, _p, a, _n in server.requests)


# -- 3. size and time limits -----------------------------------------------------------------------------------------
@pytest.fixture
def small_limits(monkeypatch):
    monkeypatch.setattr(openai_api, "MAX_REPLY_BYTES", 200_000)
    monkeypatch.setattr(openai_api, "MAX_EVENT_BYTES", 50_000)
    monkeypatch.setattr(server_probe, "MAX_REPLY_BYTES", 200_000)


@pytest.mark.parametrize("status,content_type,loop_guard", [
    (200, "application/json", False),          # an endless "completion"
    (500, "application/json", False),          # an endless error body
    (200, "text/event-stream", True),          # an endless single event of a streamed reply
    (200, "application/json", True),
])
def test_an_endless_reply_is_dropped_not_held_in_memory(small_limits, status, content_type, loop_guard):
    server = Sent()
    with Server(endless(server, status, content_type)) as real:
        t0 = time.monotonic()
        out = ask(provider(real.base_url, SECRET, loop_guard=loop_guard))
        took = time.monotonic() - t0
        real.stop.set()
        server.stop.set()
    assert isinstance(out, InferenceError) and SECRET not in str(out)
    assert server.sent < 5_000_000 and took < 15                   # far below the 40 MB the server would send
    if status == 200:
        assert "too large" in str(out)


def test_a_compressed_reply_is_refused(small_limits):
    bomb = gzip.compress(b'{"choices": [{"message": {"content": "' + b"A" * 5_000_000 + b'"}}]}')

    def behave(h):
        h.send_response(200)
        h.send_header("Content-Type", "application/json")
        h.send_header("Content-Encoding", "gzip")
        h.send_header("Content-Length", str(len(bomb)))
        h.end_headers()
        h.wfile.write(bomb)

    with Server(behave) as server:
        out = ask(provider(server.base_url))
    assert len(bomb) < 200_000                                    # small on the wire, 5 MB unpacked
    assert isinstance(out, InferenceError) and "compressed" in str(out)


@pytest.mark.parametrize("loop_guard", [False, True])
def test_a_trickling_reply_ends_at_the_time_limit(loop_guard):
    server = Sent()
    ctype = "text/event-stream" if loop_guard else "application/json"
    with Server(endless(server, 200, ctype, piece=b" ", pause=0.05)) as real:
        t0 = time.monotonic()
        out = ask(provider(real.base_url, timeout=1.0, loop_guard=loop_guard))
        took = time.monotonic() - t0
        server.stop.set()
    assert isinstance(out, InferenceError) and "timed out" in str(out) and out.busy and not out.resend
    assert took < 8


def test_a_silent_server_ends_at_the_timeout():
    release = threading.Event()
    with Server(lambda h: release.wait(20)) as server:
        t0 = time.monotonic()
        out = ask(provider(server.base_url, timeout=1.0))
        took = time.monotonic() - t0
        release.set()
    assert isinstance(out, InferenceError) and "timed out" in str(out) and took < 8


def test_probe_is_bounded_in_size_and_time(small_limits, caplog):
    server = Sent()
    with caplog.at_level(logging.DEBUG):
        with Server(endless(server)) as real:                       # an endless model list
            info = within(15, lambda: server_probe.probe(real.base_url, "m", SECRET, timeout=1.0))
            server.stop.set()
        assert info.models == [] and server.sent < 5_000_000
        slow = Sent()
        with Server(endless(slow, piece=b" ", pause=0.05)) as real:
            t0 = time.monotonic()
            info = within(20, lambda: server_probe.probe(real.base_url, "m", SECRET, timeout=0.5))
            took = time.monotonic() - t0
            slow.stop.set()
        assert info.models == [] and took < 10
        with Server(lambda h: send_json(h, {"data": [{"id": f"m{i}"} for i in range(2000)]})) as real:
            info = within(10, lambda: server_probe.probe(real.base_url, "m0"))
        assert len(info.models) == server_probe.MAX_MODELS and not info.notes
    assert SECRET not in caplog.text
    assert server_probe.probe("file:///etc/hosts").reachable is False
    assert server_probe.probe("ftp://127.0.0.1/v1").reachable is False


def test_probe_does_not_send_a_key_it_cannot_send_safely():
    with Server(lambda h: send_json(h, {"data": [{"id": "m"}]})) as server:
        info = within(10, lambda: server_probe.probe(server.base_url, "m", f"{SECRET}\r\nX-Injected: 1"))
    assert info.reachable and all(a is None for _m, _p, a, _n in server.requests)


# -- 4. what a server says is shown as plain text --------------------------------------------------------------------
@pytest.mark.parametrize("status", [400, 422, 500])
def test_server_error_text_is_plain_short_and_without_the_key(status):
    message = f"<img src=x onerror=alert(1)> key {SECRET} <a href='http://evil.example'>click</a>\n" + "z" * 5000

    with Server(lambda h: send_json(h, {"error": {"message": message, "type": "bad"}}, status)) as server:
        out = ask(provider(server.base_url, SECRET))
    text = str(out)
    assert isinstance(out, InferenceError) and "onerror" in text          # the words are kept ...
    assert "<" not in text and ">" not in text and SECRET not in text     # ... the markup and the key are not
    assert len(text) < 700


def test_stream_error_event_and_malformed_bodies():
    def stream_error(h):
        data = b'data: {"error": {"message": "<b>context size exceeded</b>"}}\n\n'
        h.send_response(200)
        h.send_header("Content-Type", "text/event-stream")
        h.send_header("Content-Length", str(len(data)))
        h.end_headers()
        h.wfile.write(data)

    with Server(stream_error) as server:
        out = ask(provider(server.base_url, loop_guard=True))
    assert isinstance(out, InferenceError) and "context size exceeded" in str(out) and "<" not in str(out)
    assert out.busy and out.resend
    for body in (b"<html>not json</html>", b"[1, 2]", b'{"choices": []}', b'{"choices": [{"message": {"content": 7}}]}'):
        def raw(h, body=body):
            h.send_response(200)
            h.send_header("Content-Type", "application/json")
            h.send_header("Content-Length", str(len(body)))
            h.end_headers()
            h.wfile.write(body)

        with Server(raw) as server:
            out = ask(provider(server.base_url))
        assert isinstance(out, InferenceError) and "malformed" in str(out)


def test_a_normal_reply_and_a_normal_stream_still_work():
    with Server(lambda h: send_json(h, GOOD_REPLY)) as server:
        prov = provider(server.base_url)
        result, info = within(10, lambda: prov.caption_prepared("data:image/jpeg;base64,AAAA", REQ, threading.Event()))
        prov.unload()
    assert result == "a red square" and info["completion_tokens"] == 3 and info["prompt_tokens"] == 9

    def stream(h):
        events = [{"choices": [{"delta": {"role": "assistant"}}]},
                  {"choices": [{"delta": {"content": "a red "}}]},
                  {"choices": [{"delta": {"content": [{"type": "text", "text": "square"}]}}]},
                  {"choices": [{"delta": {}, "finish_reason": "stop"}]}]
        data = b": keep-alive\n\n" + b"".join(b"data: " + json.dumps(e).encode() + b"\n\n" for e in events)
        data += b"data: [DONE]\n\n"
        h.send_response(200)
        h.send_header("Content-Type", "text/event-stream; charset=utf-8")
        h.send_header("Content-Length", str(len(data)))
        h.end_headers()
        h.wfile.write(data)

    with Server(stream) as server:
        assert ask(provider(server.base_url, loop_guard=True)) == "a red square"

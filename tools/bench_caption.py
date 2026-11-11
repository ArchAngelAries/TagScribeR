"""Measure captioning speed through an OpenAI-compatible server: the request pipeline as it was, and as it is.

Two ways to run it:

    python tools/bench_caption.py --server http://localhost:1234/v1 --model <id> --images 24
        Against a real server (LM Studio, llama.cpp server, ...). Sends generated test pictures only (shapes,
        colours and a rendered word: never your own images). If the model is not loaded yet, the server loads
        it on the first request, exactly as it would for the app.

    python tools/bench_caption.py --mock --images 48
        Against a simulated server started by this script (no model, no GPU): a model of a server with N
        parallel slots whose time per request grows with the picture's pixels and the reply's length. Numbers
        from this mode are simulations of the client pipeline, not of your hardware.

It prints images per minute for: the old pipeline (one request at a time, as before this change), the new
pipeline at several numbers of parallel requests (stopping when a step brings under 10 %), and the new pipeline
with a pixel budget. With a real server it also counts how many known facts of each picture (shape, colours,
a large word and a line of small print) the captions mention, so a speed setting that costs accuracy shows up:
the small print is about 2 % of the picture's height, the first thing a pixel budget makes unreadable.

Nothing here changes the server's settings. Standard library, Pillow, numpy and the `openai` client only.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import queue
import random
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# A higher level counts as better only from this factor on: two runs of a few dozen different pictures differ
# by several percent on their own, and a level that only queues in the server looks about that much "faster".
MIN_STEP_GAIN = 1.10

DEFAULT_PROMPT = ("Describe this image in detail, focusing on visual elements, colors, lighting, and composition. "
                  "Name the shapes and their colours, and quote all text you can read, including small print.")


# ======================================================================================================================
# A simulated server
# ======================================================================================================================
class MockModel:
    """Timing model of a vision-language model server.

    time per request = wait for a free slot
                     + vision encoder: ``vision_s_per_mp`` x megapixels (one picture at a time, as llama.cpp does)
                     + prompt: (image tokens + new text tokens) / ``prompt_tps`` x (1 + ``batch_penalty`` x (active - 1))
                     + reply: tokens / ``gen_tps`` x (1 + ``batch_penalty`` x (active - 1))

    ``batch_penalty`` 0 is perfect scaling with the number of active slots, 1 is none (N streams share one
    stream's speed). Image tokens = pixels / ``px_per_token``^2. All times are multiplied by ``time_scale``.
    """

    def __init__(self, slots=4, vision_s_per_mp=0.25, prompt_tps=1500.0, gen_tps=55.0, batch_penalty=0.2,
                 px_per_token=32, out_tokens=120, out_cv=0.3, prefix_cache=True, loop_rate=0.0, error_rate=0.0,
                 fail_above=0, context=0, stall_every=0, stall_seconds=30.0, time_scale=1.0, flavour="lmstudio",
                 model="mock-vlm", seed=0):
        self.cfg = dict(slots=slots, vision_s_per_mp=vision_s_per_mp, prompt_tps=prompt_tps, gen_tps=gen_tps,
                        batch_penalty=batch_penalty, px_per_token=px_per_token, out_tokens=out_tokens, out_cv=out_cv,
                        prefix_cache=prefix_cache, loop_rate=loop_rate, error_rate=error_rate, fail_above=fail_above,
                        context=context, stall_every=stall_every, stall_seconds=stall_seconds,
                        time_scale=time_scale, flavour=flavour, model=model)
        self.__dict__.update(self.cfg)
        self.lock = threading.Lock()
        self.vision_lock = threading.Lock()
        self.free_slots: queue.Queue = queue.Queue()
        for i in range(slots):
            self.free_slots.put(i)
        self.slot_text = {}             # slot -> the text prompt it saw last (its reusable prefix)
        self.active = 0
        self.waiting = 0
        self.tokens_in_flight = 0
        self.rng = random.Random(seed)
        self.stats = {"requests": 0, "completed": 0, "errors": 0, "peak_active": 0, "peak_waiting": 0,
                      "image_tokens": 0, "text_tokens": 0, "cached_tokens": 0, "reply_tokens": 0, "aborted": 0,
                      "stream_requests": 0, "max_megapixels": 0.0}

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            time.sleep(seconds * self.time_scale)

    def sleep_until(self, due: float, seconds: float) -> float:
        """Sleep so that this step ends ``seconds`` (scaled) after ``due``; returns the new due time. Steps
        timed against the clock do not add up each sleep's overshoot (a reply is hundreds of short steps)."""
        due += seconds * self.time_scale
        wait = due - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        return due

    # -- one request ---------------------------------------------------------------------------------------------
    def plan(self, body: dict) -> dict:
        """What the request costs: image tokens, text tokens, reply length, whether the reply loops."""
        text_parts, image_bytes = [], b""
        for m in body.get("messages", []):
            content = m.get("content")
            if isinstance(content, str):
                text_parts.append(content)
            elif isinstance(content, list):
                for part in content:
                    if part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                    elif part.get("type") == "image_url":
                        url = part.get("image_url", {}).get("url", "")
                        if url.startswith("data:") and "," in url:
                            image_bytes = base64.b64decode(url.split(",", 1)[1])
        megapixels, image_tokens = 0.0, 0
        if image_bytes:
            from PIL import Image
            with Image.open(io.BytesIO(image_bytes)) as im:      # the header is enough for the size
                w, h = im.size
            megapixels = w * h / 1e6
            image_tokens = max(1, round(w / self.px_per_token)) * max(1, round(h / self.px_per_token))
        text = "\n".join(text_parts)
        digest = hashlib.sha1(image_bytes or text.encode("utf-8")).digest()
        u = int.from_bytes(digest[:4], "big") / 2**32
        want = max(8, round(self.out_tokens * (1 + self.out_cv * 3 ** 0.5 * (2 * u - 1))))
        loops = (digest[4] / 256.0) < self.loop_rate
        limit = int(body.get("max_tokens") or 512)
        return {"text": text, "megapixels": megapixels, "image_tokens": image_tokens,
                "text_tokens": max(1, len(text) // 4), "reply_tokens": limit if loops else min(want, limit),
                "loops": loops, "finish": "length" if loops or want > limit else "stop", "digest": digest}

    def words(self, plan: dict):
        """The reply, one token (a word and a space) at a time."""
        if plan["loops"]:
            start = ["A", "figure", "stands", "in", "a", "plain", "room,", "wearing"]
            unit = ["a", "red", "scarf,"]
            for i in range(plan["reply_tokens"]):
                yield (start[i] if i < len(start) else unit[(i - len(start)) % len(unit)]) + " "
            return
        vocab = ("a", "large", "shape", "sits", "near", "the", "centre", "of", "frame", "with", "soft", "light",
                 "and", "clear", "edges", "against", "plain", "background", "colour", "text", "reads", "sample")
        rnd = random.Random(plan["digest"])
        for i in range(plan["reply_tokens"]):
            yield rnd.choice(vocab) + (". " if i == plan["reply_tokens"] - 1 else " ")


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"          # keep-alive, like a real server
    server_version = "MockVLM/1"

    def log_message(self, *args):          # quiet
        pass

    def _json(self, status: int, payload: dict) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _error(self, status: int, message: str) -> None:
        with self.server.model.lock:
            self.server.model.stats["errors"] += 1
        self._json(status, {"error": {"message": message, "type": "server_error", "code": status}})

    def do_GET(self):
        model: MockModel = self.server.model
        path = self.path.split("?")[0].rstrip("/")
        if path == "/v1/models":
            self._json(200, {"object": "list", "data": [{"id": model.model, "object": "model"}]})
        elif path == "/props" and model.flavour == "llamacpp":
            self._json(200, {"total_slots": model.slots, "modalities": {"vision": True},
                             "default_generation_settings": {"n_ctx": (model.context or 8192) // model.slots}})
        elif path == "/api/v0/models" and model.flavour == "lmstudio":
            self._json(200, {"object": "list", "data": [{
                "id": model.model, "object": "model", "type": "vlm", "state": "loaded",
                "loaded_context_length": model.context or 8192, "max_context_length": 32768}]})
        elif path == "/mock/stats":
            with model.lock:
                self._json(200, dict(model.stats, config=model.cfg))
        else:
            self._json(404, {"error": {"message": f"Unexpected endpoint or method. (GET {path})"}})

    def do_POST(self):
        model: MockModel = self.server.model
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length)
        if self.path.split("?")[0].rstrip("/") != "/v1/chat/completions":
            return self._json(404, {"error": {"message": "Unexpected endpoint or method."}})
        try:
            body = json.loads(raw)
            plan = model.plan(body)
        except Exception as e:  # noqa: BLE001
            return self._error(400, f"bad request: {e}")
        need = plan["image_tokens"] + plan["text_tokens"] + plan["reply_tokens"]
        with model.lock:
            model.stats["requests"] += 1
            n = model.stats["requests"]
            model.stats["max_megapixels"] = max(model.stats["max_megapixels"], plan["megapixels"])
            crowd = model.active + model.waiting + 1
            roll = model.rng.random()
            if model.fail_above and crowd > model.fail_above:
                refuse = (503, "server busy: no slot available")
            elif model.context and model.tokens_in_flight + need > model.context:
                refuse = (400, "the request exceeds the available context size")
            elif roll < model.error_rate:
                refuse = (500, "internal error (injected)")
            else:
                refuse = None
                model.waiting += 1
                model.tokens_in_flight += need
                model.stats["peak_waiting"] = max(model.stats["peak_waiting"], model.waiting - 1)
        if refuse:
            return self._error(*refuse)
        slot = model.free_slots.get()
        try:
            with model.lock:
                model.waiting -= 1
                model.active += 1
                model.stats["peak_active"] = max(model.stats["peak_active"], model.active)
                cached = 0
                if model.prefix_cache:
                    old, new = model.slot_text.get(slot, ""), plan["text"]
                    same = 0
                    for a, b in zip(old, new):
                        if a != b:
                            break
                        same += 1
                    cached = min(plan["text_tokens"], same // 4)
                model.slot_text[slot] = plan["text"]
                model.stats["image_tokens"] += plan["image_tokens"]
                model.stats["text_tokens"] += plan["text_tokens"]
                model.stats["cached_tokens"] += cached
            if model.stall_every and n % model.stall_every == 0:
                model.sleep(model.stall_seconds)
            with model.vision_lock:
                model.sleep(model.vision_s_per_mp * plan["megapixels"])
            with model.lock:
                busy = model.active
            model.sleep((plan["image_tokens"] + plan["text_tokens"] - cached) / model.prompt_tps
                        * (1 + model.batch_penalty * (busy - 1)))
            self._generate(model, body, plan)
        finally:
            with model.lock:
                model.active -= 1
                model.tokens_in_flight -= need
            model.free_slots.put(slot)

    def _generate(self, model: MockModel, body: dict, plan: dict) -> None:
        stream = bool(body.get("stream"))
        if stream:
            with model.lock:
                model.stats["stream_requests"] += 1
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
        produced, pending, sent = [], [], 0
        words = list(model.words(plan))
        due = time.monotonic()
        try:
            for i, word in enumerate(words):
                produced.append(word)
                pending.append(word)
                if len(pending) >= 4 or i == len(words) - 1:
                    with model.lock:
                        active = model.active
                    due = model.sleep_until(due, len(pending) / model.gen_tps
                                            * (1 + model.batch_penalty * (active - 1)))
                    sent += len(pending)
                    if stream:
                        self._event({"choices": [{"index": 0, "delta": {"content": "".join(pending)},
                                                  "finish_reason": None}]})
                    pending = []
            if stream:
                self._event({"choices": [{"index": 0, "delta": {}, "finish_reason": plan["finish"]}]})
                self._chunk(b"data: [DONE]\n\n")
                self._chunk(b"")
        except (BrokenPipeError, ConnectionError, OSError):
            with model.lock:
                model.stats["aborted"] += 1          # the client closed the stream: generation stops here
                model.stats["reply_tokens"] += sent
            self.close_connection = True
            return
        with model.lock:
            model.stats["completed"] += 1
            model.stats["reply_tokens"] += len(produced)
        if not stream:
            self._json(200, {"id": "mock", "object": "chat.completion", "model": model.model, "choices": [{
                "index": 0, "finish_reason": plan["finish"],
                "message": {"role": "assistant", "content": "".join(produced).strip()}}],
                "usage": {"prompt_tokens": plan["image_tokens"] + plan["text_tokens"],
                          "completion_tokens": len(produced),
                          "total_tokens": plan["image_tokens"] + plan["text_tokens"] + len(produced)}})

    def _event(self, payload: dict) -> None:
        payload = dict(payload, id="mock", object="chat.completion.chunk", model=self.server.model.model)
        self._chunk(b"data: " + json.dumps(payload).encode("utf-8") + b"\n\n")

    def _chunk(self, data: bytes) -> None:
        self.wfile.write(f"{len(data):X}\r\n".encode("ascii") + data + b"\r\n")
        self.wfile.flush()


class MockServer:
    """A ``MockModel`` behind HTTP on 127.0.0.1 and a free port. Use as a context manager."""

    def __init__(self, **config):
        self.model = MockModel(**config)
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.httpd.daemon_threads = True
        self.httpd.model = self.model
        self.port = self.httpd.server_address[1]
        self.base_url = f"http://127.0.0.1:{self.port}/v1"
        self._thread = threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.05},
                                        name="mock-vlm", daemon=True)

    def __enter__(self) -> "MockServer":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()

    @property
    def stats(self) -> dict:
        with self.model.lock:
            return dict(self.model.stats)


# ======================================================================================================================
# Generated test pictures with known content
# ======================================================================================================================
COLOURS = {"red": (200, 40, 40), "green": (40, 150, 60), "blue": (40, 80, 200), "yellow": (235, 205, 50),
           "purple": (130, 60, 170), "orange": (235, 130, 40), "white": (240, 240, 240), "black": (25, 25, 25)}
SHAPES = ("circle", "square", "triangle")
WORDS = ("HARBOR", "WINDOW", "GARDEN", "BRIDGE", "CANDLE", "MARKET", "PLANET", "RIVER", "SUMMIT", "LANTERN")
SMALL_WORDS = ("copper", "meadow", "violin", "pebble", "anchor", "saddle", "thimble", "walnut", "lagoon", "ribbon")
SIZES = ((1216, 832), (1600, 1200), (2048, 1536), (1536, 2048), (3000, 2000), (1024, 1024))


def make_pictures(folder: Path, count: int, seed: int = 0, sizes=SIZES) -> list[tuple[Path, dict]]:
    """``count`` pictures of a coloured shape on a coloured background with a large word and a line of small
    print (about 2 % of the picture's height), and what each shows."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    rnd = random.Random(seed)
    noise = np.random.default_rng(seed)
    folder.mkdir(parents=True, exist_ok=True)
    out = []
    for i in range(count):
        w, h = sizes[i % len(sizes)]
        background, colour = rnd.sample(sorted(COLOURS), 2)
        shape, word, small = rnd.choice(SHAPES), rnd.choice(WORDS), rnd.choice(SMALL_WORDS)
        base = np.empty((h, w, 3), np.float32)
        base[:] = COLOURS[background]
        base *= np.linspace(0.85, 1.0, w, dtype=np.float32)[None, :, None]         # a soft gradient
        base += noise.normal(0, 4, (h // 4 + 1, w // 4 + 1, 1)).repeat(4, 0).repeat(4, 1)[:h, :w]   # mild texture
        im = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), "RGB")
        d = ImageDraw.Draw(im)
        side = min(w, h) // 2
        x0, y0 = (w - side) // 2, (h - side) // 3
        box = [x0, y0, x0 + side, y0 + side]
        if shape == "circle":
            d.ellipse(box, fill=COLOURS[colour])
        elif shape == "square":
            d.rectangle(box, fill=COLOURS[colour])
        else:
            d.polygon([(x0 + side // 2, y0), (x0, y0 + side), (x0 + side, y0 + side)], fill=COLOURS[colour])
        font = ImageFont.load_default(size=max(24, h // 10))
        ink = (255, 255, 255) if background in ("black", "blue", "purple", "red", "green") else (0, 0, 0)
        d.text((w // 2, h - h // 8), word, fill=ink, font=font, anchor="mm")
        d.text((w // 2, h // 16), f"serial {small}", fill=ink, font=ImageFont.load_default(size=max(10, h // 50)),
               anchor="mm")
        path = folder / f"bench_{seed}_{i:04d}.{'jpg' if i % 2 else 'png'}"
        im.save(path, **({"quality": 93} if path.suffix == ".jpg" else {}))
        out.append((path, {"background": background, "colour": colour, "shape": shape, "word": word,
                           "small": small}))
    return out


def facts_found(caption: str, facts: dict) -> tuple[int, int]:
    """How many of the picture's known facts the caption mentions: (found, of)."""
    low = caption.lower().replace("color", "colour")
    wanted = [facts["background"], facts["colour"], facts["word"].lower(), facts["small"]]
    shape = facts["shape"] in low or (facts["shape"] == "square" and "rectangle" in low)
    return sum(w in low for w in wanted) + int(shape), 5


# ======================================================================================================================
# The two client pipelines
# ======================================================================================================================
def _legacy_load_rgb(path, max_side):
    """inference.image_prep.load_rgb as it was (every picture decoded at full size)."""
    from PIL import Image, ImageOps
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
            im = im.convert("RGBA")
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, mask=im.getchannel("A"))
            im = bg
        else:
            im = im.convert("RGB")
        if max_side and max(im.size) > max_side:
            im.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        else:
            im.load()
        return im


def run_legacy(paths, base_url, model, api_key, prompt, max_tokens, concurrency=1, max_side=1536, timeout=180.0,
               temperature=0.7):
    """The request pipeline as it was before this change, reproduced for comparison: chunks of ``concurrency``
    pictures, the next chunk decoded on one helper thread, each chunk waited for as a whole, and each request
    polled for its end four times a second."""
    from openai import OpenAI
    from inference.image_prep import to_jpeg_base64
    client = OpenAI(base_url=base_url, api_key=api_key or "not-needed", timeout=timeout, max_retries=0)
    never = threading.Event()

    def one(img):
        url = f"data:image/jpeg;base64,{to_jpeg_base64(img)}"
        kwargs = {"model": model, "max_tokens": max_tokens, "temperature": temperature, "top_p": 0.9,
                  "messages": [{"role": "user", "content": [{"type": "text", "text": prompt},
                                                            {"type": "image_url", "image_url": {"url": url}}]}],
                  "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
        box = {}

        def call():
            try:
                box["text"] = client.chat.completions.create(**kwargs).choices[0].message.content or ""
            except Exception as e:  # noqa: BLE001
                box["text"] = e

        t = threading.Thread(target=call, daemon=True)
        t.start()
        while t.is_alive():
            never.wait(0.25)
        return box["text"]

    chunks = [paths[i:i + concurrency] for i in range(0, len(paths), concurrency)]
    results = []
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(lambda ps: [_legacy_load_rgb(p, max_side) for p in ps], chunks[0])
        for idx, chunk in enumerate(chunks):
            images = pending.result()
            if idx + 1 < len(chunks):
                pending = pool.submit(lambda ps: [_legacy_load_rgb(p, max_side) for p in ps], chunks[idx + 1])
            if len(images) == 1:
                texts = [one(images[0])]
            else:
                with ThreadPoolExecutor(max_workers=len(images)) as reqs:
                    texts = list(reqs.map(one, images))
            results.extend(zip(chunk, texts))
    seconds = time.monotonic() - t0
    client.close()
    return {"seconds": seconds, "results": results, "parallel": concurrency, "events": []}


def run_new(paths, base_url, model, api_key, prompt, max_tokens, concurrency=1, auto=False, max_parallel=0,
            max_side=1536, max_pixels=0, loop_guard=False, image_first=False, timeout=180.0, temperature=0.7):
    """The request pipeline the app uses now (inference/pipeline.py with the API provider)."""
    from inference.base import CaptionRequest, GenerationParams, ProviderSpec
    from inference.image_prep import load_rgb
    from inference.openai_api import OpenAICompatible
    from inference.pipeline import run_pipeline
    options = dict(base_url=base_url, api_key=api_key, concurrency=concurrency, auto_parallel=auto,
                   loop_guard=loop_guard, image_first=image_first, timeout=timeout, max_retries=0)
    if max_parallel:
        options["max_parallel"] = max_parallel
    provider = OpenAICompatible(ProviderSpec.make("openai", model, **options))
    provider.load(lambda _msg: None)
    request = CaptionRequest(prompt=prompt, max_image_side=max_side, max_image_pixels=max_pixels,
                             params=GenerationParams(max_new_tokens=max_tokens, temperature=temperature))
    limiter = provider.make_limiter()
    results, latencies, peak = [], [], limiter.limit
    t0 = time.monotonic()
    for out in run_pipeline(paths, lambda p: provider.prepare(load_rgb(p, max_side, max_pixels=max_pixels), request),
                            lambda prepared, _p, stop: provider.caption_prepared(prepared, request, stop), limiter):
        results.append((out.key, out.result))
        latencies.append(out.seconds)
        peak = max(peak, limiter.limit)
    seconds = time.monotonic() - t0
    provider.unload()
    return {"seconds": seconds, "results": results, "parallel": limiter.limit, "peak_parallel": peak,
            "events": list(limiter.events), "latencies": latencies, "totals": dict(provider.totals),
            "settled": limiter.settled}


# ======================================================================================================================
# The benchmark
# ======================================================================================================================
def _row(name: str, run: dict, facts: dict | None) -> dict:
    ok = [(p, t) for p, t in run["results"] if isinstance(t, str)]
    failed = [(p, t) for p, t in run["results"] if not isinstance(t, str)]
    n = len(run["results"])
    found = total = 0
    if facts is not None:
        for p, text in ok:
            f, of = facts_found(text, facts[p])
            found, total = found + f, total + of
    return {"name": name, "images": n, "ok": len(ok), "failed": len(failed), "seconds": round(run["seconds"], 2),
            "per_minute": round(60 * len(ok) / run["seconds"], 1) if run["seconds"] > 0 else 0.0,
            "parallel": run["parallel"], "facts": f"{found}/{total}" if total else "n/a",
            "first_error": str(failed[0][1]).splitlines()[0][:120] if failed else "",
            "events": run.get("events", []), "totals": run.get("totals", {})}


def _print(rows: list[dict]) -> None:
    print(f"\n{'pipeline':46s} {'images':>6s} {'failed':>6s} {'seconds':>8s} {'images/min':>10s} {'parallel':>8s} "
          f"{'facts found':>11s}")
    for r in rows:
        print(f"{r['name']:46s} {r['images']:6d} {r['failed']:6d} {r['seconds']:8.1f} {r['per_minute']:10.1f} "
              f"{r['parallel']:8d} {r['facts']:>11s}")
        if r["first_error"]:
            print(f"    first error: {r['first_error']}")


def benchmark(base_url: str, model: str, api_key: str, images: int, folder: Path, *, levels, budget_pixels: int,
              prompt: str, max_tokens: int, timeout: float, auto_images: int, check_facts: bool,
              loop_guard: bool) -> dict:
    rows, seed = [], 0

    def pictures(count):
        nonlocal seed
        seed += 1                     # new pictures for every run: a server cache must not flatter a later run
        made = make_pictures(folder / f"run{seed}", count, seed)
        return [p for p, _ in made], ({p: f for p, f in made} if check_facts else None)

    common = dict(base_url=base_url, model=model, api_key=api_key, prompt=prompt, max_tokens=max_tokens,
                  timeout=timeout)
    paths, facts = pictures(images)
    print("old pipeline, one request at a time ...", flush=True)
    rows.append(_row("old: 1 at a time, 1536 px", run_legacy(paths, **common), facts))
    best, best_rate = 1, 0.0
    for level in levels:
        paths, facts = pictures(images)
        print(f"new pipeline, {level} at a time ...", flush=True)
        row = _row(f"new: {level} at a time, 1536 px", run_new(paths, concurrency=level, loop_guard=loop_guard,
                                                              **common), facts)
        rows.append(row)
        if row["failed"] > row["images"] // 4:
            print(f"    {row['failed']} of {row['images']} failed at {level} at a time: not going higher.")
            break
        if row["per_minute"] < best_rate * MIN_STEP_GAIN:
            break                                              # the next level brought nothing real: stop here
        best, best_rate = level, row["per_minute"]
    if budget_pixels:
        paths, facts = pictures(images)
        print(f"new pipeline, {best} at a time, pixel budget {budget_pixels / 1e6:.1f} MP ...", flush=True)
        rows.append(_row(f"new: {best} at a time, {budget_pixels / 1e6:.1f} MP budget",
                         run_new(paths, concurrency=best, max_pixels=budget_pixels, loop_guard=loop_guard, **common),
                         facts))
    auto = None
    if auto_images:
        paths, facts = pictures(auto_images)
        print(f"new pipeline, automatic parallel requests over {auto_images} pictures ...", flush=True)
        run = run_new(paths, auto=True, loop_guard=loop_guard, **common)
        auto = _row("new: automatic, 1536 px", run, facts)
        auto["settled"] = run["settled"]
        rows.append(auto)
    _print(rows)
    old = rows[0]["per_minute"]
    top = max(rows[1:], key=lambda r: r["per_minute"])
    print(f"\nBest fixed number of parallel requests found: {best}")
    if auto is not None:
        print(f"The automatic setting ended at {auto['parallel']} at a time "
              f"({'settled' if auto['settled'] else 'still searching: use more pictures'}); its decisions:")
        for e in auto["events"]:
            print(f"    {e}")
    if old > 0:
        print(f"Fastest: '{top['name']}' at {top['per_minute']:.1f} images/min, {top['per_minute'] / old:.2f}x the old "
              f"pipeline ({old:.1f}).")
        print(f"At that rate 5,000 images take about {5000 / top['per_minute'] / 60:.1f} h "
              f"(old pipeline: {5000 / old / 60:.1f} h).")
    return {"rows": rows, "best_parallel": best}


def _spawn_mock(args) -> tuple[subprocess.Popen, str]:
    """The simulated server in its own process, so its threads do not share this benchmark's interpreter."""
    cmd = [sys.executable, str(Path(__file__).resolve()), "--serve-mock", "--mock-slots", str(args.mock_slots),
           "--mock-scale", str(args.mock_scale), "--mock-flavour", args.mock_flavour,
           "--mock-loop-rate", str(args.mock_loop_rate), "--mock-penalty", str(args.mock_penalty),
           "--mock-context", str(args.mock_context)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
    line = proc.stdout.readline().strip()
    if not line.startswith("http://127.0.0.1:"):
        proc.kill()
        raise SystemExit(f"the simulated server did not start: {line!r}")
    return proc, line


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", help="API base URL of a real server, e.g. http://localhost:1234/v1")
    ap.add_argument("--model", default="", help="model id as the server names it")
    ap.add_argument("--api-key-env", default="", help="name of an environment variable holding the API key")
    ap.add_argument("--images", type=int, default=24, help="pictures per run (default 24)")
    ap.add_argument("--levels", default="1,2,4,8", help="parallel requests to try, in order (default 1,2,4,8)")
    ap.add_argument("--budget", type=float, default=1.0, help="pixel budget of the last run in megapixels; 0 skips it")
    ap.add_argument("--auto-images", type=int, default=0,
                    help="also run the automatic setting over this many pictures (it needs about 60 to settle)")
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--loop-guard", action="store_true", help="stream replies and stop one that repeats itself")
    ap.add_argument("--keep", default="", help="write the generated pictures into this (empty) folder and keep them")
    ap.add_argument("--json", default="", help="also write the results to this file")
    ap.add_argument("--mock", action="store_true", help="run against the simulated server (no model needed)")
    ap.add_argument("--serve-mock", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--mock-slots", type=int, default=4)
    ap.add_argument("--mock-scale", type=float, default=1.0, help="multiply the simulated server's times")
    ap.add_argument("--mock-flavour", default="lmstudio", choices=("lmstudio", "llamacpp", "plain"))
    ap.add_argument("--mock-loop-rate", type=float, default=0.0)
    ap.add_argument("--mock-penalty", type=float, default=0.2)
    ap.add_argument("--mock-context", type=int, default=0)
    args = ap.parse_args(argv)

    if args.serve_mock:
        with MockServer(slots=args.mock_slots, time_scale=args.mock_scale, flavour=args.mock_flavour,
                        loop_rate=args.mock_loop_rate, batch_penalty=args.mock_penalty,
                        context=args.mock_context) as server:
            print(server.base_url, flush=True)
            try:
                threading.Event().wait()         # until the benchmark that started it ends it
            except KeyboardInterrupt:
                pass
        return 0

    proc = None
    if args.mock:
        proc, base_url = _spawn_mock(args)
        model, key = "mock-vlm", ""
        print(f"SIMULATION: a modelled server with {args.mock_slots} parallel slot(s), times x {args.mock_scale}. "
              "These numbers describe the client pipeline, not your hardware.")
    else:
        if not args.server or not args.model:
            ap.error("give --server and --model for a real server, or --mock for the simulated one")
        base_url, model = args.server.strip(), args.model.strip()
        key = os.environ.get(args.api_key_env, "") if args.api_key_env else ""
        from inference.server_probe import probe
        info = probe(base_url, model, key)
        print(f"Server: {info.summary()}")
        for note in info.notes:
            print(f"  note: {note}")
        if info.loaded is False:
            print("  The model is not loaded: the server will load it on the first request (this takes a while "
                  "and is not part of the measurement only if you load it first).")
    levels = [int(x) for x in args.levels.split(",") if x.strip()]
    try:
        if args.keep:
            folder = Path(args.keep)
            folder.mkdir(parents=True, exist_ok=True)
            if any(folder.iterdir()):
                ap.error("--keep needs an empty folder")
            out = benchmark(base_url, model, key, args.images, folder, levels=levels,
                            budget_pixels=int(args.budget * 1e6), prompt=args.prompt, max_tokens=args.max_tokens,
                            timeout=args.timeout, auto_images=args.auto_images, check_facts=not args.mock,
                            loop_guard=args.loop_guard)
        else:
            with tempfile.TemporaryDirectory(prefix="tagscriber_bench_") as tmp:
                out = benchmark(base_url, model, key, args.images, Path(tmp), levels=levels,
                                budget_pixels=int(args.budget * 1e6), prompt=args.prompt, max_tokens=args.max_tokens,
                                timeout=args.timeout, auto_images=args.auto_images, check_facts=not args.mock,
                                loop_guard=args.loop_guard)
        if args.json:
            Path(args.json).write_text(json.dumps(out, indent=2), encoding="utf-8")
    finally:
        if proc is not None:
            proc.kill()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

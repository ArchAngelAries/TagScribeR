# Captioning speed: where the time goes and what was done about it

Scope: the Auto Caption tab's three providers (an OpenAI-compatible server such as LM Studio, local Transformers
models, the WD tagger), with the server path first because that is where most large jobs run. Written 2026-11-10.

Three kinds of number appear below and they are never mixed:

- **measured**: timed on this development machine, on generated pictures, with no model loaded (client work only);
- **simulated**: timed against a simulated server (`tools/bench_caption.py --mock`) whose behaviour is a stated model;
- **assumed**: a figure for a real model server that nobody has measured here. No real model was loaded and no
  request was sent to LM Studio for this work (nothing was listening on the usual port when it was looked for), so
  every speed-up on real hardware is still to be confirmed by `tools/bench_caption.py` on the owner's machine.

## 1. The short version

| What | Where it helps | Gain | Status |
|---|---|---|---|
| No fixed wait after each reply (the request thread was polled four times a second) | every server request | 125 ms per image on average; measured 266 -> 39 ms of client time per image against a server that answers at once | done, always on |
| A steady stream instead of chunks: a new request starts when one finishes | "Parallel" above 1 | simulated +29 % at 4 in flight when reply lengths vary widely | done, always on |
| The right number of parallel requests, found by measuring | servers with parallel slots (LM Studio, llama.cpp, vLLM, cloud) | simulated 2.2x from 1 to 4 requests in flight on a 4-slot server model (section 4); nothing on a 1-slot server, by design | done; needs a tick box (recommended on) |
| A pixel budget per picture | grid models (Qwen-VL family): image tokens fall with the pixels | assumed 1.2x to 1.5x per request; makes 4 parallel requests fit an 8k context | done; needs a control (default: none) |
| Stop a reply that repeats itself | models that sometimes loop | simulated +21 % when 10 % of replies run away; saves nothing otherwise | done; needs a tick box (default off) |
| Decode, resize and encode on four threads ahead of need; large JPEGs decoded at reduced size | large source files, the tagger, fast servers | measured 2x on 12 MP+ JPEG decode; 3.5x to 4x decode throughput on four threads | done, always on |
| Tagger: scores read with numpy, padding and resizing done ahead on helper threads | WD tagger | measured 16.7 -> 6.9 ms per batch of 8 for the scores; 10 to 38 ms per image moved off the model's thread | done, always on |
| Live images per minute, time left, pause and resume | every job | (not a speed-up) | done in the worker; needs UI |

## 2. Where the time goes

### 2.1 A request to a server, piece by piece

```
time per image = client work + wait for a free slot + vision encoder + prompt + reply
```

**Client work** (measured, 24-thread desktop CPU, generated pictures, Pillow 12):

| Step | 1 MP | 2 MP | 4 MP | 12 MP | 24 MP |
|---|---|---|---|---|---|
| Open + convert + shrink to 1536 px, PNG | 11 ms | 56 ms | 95 ms | 262 ms | 474 ms |
| the same, JPEG | 8 ms | 48 ms | 80 ms | 216 ms | 381 ms |
| the same, WebP | 31 ms | 99 ms | 194 ms | 571 ms | 1063 ms |
| JPEG (quality 92) + base64 for the server | 4 ms | 8 ms | 7 ms | 6 ms | 5 ms |
| Size of what is sent | 0.42 MB | 0.70 MB | 0.56 MB | 0.43 MB | 0.28 MB |

On four threads the 2 MP PNG case runs at 25 ms per image of wall time (40 images a second) and the 12 MP JPEG case
at 73 ms. Connection setup is paid once (one pooled keep-alive client per server); sending 0.7 MB to the same
machine takes well under a millisecond. Writing a caption (atomic, flushed to disk) is about 3 ms here and 10 to
50 ms on a busy NTFS volume (`IMPROVEMENT_IDEAS.md`, A4).

So for a server that needs seconds per image the client was never the bottleneck, with one exception that was
pure waste: after sending a request the old code looked for the answer every 250 ms, so each image waited 125 ms on
average for nothing. Against a simulated server that answers at once the old pipeline needs **266 ms** of client
time per image and the new one **39 ms** (25 ms at four in flight). That fixed cost is 3 % of a 4-second request and
20 % of a 0.6-second one.

**The server** (assumed figures for illustration: an 8B model at 4-bit on a 20 GB card; replace them with what the
benchmark prints):

- vision encoder: about 0.25 s per megapixel, one picture at a time (llama.cpp encodes pictures serially even
  when it generates for several requests at once);
- prompt: 1500 tokens per second; reply: 55 tokens per second for a single request;
- instruction text: about 40 tokens; caption: about 120 tokens.

**Image tokens** depend on the model family (`inference/image_budget.py`; estimates from the public model cards):

| Family | How pixels become tokens | 1536 x 1152 (1.8 MP) | 1.0 MP | 0.6 MP |
|---|---|---|---|---|
| Qwen3-VL, Qwen3.5 and later | one token per 32 x 32 pixels | 1728 | 972 | 588 |
| Qwen2-VL, Qwen2.5-VL | one token per 28 x 28 pixels | 2255 | 1271 | 768 |
| Pixtral / Mistral Small 3 | one per 16 x 16; a longer side is shrunk to 1024 by the model | 3072 | 3072 | 2352 |
| Gemma 3 | always 896 x 896 -> 256 tokens | 256 | 256 | 256 |
| LLaVA 1.5 | always 336 x 336 -> 576 tokens | 576 | 576 | 576 |
| JoyCaption (SigLIP 384) | always 384 x 384 -> 729 tokens | 729 | 729 | 729 |
| LLaVA-NeXT, InternVL, SmolVLM, MiniCPM-V | an overview plus fixed-size tiles, up to a limit | 2880 / 3328 / 1053 / 640 | fewer tiles | fewer tiles |
| Gemma 4 | a token budget (70 to 1120); the mapping here is a planning figure, check the model card | up to 1120 | | |

**Putting it together** for a Qwen3-style model with the assumed speeds:

| Picture sent | Vision | Prompt (image + text) | Reply (120 tokens) | Total | Share spent on the picture |
|---|---|---|---|---|---|
| 2.0 MP (1938 tokens) | 0.50 s | 1.32 s | 2.18 s | 4.00 s | 45 % |
| 1536 x 1152 (1728 tokens) | 0.44 s | 1.18 s | 2.18 s | 3.80 s | 42 % |
| 1.0 MP (972 tokens) | 0.25 s | 0.67 s | 2.18 s | 3.10 s | 29 % |
| 0.6 MP (588 tokens) | 0.15 s | 0.42 s | 2.18 s | 2.75 s | 20 % |

- **2 MP against 0.6 MP**: 4.00 s against 2.75 s, 1.45 times as many images per minute, if the model still sees
  what matters (section 5).
- **The instruction text**: 40 tokens at 1500 per second is 27 ms, 0.7 % of the request. Putting the instruction
  first (as the app does) lets a server's prompt cache skip it, which is worth that 0.7 %: real, but not a lever.
- **A 150-token limit against 400**: a normal 120-token caption costs the same under either. The limit only
  bounds a reply that does not stop: 400 tokens are 7.3 s, 150 are 2.7 s. If 5 % of replies run away, the mean
  reply time is 2.43 s with a limit of 400 and 2.21 s with 150: 6 % of the request. Stopping a looping reply
  after about 45 tokens (the loop guard) brings it to 2.11 s, and what is saved is a caption nobody wanted.
- **Parallel slots**: with k requests in a server that batches them, each stream runs slower but together they
  produce more: k / (1 + b (k - 1)) times one stream's speed, where b is how much one more stream slows the others
  (0 = not at all, 1 = no gain). The vision encoder does not take part (one picture at a time), so per image:
  `vision + (prompt + reply) x (1 + b (k - 1)) / k`. With b = 0.2 and k = 4 at 1536 x 1152 that is
  0.44 + 3.36 x 0.4 = 1.78 s, 2.1 times the single-request rate; with a 1.0 MP budget 0.25 + 2.85 x 0.4 = 1.39 s,
  2.7 times. b is the number the benchmark measures for a real server; 0.2 is an assumption.
- **A server with one slot**: extra requests wait in its queue. Each takes k times as long from the client's
  point of view and the rate is unchanged, so the right number is 1.

### 2.2 The context has to hold every request in flight

llama.cpp, and LM Studio on top of it, gives its parallel slots one context to divide or share. One request needs
`image tokens + instruction + reply limit + about 64` tokens:

| Picture | Reply limit | One request | 4 in flight |
|---|---|---|---|
| 1536 x 1536 (2304 tokens, the largest the default lets through) | 400 | 2808 | 11232 |
| 1536 x 1152 | 400 | 2232 | 8928 |
| 1.0 MP | 300 | 1376 | 5504 |
| 0.6 MP | 300 | 992 | 3968 |

A context of 4096 holds one full-size request and nothing beside it; 8192 holds four 1.0 MP requests with room to
spare. Every 1000 tokens of context cost very roughly 0.15 GB of memory for an 8B model (key / value cache at
16-bit; an estimate from the layer count, not a measurement). This is the second reason for a pixel budget: it is
what makes parallel requests fit without paying for a 16k context.

When the context is too small the server refuses the extra requests. The limiter treats that as overload: it
halves the number in flight, and the refused pictures are sent once more from the copy already encoded (test:
`test_real_client_backs_off_when_the_context_is_too_small_and_loses_nothing`).

### 2.3 Local Transformers models

`time per batch = processor (CPU) + prefill of the padded batch + (longest reply in the batch) x time per step`.
The batch waits for its longest reply and pads to its longest prompt, so the levers are the batch size, the image
size (the same token arithmetic as above), the reply limit, and how similar the pictures in a batch are. Nothing in
`inference/transformers_vlm.py` was changed: none of it can be verified without loading a model (standing rule 3).
What the path gained is shared: pictures are opened on four threads instead of one, a pixel budget is applied
before the processor sees the picture, and the worker reports the rate and can pause.

### 2.4 The WD tagger

Per batch of 8 (measured, no model loaded): reading 10,861 scores per image in a Python loop took 16.7 ms and now
takes 6.9 ms with identical results; padding and resizing a picture to 448 x 448 takes 10 to 38 ms (1 to 4 MP) and
now happens on the helper threads while the previous batch is in the model, as does opening the files (on four
threads instead of one). With the CPU provider the model itself needs hundreds of milliseconds per image, so this
is a few percent. With a GPU provider (tens of milliseconds per image) the old path was limited by opening and
preparing pictures on one thread, about 100 ms per image for 2 MP files, and the new one is not: up to about four
times the rate. That last figure is arithmetic from the measured parts; a run with a GPU provider has to confirm it.

## 3. The levers, one by one

### Done and always on (no change to what is captioned)

1. **No polling delay** (`openai_api._cancellable`): the request thread is joined, not polled.
2. **Steady stream** (`inference/pipeline.py`, `worker._process_stream`): the worker keeps N requests in flight
   and saves each caption as it arrives. With one in flight the order is exactly as before. With several, results
   arrive in the order they finish (the grid's badges and the review queue follow that order).
3. **Prepare ahead, bounded**: at most `N + 1` prepared pictures wait while `N` are in flight, so at most
   `2 N + 1` encoded pictures are held (under 1 MB each at the default size: 33 MB at the highest setting of 16),
   plus up to four pictures being decoded at full size at a time. Memory does not grow with the dataset.
4. **Encode once**: the picture is encoded on a helper thread and the same bytes are used if the request is sent
   again. Nothing is re-read or re-encoded on a retry.
5. **Large JPEGs decoded at reduced size** (`image_prep.load_rgb`): a JPEG that will be shrunk anyway is decoded
   at 1/2, 1/4 or 1/8 scale when that still leaves 1.5 times the target for the final high-quality resample.
   Measured: 12 MP -> 1024 px in 105 ms instead of 208 ms, 24 MP -> 1536 px in 191 ms instead of 378 ms; the
   result differs from the full decode by a PSNR of 49 dB or more, which is less than the JPEG sent on adds. PNG
   and WebP are unaffected (pixel-identical to before; test `test_load_rgb_limits_and_matches_the_old_result`).
6. **Nothing lost on abort**: answers that had already arrived when Abort was pressed are saved, not dropped.
7. **Tagger**: scores read with numpy (identical lists, test `test_tagger_scores_are_read_exactly_as_before`),
   preparation on helper threads, files opened on four threads.
8. **Rate, time left, pause** in the worker (`BatchWorker.stats`, `pause`, `resume`); section 7 lists the wiring.

### Done, behind an option (the default is today's behaviour)

9. **Automatic number of parallel requests** (`auto_parallel`, `inference/concurrency.py`). How it decides:
   - With N in flight all the time, images per second = N / (mean time per request). Up to the server's slot
     count the time per request barely changes; past it, requests queue, the time per request grows in
     proportion and the rate stays flat. The limiter compares N / latency between neighbouring values of N.
   - Reply length is the main noise. Each window's latency is read at a common reply length (a least-squares
     slope inside the window), and what the length does not explain is the window's measurement error: about 1 %
     while requests run side by side, 4 to 9 % once they queue. One more is kept only when it is better by 5 %
     plus 2.5 times that error, so a queueing level, which is never truly better, does not win by luck.
   - It starts at the number the user set, tries one more after 6 results (3 x N when that is more), and keeps
     going while it pays. Started above 1, it first checks that one fewer is really slower, so a number set too
     high for the server is walked down. It looks at both neighbours again after 120 results, then 240, 480 ...
   - A failure that points at overload (timeout, refused connection, 429, 5xx, "context exceeded") halves the
     number; a level that failed twice is not tried again in that job. It does not try one more when waiting in
     the server's queue could take a request past half the timeout. It tries one fewer when requests at a
     settled level become 1.6 times slower than when it was accepted.
   - It never exceeds the cap (8 for a local server, 16 for a cloud service, or what the user sets). A llama.cpp
     server is asked for its slot count (`GET /props`), which then is the cap.
   - Cost of searching: about 50 images on a 4-slot server before it has settled; in the event model the whole
     job runs at 97 % of the best fixed setting over 600 images, more over 5,000.
   - On a 1-slot server it tries 2 for six requests, sees no gain, returns to 1 and tries again only after 120,
     then 240 ... results. While it tries, requests take twice as long each but the rate is the same.
   Recommended default for the tick box: **on**. It cannot make the server use more memory (slots and context are
   fixed when the model is loaded) and it does not change what is asked.
10. **Pixel budget** (`CaptionRequest.max_image_pixels`): pictures are shrunk to at most this many pixels, in
    addition to "Max image size". Default: none. See section 5 for what it costs.
11. **Stop a reply that repeats itself** (`loop_guard`): the reply is streamed; when its tail is the same phrase
    four times over (or 81 identical characters) the connection is closed, which stops the server, and the
    caption is kept up to the first occurrence. The job summary counts these ("3 cut short (repeating)") and the
    log names the files. Default off: streaming is a different code path on the server and only a real run shows
    that the owner's LM Studio handles it as llama.cpp does.
12. **Picture before the instruction** (`image_first`): most vision models were trained with the picture first;
    the app's API path sends the instruction first (good for the prompt cache, worth 0.7 %). Whether picture-first
    gives better captions for a given model is a question for the benchmark's fact count. Default off.
13. **Tagger batch size** (`batch_size` in the tagger's spec; default 8 as before).
14. **Ask the server what it is** (`inference/server_probe.py`): read-only `GET`s that never load a model. Gives the
    kind of server, whether the model is loaded, its context length and, for llama.cpp, its slots, plus
    plain-language warnings when the context cannot hold the planned requests.
15. **A fitting reply limit per built-in instruction** (`prompts.suggested_max_tokens`): advice for the UI, never
    applied by itself.

### Proposed, not built

| Idea | Expected gain | Why not now |
|---|---|---|
| Caption byte-identical files once and reuse the caption (opt-in) | the share of exact duplicates, usually under 2 % | needs a choice in the UI and a rule for "Append" modes; the health scan already offers to remove duplicates |
| A first pass with the tagger or a small model, the large model only where the first pass is unsure | large when most pictures are easy | needs a confidence signal (token log-probabilities: llama.cpp reports them, LM Studio may not) and a second profile in the UI; opt-in by nature |
| Draft, then verify low-confidence captions with a short yes / no prompt | accuracy, at about +20 % time for the flagged share | same dependency on a confidence signal |
| One vocabulary pass over a tag-style job (spelling variants, singular / plural, order) | accuracy and consistency at no model cost | changes saved captions: belongs in the review step, as an offered change set |
| Group pictures of similar shape into the same batch (local models) | 5 to 20 % on mixed datasets at batch 2 and above | changes the processing order; needs a real model to show the gain |
| Run the processor for the next batch while the GPU generates (local models) | the processor's share, 20 to 100 ms per picture | fast tokenizers are not safe to use from two threads at once; needs a real run |
| Queue a caption job while a training run holds the GPU | no failed jobs, no memory fights | Train tab and Auto Caption tab wiring (`IMPROVEMENT_IDEAS.md` I5) |
| Agreement between two runs as an accuracy measure in the benchmark | a number for "how stable is this setting" | small; next step for `bench_caption.py` |

### Rejected

| Idea | Why |
|---|---|
| Reuse captions for near-duplicates (perceptual hash) | An 8 x 8 difference hash cannot tell two expressions, two colours of the same garment, or an edit pair apart: exactly the pictures a slider or edit dataset is made of. Wrong captions, silently. |
| Stop sequences by default | The model's own end token already ends the reply; a stop string can cut a legitimate caption. |
| Lower JPEG quality or another upload format | Encoding costs 4 to 8 ms and the upload is free on the same machine: nothing to win, fine detail to lose. `jpeg_quality` exists as an option. |
| Several pictures in one request | Captions bleed into each other and the context grows with every picture. |
| Replacing the HTTP client, HTTP/2, hand-made sockets | The client already keeps its connections; the overhead that existed was the polling delay, now gone. |
| A lower default reply limit, a lower default temperature | Both change results. The reply limit only matters for runaways (the loop guard handles those); a temperature of 0.2 to 0.5 with a fixed seed makes captions repeatable and a little more literal, and stays the user's choice. |
| Pinned memory, compiled graphs, a static cache for local models | Under 3 % or unverifiable without a model, and compiled graphs are a known risk on ROCm. |
| A shorter time limit per request than the profile's timeout | Slow is not broken on a 20 GB card running a 27B model. |

## 4. Simulations

All numbers in this section come from `tools/bench_caption.py --mock`: the real client code against a simulated
server in another process. They show what the **client pipeline** does given a server that behaves as modelled;
they say nothing about any real model.

The server model: N parallel slots; vision 0.25 s per megapixel, one picture at a time; one token per 32 x 32
pixels; prompt at 1500 tokens per second and replies (120 tokens on average, spread 30 %) at 55 tokens per second
for one stream, both slowed by 20 % of a stream for every further busy slot (the b = 0.2 of section 2.1); a prompt
cache per slot. Pictures: generated, 1 to 6 MP, PNG and JPEG alternating, sent at up to 1536 px, a fresh set for
every run. Times are not scaled. Two runs over different pictures differ by a few percent by themselves.

**Four slots** (`--mock --mock-scale 0.1 --images 24 --auto-images 48 --levels 1,2,4,8 --budget 0`; this one
run used the simulated server at one tenth of its modelled times, so the fixed client costs weigh ten times more
than they would against a real model: read the ratios between the "new" rows, not the old-to-new ratio):

| Pipeline | Images | Seconds | Images / min | In flight at the end |
|---|---|---|---|---|
| old: 1 at a time | 24 | 12.5 | 115.5 | 1 |
| new: 1 at a time | 24 | 8.1 | 178.3 | 1 |
| new: 2 at a time | 24 | 5.3 | 272.6 | 2 |
| new: 4 at a time | 24 | 3.6 | 397.2 | 4 |
| new: 8 at a time | 24 | 3.4 | 428.7 | 8 |
| new: automatic | 48 | 8.8 | 328.0 | 5 (still searching after 48 pictures: it kept 2, 3 and 4 and was trying 5) |

Going from 1 to 4 requests in flight on a 4-slot server gave 2.2 times the rate in this run; 8 added 8 % over 4.
The automatic setting climbed 1, 2, 3, 4 within 48 pictures and averaged 328 images a minute while searching.

**One slot** (`--mock --mock-slots 1 --images 16 --auto-images 60 --levels 1,2`):

| Pipeline | Images | Seconds | Images / min | In flight at the end |
|---|---|---|---|---|
| old: 1 at a time | 16 | 62.5 | 15.4 | 1 |
| new: 1 at a time | 16 | 57.2 | 16.8 | 1 |
| new: 2 at a time | 16 | 55.8 | 17.2 | 2 |
| new: 1 at a time, 1.0 MP budget | 16 | 50.5 | 19.0 | 1 |
| new: automatic | 60 | 213.8 | 16.8 | 1 (settled; it tried 2 once and came back) |

On a one-slot server the automatic setting ends at 1 and runs at the rate of the fixed setting of 1 (16.8 against
16.8); the 9 % over the old pipeline is the removed polling delay. Two at a time is within noise of one.

**Other simulated comparisons** (same model unless said):

| Question | Old | New |
|---|---|---|
| Client time per image against a server that answers at once, 1 in flight | 266 ms | 39 ms |
| ... 4 in flight | 77 ms | 25 ms |
| 4 slots, 4 in flight, reply lengths spread by 55 %: chunks of 4 against a steady 4 | 32.3 images/min | 41.8 images/min |
| 4 slots, 5 of 48 replies never stop (limit 400): loop guard off against on | 37.6 images/min | 45.4 images/min, 26 % fewer tokens generated |

**The limiter in a discrete-event model** (`tests/caption_sim.py`; 200 random seeds each, 600 requests, reply
lengths spread by 30 %): a 1-slot server ends at 1 in 200 of 200 runs; 2 slots at 2 in 200; 4 slots at 4 in 200;
a 4-slot server where batching brings nothing ends at 1 in 200; started at 6 on a 1-slot server it ends at 1 in 199
(2 in one); started at 8 on a 4-slot server it ends at 4 in 200. With the spread doubled to 60 % the worst case is
192 of 200. An 8-slot server whose streams slow each other by 20 % ends at 6, because the seventh and eighth add
less than 5 % each: that is the threshold doing its job, not an error.

## 5. Accuracy: what a smaller picture costs, and how to measure it without real data

- **Fixed-size models lose nothing.** Gemma 3 reads every picture at 896 x 896 and JoyCaption at 384 x 384 whatever
  is sent. A budget only saves upload and the server's own decoding (tens of milliseconds).
- **Grid models lose fine detail first.** Halving the pixel count shrinks every length by 0.71. Text needs roughly
  10 pixels of letter height to be read: a 20-pixel line in a 4 MP picture is 14 pixels at 2 MP, 10 at 1 MP and
  under 8 at 0.6 MP. Faces in a group shot, jewellery, small logos and thin patterns go the same way. Composition,
  subject, clothing, colours, lighting and style survive 0.6 MP for single-subject pictures.
- **A reasonable rule**: 1.0 MP for character, style and object datasets; 1.6 MP or full size for screenshots,
  documents, comics with speech bubbles, crowded scenes; never below the model's own minimum (llama.cpp prints a
  warning to the effect that Qwen-VL models want about 1000 image tokens for precise localisation).
- **Measuring it**: every benchmark picture has five known facts (background colour, shape, shape colour, a large
  word, and a line of small print about 2 % of the picture's height). The benchmark counts how many of them each
  caption mentions, per setting, so "1.0 MP budget: 96/120 facts against 110/120 at full size" is a number the
  owner can read off for their model. The small print is the first fact a budget destroys, on purpose.
- **What also helps accuracy at no cost**: tag hints from the tagger in the prompt (exists); the subject name
  (exists); a temperature of 0.2 to 0.5 and a fixed seed for repeatable captions; reasoning switched off for
  captioning (the app asks for that on local servers; the benchmark shows it in the tokens per caption).

## 6. What to set in LM Studio

Names are those of LM Studio 0.4; an older version may call them differently or not have them. None of this was
verified against a running LM Studio for this document: the benchmark is the judge.

1. **Load the model yourself, before the job**, in the Developer tab, and keep it loaded. A model loaded on demand
   by the first request is unloaded again after its idle time ("Auto unload unused JIT loaded models"): turn that
   off or raise the time, or the first picture after a pause pays for a full reload.
2. **GPU Offload: all layers.** If the model and its context do not fit in 20 GB, the remainder runs on the CPU and
   everything is several times slower. Pick a quantisation that fits with room for the context: 8B to 12B models
   at 4 to 6 bit are comfortable; 27B+ at 4 bit is tight before any context.
3. **Parallel requests**: in the model's load settings, "Max Concurrent Predictions" (LM Studio 0.4 with its
   llama.cpp engine 2.x). Start with 4. An older LM Studio answers one request at a time; TagScribeR's automatic
   setting then stays at 1 and nothing is lost.
4. **Context Length**: `parallel x (image tokens + reply limit + about 100)`. With TagScribeR's defaults (pictures
   up to 1536 px, reply limit 512) and 4 parallel requests that is about 12,000: set 12288 or 16384. With a
   1.0 MP budget and a reply limit of 300, 8192 is enough. Too small shows up as "context" errors; TagScribeR then
   lowers the number of parallel requests by itself.
5. **Flash Attention on, "Offload KV Cache to GPU Memory" on.** Both are plain speed for a model that fits.
6. **Vision models need their vision file.** A GGUF vision model comes with an "mmproj" file; LM Studio's own
   downloads include it. Without it the model cannot see the picture and the captions are inventions.
7. **Reasoning off.** A model that thinks before it answers spends hundreds of tokens per picture on text that is
   thrown away. TagScribeR asks local servers to switch it off; if captions arrive slowly and the benchmark shows
   several hundred tokens per caption, switch reasoning off in the model's settings in LM Studio.
8. **Runtime**: on an AMD card LM Studio offers a Vulkan and (where available) a ROCm llama.cpp runtime. Run the
   benchmark once with each: which is faster depends on the driver and the model.
9. **Nothing else on the card.** A training run, ComfyUI or a second loaded model takes the memory the context
   needs. On Windows an over-full card does not fail, it silently slows to a crawl (shared memory).

What a picture should be for common models: Qwen-VL family: 1.0 MP is a good default, full size for small text.
Gemma 3: anything from 896 px up is the same to the model. JoyCaption: anything from 768 px up. LLaVA 1.5: from
672 px up.

## 7. Wiring list for the Auto Caption tab

Nothing under `tabs/` was changed. Every item names the function to call; the texts are proposals in TagScribeR's
own wording. Settings keys with defaults are already in `core/config.py`.

| # | Control | Label | Default | Where it goes |
|---|---|---|---|---|
| 1 | Tick box, API options | "Find the best number of parallel requests" | on (`api.auto_parallel`) | `ProviderSpec.make("openai", ..., auto_parallel=True)` |
| 2 | Existing spin box | "Parallel requests:" (with 1 ticked: "Start with:") | 1 (`api.concurrency`) | `concurrency=` as today |
| 3 | Spin box 1 to 32, enabled with 1 | "At most:" | 8 (`api.max_parallel`) | `max_parallel=` |
| 4 | Combo box next to "Max image size" | "Picture detail:" with `image_budget.BUDGET_CHOICES` | "Full size" (`caption.max_image_pixels` = 0) | `CaptionRequest(max_image_pixels=...)`; applies to local models too |
| 5 | Hint label under 4 | text of `image_budget.describe_budget(model, max_side, max_pixels)` | | refresh when the model, the size or the detail changes |
| 6 | Tick box, API options | "Stop a reply that repeats itself" | off (`api.loop_guard`) | `loop_guard=True` |
| 7 | Tick box, API options, "Show all settings" only | "Send the picture before the instruction" | off (`api.image_first`) | `image_first=True` |
| 8 | Status line under the progress bar | "{per_minute:.1f} images/min, about {eta_text} left, {parallel_state}" | | `worker.stats.connect(...)`; the dict's keys are listed in `BatchWorker._report` |
| 9 | Button beside Abort | "Pause" / "Resume" | | `worker.pause()`, `worker.resume()`, `worker.paused` |
| 10 | Button beside "List" | "Check server" | | `server_probe.probe(url, model, key)` on a background task; show `info.summary()` and each line of `server_probe.advice(info, image_tokens, 60, max_tokens, parallel)` |
| 11 | Hint beside "Max tokens" | "Suggested for this instruction: {n}" | | `prompts.suggested_max_tokens(preset, tag_output, thinking)`; None = no hint |
| 12 | Spin box 1 to 64, tagger options | "Images per run:" | 8 (`tagger.batch_size`) | `ProviderSpec.make("wd_tagger", ..., batch_size=n)` |

Tooltips:

1. "TagScribeR starts with the number below, then tries one more and keeps it only if captions really come faster.
   A server that handles one request at a time ends at 1; a server with parallel slots (LM Studio's Max Concurrent
   Predictions, llama.cpp's -np) ends at its slot count. It backs off by itself when the server reports overload."
2. "How many requests are sent at the same time. More only helps when the server works on several at once."
3. "The automatic setting never sends more requests at a time than this."
4. "Shrinks each picture to at most this many pixels before it is sent, on top of Max image size. Models that turn
   every patch of the picture into tokens (the Qwen family) answer faster with fewer pixels; small text and fine
   detail are the first things lost. Models that read every picture at one fixed size lose nothing."
6. "Reads the reply while it is written and stops it when the model starts repeating the same phrase. The caption
   is kept up to the first repeat, and the summary says how many were cut. Saves the time a runaway reply would
   take up to the token limit."
7. "Most vision models were trained with the picture first. Try it if captions ignore parts of the instruction;
   compare with the benchmark."
9. "Pause: nothing new is started; captions being written finish and are saved. Resume continues where it stopped."
10. "Asks the server what it is and how the model is loaded. It only reads; it never loads a model."

The current tooltip of "Parallel" ("Raise for cloud APIs; keep 1 for a single local GPU") is no longer right for
LM Studio 0.4 and llama.cpp with parallel slots: replace it with tooltip 2.

Help Center paragraph (Auto Caption, "Speed"):

> **Captioning faster.** Three things decide how long a large job takes: how many requests the server works on at
> once, how large each picture is, and how long each caption is. *Parallel requests*: with "Find the best number of
> parallel requests" ticked, TagScribeR measures as it goes and settles on the number that really is fastest for
> your server: 1 for a server that answers one request at a time, more for LM Studio or llama.cpp with parallel
> slots. *Picture detail*: a pixel limit on top of Max image size. For models of the Qwen family fewer pixels mean
> fewer image tokens and a faster answer; keep "Full size" or "Detailed" when small text or fine detail matters.
> *Max tokens* bounds a reply that will not stop; "Stop a reply that repeats itself" ends such a reply early and
> keeps the caption up to the first repeat. The line under the progress bar shows images per minute and the time
> left; Pause holds the job without losing anything. In LM Studio: load the model before you start, set GPU Offload
> to all layers, set Max Concurrent Predictions to 4, and give the model a Context Length of at least 8192 (16384
> for full-size pictures). `tools/bench_caption.py` measures all of this on generated test pictures.

Hotkey proposal: `Ctrl+P` for Pause / Resume while a job runs (free in the Auto Caption tab today as far as the
tab's own `setup_hotkeys` shows; check the palette's bindings).

**Strings that reach the UI and must be shown as plain text** (the log view appends rich text today): every
`status` signal text (it contains file names), every `item_failed` reason and `JobSummary.fatal` (an
`InferenceError` text: the message, a line break, an arrow and the hint), `JobSummary.text()`, `limiter.describe()`,
`ServerInfo.summary()`, `ServerInfo.notes`, `server_probe.advice(...)`, `image_budget.describe_budget(...)`, and
of course every caption (model output). Server-supplied words inside error messages are cut to one line of at most
300 characters with angle brackets replaced and anything key-shaped blanked (`inference/textsafe.py`), but the
captions and `ServerInfo.models` are passed through exactly as the server sent them: they are data.

## 8. Safety of the API path

Asked for by the security review while this work was in progress; tests in `tests/test_caption_security.py`.

- **The key** is stripped of surrounding blanks, and a key with a blank inside, a line break, a control character
  or a non-ASCII character is refused before it can reach a header ("Paste the key again ..."). The HTTP library's
  own complaint about such a key quotes it, which is how a key could reach the log before.
- **Logs** get the exception's type and a redacted one-line message: no traceback and no raw exception text from
  the HTTP client.
- **Redirects**: the API client follows a redirect only on the same host (same scheme and port, or http to https).
  Before, the HTTP library dropped the key on a redirect to another host but still sent the picture and the prompt
  there. The server probe follows no redirect at all. This narrows one thing that worked: a server that redirects
  to a different host name is no longer followed; enter the final address.
- **Limits**: every reply (normal, error or streamed) is cut off at 8 MB and at the profile's timeout for the whole
  body; a single streamed event may be 1 MB; a compressed reply is refused (none is asked for, and a small one can
  unpack to gigabytes). The probe reads at most 2 MB and takes at most about twice its timeout per request.
- **Messages** never contain the key, a URL's user name, password or query string, or markup from the server.

## 9. Running the benchmark

```
venv\Scripts\python.exe tools\bench_caption.py --server http://localhost:1234/v1 --model <id as LM Studio shows it> --images 24
```

It generates its own test pictures in a temporary folder (deleted afterwards; `--keep <empty folder>` keeps them),
prints what the server says about itself, then runs: the old pipeline one at a time; the new pipeline at 1, 2, 4
and 8 at a time, stopping when a step brings under 10 %; the new pipeline at the best of those with a 1.0 MP
budget. For each it prints images per minute, failures, the number in flight and the facts found, and at the end
the best number of parallel requests and what a 5,000-image job would take. Useful additions:

- `--auto-images 80`: also run the automatic setting over 80 pictures and print its decisions;
- `--loop-guard`: stream replies in the new-pipeline runs (to check that streaming works with the server);
- `--budget 0.6`, `--levels 1,2,3,4,6`, `--max-tokens 200`, `--prompt "..."`, `--json result.json`;
- `--api-key-env NAME`: read an API key from that environment variable (it is never printed).

With `--images 24` and four levels this sends about 150 requests: around ten minutes for a model that needs four
seconds per picture. Load the model in LM Studio first, and do not run it while a training run uses the card.

`--mock` runs the same comparison against the simulated server (`--mock-slots`, `--mock-scale`, `--mock-penalty`,
`--mock-loop-rate`, `--mock-context`, `--mock-flavour lmstudio|llamacpp|plain`).

## 10. What only a run on the real server can confirm

1. That LM Studio on this machine runs requests in parallel at all, how many, and what the batching gain is (the
   number b of section 2.1). Everything about parallel requests rests on this.
2. The real speeds: vision seconds per megapixel, prompt and reply tokens per second, and with them the true share
   of the picture in a request and the true gain of a pixel budget.
3. Image tokens per picture as the server's runtime counts them (it may clamp differently from the model card).
4. That a streamed reply closed early really stops the generation in LM Studio (it does in llama.cpp), and that
   LM Studio accepts the "no reasoning" request field.
5. The accuracy cost of each budget for the owner's model: the benchmark's fact count, above all the small print.
6. Whether picture-first helps or hurts a given model.
7. The tagger with a GPU provider, and anything about local Transformers models: none of it was run.
8. That the refused-redirect rule does not break a setup that relied on a redirect to another host name.

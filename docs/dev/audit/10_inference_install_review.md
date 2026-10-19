# Audit 10 - inference, installer, network / secrets / supply chain (adversarial review)

Read-only review of `inference/`, `tools/install.py`, the launchers (`install.bat`, `install.sh`, `start.bat`,
`start.sh`, `update.bat`), `requirements*.txt`, and every network / secret / supply-chain path they touch
(`core/secrets.py`, `core/hardware.py`'s env defaults, `core/log.py`, the model-folder scan, Hugging Face downloads,
the API key flow in `tabs/caption.py`). Training and the other tabs are covered by other audits and were only read
where this area is called from them.

What was run: the inference, secrets, hardware, config and no-GPU tests (35 passed) and four scratch scripts against
the installed libraries (Transformers 5.19, openai 3.26, httpx 0.28, huggingface-hub 1.33, torch 2.14 CPU). No model
weights were loaded, no dataset was touched, no network call was made; third-party behaviour was checked by reading
the installed library source, with line numbers given. "Reproduced" below means a scratch script showed it;
"reasoned" means it follows from the code and library source but was not executed.

Severity scale: critical / high / medium / low. Nothing critical or high was found. Few findings are listed on
purpose: everything speculative was dropped.

---

## Findings, most severe first

### 1. [medium, security] "Allow custom model code" is one global switch, and the custom-code gate misses the case that matters

**Where.** `tabs/caption.py:304-305, 390, 422, 866-870`; `core/config.py:67` (`local.trust_remote_code`);
`inference/transformers_vlm.py:43-51` (`needs_remote_code`), `94-100` (the gate), `107` (`common`);
`inference/models.py:199-204` (`download_model` fetches `*.py`).

**What happens.** The checkbox is stored in `settings.json` and passed as `trust_remote_code` to *every*
`AutoProcessor` / model load from then on, for every folder the scan finds in `models/`, the extra model dirs and the
Stability Matrix `Models/LLM` folder, and for every HF repo id. The only app-side gate, `needs_remote_code()`,
returns False whenever `config.json` names a model type Transformers already knows:

```
needs_remote_code({"model_type": "qwen3_vl", "auto_map": {...}})  -> False     (reproduced)
```

but Transformers itself does the opposite when the flag is True: `auto_factory.py:227` and `:385`
(`has_remote_code and trust_remote_code and not explicit_local_code`) load the class named in `auto_map` from the
folder's `.py` file *in preference to* its built-in `qwen3_vl` class. So a folder that says `model_type: qwen3_vl`
plus an `auto_map` runs its own `modeling_*.py` with no "uses custom code" message at all. The gate is also skipped
entirely for repo ids (`read_local_config` returns `{}` for anything that is not a directory).

**Scenario.** A user enables the box once for one model that needs it (an InternVL / MiniCPM-V style repo in their
Stability Matrix folder), and it stays on. Later they drop a third-party "quantised JoyCaption" folder into
`models/` or point an extra dir at a download folder. Selecting that model executes whatever Python it ships, as the
user, the moment Load runs. `download_model` also pulls `*.py` for every catalog model although none of the nine
catalog entries needs custom code, so the code is already on disk for anything the Hub repo later adds.

**Reproduced / reasoned.** `needs_remote_code` result reproduced with a scratch script; the Transformers preference
for remote code read from `transformers/models/auto/auto_factory.py` lines 212-245 and 363-397 of the installed
5.19.0. Not executed with a real model (rule 3).

**Fix (minimal).**
- Make trust per model: keep a set of trusted paths / repo ids in settings (`local.trusted_models`), set
  `trust_remote_code=True` only for the model being loaded when it is in that set, and prompt (once, naming the file)
  when a model has an `auto_map` and is not yet trusted. Drop the global checkbox or turn it into "ask me".
- In `needs_remote_code`, treat any `auto_map` as custom code (`return bool(cfg.get("auto_map"))`): when the flag is
  on Transformers will run it regardless of `model_type`.
- Remove `"*.py"` from `allow_patterns` in `download_model` unless the catalog entry says the model needs it (none
  does today).
- Help text (`tabs/help_content.py:256, 698`) should say that the switch applies to every model, not just the one
  selected, until this is changed.

### 2. [medium, supply chain] The Windows ROCm bitsandbytes wheel is installed from a personal GitHub release with no integrity check

**Where.** `tools/install.py:57-61` (`BNB_ROCM_WIN_WHEEL`), `225-227` (`uv([BNB_ROCM_WIN_WHEEL])`), and again on
every `update.bat` run where bitsandbytes is missing (`288-290`).

**What happens.** `uv pip install <https://github.com/<account>/.../bitsandbytes-...whl>` has no hash, so the
installer trusts whatever bytes that URL serves. GitHub release assets can be deleted and re-uploaded under the same
file name by the account owner, or by anyone who gains the account. The wheel is imported inside the app process
(8/4-bit loading, the 8-bit optimiser), so a swapped wheel is code execution on every Windows AMD install and every
update.

**Reproduced / reasoned.** Reasoned from the installer; the URL itself was not fetched. The comment in the file
says this is the same wheel Fizgig installs, so this is as much a question for the owner as a finding (rule 1):
keep the source, add verification.

**Fix (minimal).** Record the wheel's SHA-256 next to the URL and install through a one-line requirements file with
`--require-hashes` (`bitsandbytes @ <url> --hash=sha256:<digest>`), or download with `urllib`, verify the digest,
then `uv pip install <local file>`. Print the digest in the install log so the owner can compare it with Fizgig's.
The same would apply to any future wheel URL in `install.py`.

### 3. [medium, bug] A partially downloaded catalog model shows as installed; the promised resume is unreachable

**Where.** `inference/models.py:94-107` (`_is_vlm_folder`: config + processor config + *any* `*.safetensors`),
`171-176` (`catalog_local_path`); `tabs/caption.py:474-484` (`_fill_models`), `503-505` (`is_remote` → Download
button disabled), `541-545`, `566-569` ("partial files are resumed next time").

**What happens.** `snapshot_download(local_dir=...)` writes finished files into place one by one. If the download
is interrupted after `config.json`, `preprocessor_config.json` and the first shard (an 18 GB model takes a while),
the folder satisfies `_is_vlm_folder`, so on the next start it is listed under "Installed", `catalog_local_path`
returns it, the Download button reads "Available" and is disabled, and loading fails inside Transformers with a
missing-shard error. The user has no UI route to resume except deleting the folder; the status text that says
partial files are resumed is wrong for this case.

**Reproduced.** Scratch folder with `config.json`, `preprocessor_config.json`, `model.safetensors.index.json`
listing two shards, one shard present and the other still `.incomplete` under `.cache/huggingface/download/`:
`_is_vlm_folder -> "qwen3_vl"`, `catalog_local_path -> <folder>`, `discover_local_models` lists it.

**Fix (minimal).** In `_is_vlm_folder` (or a sibling `is_complete`), when `model.safetensors.index.json` exists
require every file in `weight_map` to be present, and treat a non-empty `<dir>/.cache/huggingface/download/` as
incomplete; have `_fill_models` keep the Download button enabled (label "Resume download") for an incomplete folder,
and route it to `download_model` with the same target. `hf_fully_cached()` already does the shard check for the HF
cache; reuse that logic.

### 4. [medium, bug] API captioning: 180 s timeout × 3 retries that duplicate work, cannot be cancelled, and can freeze the app on exit

**Where.** `tabs/caption.py:876-877` (`timeout=180.0, max_retries=3`, hard-coded; the profile's own `timeout` and
`max_retries` fields in `core/secrets.py:29-30` are never used); `inference/openai_api.py:84-86` (client),
`97-122` (`_one`: cancel checked only *before* the call); `inference/manager.py:100-105` (the session lock is held
for the whole job); `inference/worker.py:299-304` (`cancel_all` waits 15 s per thread); `main.py:203-205`
(`manager().unload_all(blocking=True)` on the UI thread).

**What happens.** The openai SDK retries timeouts: `openai/_base_client.py:1150-1160` (`except
timeout_exceptions(): if remaining_retries > 0 ... continue`). LM Studio, llama-server and Ollama process requests
one at a time, so when one image takes longer than 180 s (a 27B GGUF on a mid-range GPU, a large "Max tokens", a
reasoning model) the client gives up, the server keeps generating the abandoned request, and the retry queues behind
it, so the next attempt is slower still: the batch degrades into a cascade of duplicate generations and finally one
`APITimeoutError` per image after ~12 minutes each. Abort cannot interrupt any of it: `_one` only checks `cancel`
before the request, so "Stopping after the current image…" can mean ten more minutes. Closing the app then runs
`cancel_all` (returns after 15 s with the worker thread still inside the request) and `unload_all(blocking=True)`,
which blocks the UI thread on the manager lock until the request chain returns: the window stops repainting and
Windows offers "close the program". A second job started from another tab (captioning plus Auto Tag) blocks on the
same lock with no status and no way to cancel the wait.

**Reproduced / reasoned.** Reasoned from the code and the installed SDK source; not run against a server (no
network). The 35 inference tests pass because the fake provider returns instantly.

**Fix (minimal).**
- Use the profile's `timeout` / `max_retries` instead of the constants, default `max_retries=0` for local URLs
  (`is_local_url`), and keep retries for cloud hosts only. The SDK already retries 429 / 5xx; timeouts on a local
  server are never worth a retry.
- Make the request cancellable: `stream=True` and check `cancel` between chunks (the server stops on disconnect for
  llama.cpp-based servers), or at least run the request with a shorter poll loop.
- In `closeEvent`, give `unload_all` a bounded wait (`acquire(timeout=...)`) and leave the model to process exit if
  the job is still inside a request, instead of blocking the UI thread.
- In `ModelManager.session`, poll `cancel` while waiting for the lock (`acquire(timeout=0.5)` in a loop) so a queued
  second job can be aborted.

### 5. [low, bug] Any unexpected exception from `provider.generate` aborts the whole batch

**Where.** `inference/worker.py:190-197` (only `Cancelled` and `InferenceError` are caught around `generate`),
`130-132` (everything else becomes `summary.fatal`); `inference/openai_api.py:123-133` (the reply parse only guards
`AttributeError / IndexError / TypeError` on the attribute path, not on the value).

**What happens.** The worker's documented guarantee ("one failed image never aborts the batch") only holds for
exceptions that are already `InferenceError`. The API provider turns SDK errors into one, but a reply whose
`choices[0].message.content` is a list of parts, or anything but a string, raises `TypeError` inside
`captions.clean_model_output` and escapes `generate`; the job ends with "Unexpected error" and every remaining image
is left uncaptioned. The same path is open for the Transformers provider: a `RuntimeError` / `ValueError` from
`model.generate` for one odd image ("image features and image tokens do not match" is the classic) kills the batch.

**Reproduced.** Fake client whose reply has `content=[{"type": "text", "text": "a cat"}]` and `content=7`: both
escape `OpenAICompatible.generate()` as `TypeError`. The openai SDK builds response models without validation, so a
non-conforming server reply reaches this code unchanged.

**Fix (minimal).** In `_one`, wrap the whole parse in `try/except Exception` and coerce list content
(`"".join(part.get("text", "") for part in content if isinstance(part, dict))`) before returning
`InferenceError("The API returned a malformed response.")`. In `worker._process`, catch `Exception` from
`provider.generate` (after `Cancelled`), log it with `log.exception`, and fail the chunk's images individually.

### 6. [low, supply chain] Hugging Face downloads are not pinned to a revision

**Where.** `inference/models.py:202` (`snapshot_download`, no `revision`), `183` (`model_info`, size shown for
"latest"); `inference/wd_tagger.py:122-126` (`hf_hub_download` of `model.onnx` / `selected_tags.csv`, no
`revision`); `inference/models.py:15-27` (`CatalogModel` has no revision field).

**What happens.** The catalog says "verified against the Hub on 2026-10-02", but a download fetches whatever the
repo's `main` points at on the day. A repo owner (or a compromised owner account) can change weights, tokenizer,
chat template or add `auto_map` + a `.py` file; with finding 1 in its current shape that `.py` is downloaded and, if
the switch is on, executed. With the switch off the blast radius is bad weights or a bad template (Transformers
renders chat templates in a sandboxed Jinja environment), so this is low on its own.

**Reasoned.** Code reading only.

**Fix (minimal).** Add `revision: str = "main"` to `CatalogModel` and a revision per `KNOWN_TAGGERS` entry, pass
`revision=` to `snapshot_download`, `model_info` and `hf_hub_download`, and record the commit hash in the download
log line. Bumping a pin is then a reviewed change in the repo rather than an implicit one at download time.

### 7. [low, security] A profile's API key stays in the masked field while the URL is edited, so it can be sent to another host

**Where.** `tabs/caption.py:605-611` (`apply_api_preset` fills URL, key, model), `632` and `871-877` (whatever is
in the fields is used); `inference/openai_api.py:84` (the key becomes the Bearer token for `base_url`).

**What happens.** The key field uses `QLineEdit.Password`, so after choosing a cloud profile and then typing a
different URL (switching to a LAN LM Studio box, say `http://192.168.x.y:1234/v1`), the cloud key is still there,
invisible, and is sent as `Authorization: Bearer` to the new host, over plain HTTP if that is what was typed. The
redirect case is already covered: httpx strips `Authorization` on cross-origin redirects
(`httpx/_client.py:554-556`), and the openai client enables `follow_redirects` (`_base_client.py:937`), so only the
typed host sees the key.

**Reasoned.** Code reading only.

**Fix (minimal).** Clear the key field when the URL's host changes away from the selected profile's host (or reset
the profile combo to "Select profile…" and clear the key), and show a one-line note when a non-local `http://` URL is
combined with a non-empty key.

### 8. [low, bug] `items_started` is never emitted, so the "working" badges never appear

**Where.** `inference/worker.py:104` (declared), `186-193` (`_process` goes straight to `generate` without
emitting); `tabs/caption.py:955` (connected to `_mark_many(keys, JOB_WORKING)`).

**What happens.** Images go from "queued" to done / failed with no "working" state; the status label is the only
live indication. Verified by grep: the only occurrences are the declaration, the relay and the connection.

**Fix.** `self.items_started.emit([str(p) for p in ok_paths])` just before `provider.generate(...)`.

### 9. [low, UX / bug] Gated catalog models cannot be downloaded from the app, and the failure is a raw error

**Where.** `inference/models.py:27` (`gated: bool = False`, never set to True, never read), `48-52` (the Gemma 4
entries); `tabs/caption.py:566-569` (`Download failed: <exception>`); no HF token / login anywhere in the app or
docs (grep for `HF_TOKEN`, `hf auth`, `GatedRepo` finds nothing).

**What happens.** Google's Gemma repos on the Hub require accepting a licence; without a token on the machine
`snapshot_download` raises `GatedRepoError` (HTTP 401) and the user sees the raw message with no hint that they
need `hf auth login` or `HF_TOKEN`. Not verified for the gemma-4 repos specifically (no network calls were made),
but the `gated` field exists for exactly this and is dead.

**Fix (minimal).** Set `gated=True` on the Gemma entries, show a note next to the Download button for gated
models, and map `GatedRepoError` / 401 in `failed()` to "This model needs a Hugging Face login that has accepted
its licence: run `hf auth login` in the venv or set HF_TOKEN, then retry."

---

## Nits (not worth a line each in the ranking)

- `tools/install.py:177-182`: `ensure_uv` is satisfied by a system `uv` on PATH, but every install step runs
  `python -m uv`, which fails when the venv lacks the module. Only `tools/install.py` run by hand is affected
  (`install.bat` / `install.sh` install `uv` into the venv first).
- `inference/models.py:110-118`: `_folder_size` adds `*.bin` and `*.safetensors`, so a folder that ships both shows
  twice its size.
- `inference/models.py:190-196` vs `199-204`: the size shown before a download (`_wanted`) and the files actually
  fetched (`allow_patterns`) are different filters; the shown figure can be off by README images and such.
- `inference/openai_api.py:25-30`: `is_local_url` misses `127.0.0.2`-`127.255.255.255` and `host.docker.internal`;
  only affects the `enable_thinking` default.
- `tabs/caption.py:651-652`: server-controlled error text is shown verbatim in a message box; harmless, but a
  malicious server can put any sentence there.

---

## Checked and found sound (so the next reviewer can skip them)

- **No TLS verification is disabled** anywhere (`verify=False`, custom SSL contexts: none). `HF_HUB_DISABLE_TELEMETRY`
  is set; no other outbound host is contacted besides the user's API URL and the Hub.
- **Secrets.** Keys live in the OS credential store via `keyring`, the profile file holds only non-secret fields,
  and the plaintext fallback is warned about and rewritten away once a store works (`core/secrets.py:113-123`;
  `tests/test_secrets.py`). No log line, status line, window title or screenshot path includes a key: `ProviderSpec`
  (which carries `api_key` in its options) is never logged or `repr`'d; `display_name` for the API provider is
  `model @ base_url`. `httpx` / `openai` / `httpcore` loggers are capped at WARNING (`core/log.py:42-43`).
  `tools/make_screenshots.py` points `TAGSCRIBER_USER_DATA` at a temp dir, so real profiles never reach a screenshot.
  `tools/check_release.py` flags `sk-`, `hf_`, bearer tokens and the user-data files.
- **Pickle.** `_is_vlm_folder` accepts `*.bin`, and Transformers 5.19 loads them with `torch.load(weights_only=True)`
  (`modeling_utils.py:189, 322, 366`); the pinned torch builds (2.10 CUDA, 2.12 ROCm) are past the
  `weights_only` bypass CVEs. The tagger loads ONNX (no code execution). `download_model`'s patterns exclude
  `*.bin` / `*.pt`.
- **Offline loading.** Cached repo ids are loaded with `local_files_only=True`, which also disables the
  Mistral-tokenizer Hub probe (`tokenization_utils_tokenizers.py:1420-1421`), so a cached model loads without a
  network round-trip.
- **Wrong-image writes.** Results are matched to paths by position in the same lists that were decoded
  (`worker.py:179-204`); the Transformers provider keeps order through `apply_chat_template` / `batch_decode`, the
  API provider through `pool.map`, the tagger through `np.stack`; the OOM fallback re-generates one image at a time in
  the same order; the UI applies results by path (`on_item_done(path, text)`, `on_tagger_finished(path, ...)`) with a
  folder-generation guard. No ordering bug found.
- **Cancellation inside Transformers generation** works through the stopping criterion, and a cancelled batch's
  partial output is discarded, not saved. **Model unload** drops the only references and empties the allocator
  cache; a failed heavy load is unloaded by the manager (`manager.py:74-81`).
- **Device / dtype selection.** `cuda:N` strings on both vendors, bf16 probed not assumed, fp16 refused on CPU,
  math-only SDPA forced on RDNA1/2, TF32 only on CUDA. Consistent with `docs/ARCHITECTURE.md`.
- **Installer and launchers.** All subprocess calls use argument lists (no `shell=True`, no user text in commands);
  `install.bat` quotes every path; `update.bat` is `--ff-only`. The CUDA path uses `--index-url cu128` first with
  uv's default first-index strategy, so `torch` cannot be satisfied from PyPI; `requirements.txt` never lists torch,
  so a later `uv pip install -r` cannot replace the GPU build. Package names in `requirements.txt` are the canonical
  PyPI names (`nvidia-ml-py`, `hf_xet` → `hf-xet`, `triton-windows` is the same community build Fizgig uses).
- **Model-folder scan.** `discover_local_models` only reads `config.json` and stats weights; depth-limited, skips
  dot-folders, never imports or executes anything. The Stability Matrix paths are read-only guesses.
- **Batched generation with Transformers 5.x** passes `processor_kwargs={"padding": True}`, which is the 5.x
  signature (`processing_utils.py:1999-2073`); left padding is set on both tokenizer and processor.

## What only a real run can confirm

Everything in findings 1, 4 and 9 that involves a real model, a real server or the Hub: that a specific gemma-4
download is refused without a token, how long LM Studio takes to drain abandoned requests, and that a planted
`auto_map` folder is in fact imported by Transformers (the code path was read, not executed, per rule 3).

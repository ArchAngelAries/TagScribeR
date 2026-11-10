# Audit 14 - security, privacy and safety review of the whole app

Date: 2026-11-10. Branch `port/v8-wave2` (head `ec12fb9` plus the uncommitted theme work, which was read but not
judged line by line). Scope: `core/`, `inference/`, `tabs/`, `training/` (shared layer, `ft.py`, `modules/rotation.py`,
`modules/safetensors_io.py`, `families/moss_image/base_check.py`, the fine-tune and resume code), `tools/`, the
launchers, the repository itself. Audits 08, 09 and 10 were read first; their fixed findings are not repeated, and a
sample of those fixes was checked again (section "Earlier fixes checked again").

How findings were confirmed: **RUN** = reproduced with a small script on generated files, an isolated `user_data`, an
offscreen `QApplication` and, for the network items, HTTP servers started on `127.0.0.1` by the script itself.
**READ** = from the source only. **SUSPECTED** = the mechanism was confirmed up to a step that could not be run here
(it would need a network request to another machine, a real model, or a crash of the app). No training was run, no
model weights were loaded, no real dataset, image or settings file was used.

Note on moving code: while this review ran, another change set was being written in `inference/` (uncommitted:
`worker.py`, `image_prep.py`, `openai_api.py`, `manager.py`, `base.py`, `wd_tagger.py` and the new `concurrency.py`,
`image_budget.py`, `loopguard.py`, `pipeline.py`, `server_probe.py`). Findings H2, H4, M4, L3 and L12 were confirmed
on the committed files and then read again in the working copy, where they still apply; for those the line numbers
are given as "HEAD / working copy". L14 is about the new, uncommitted `server_probe.py`. `tabs/theme.py` was also in
progress.

Severity scale (the owner's): CRITICAL = remote or file-borne code execution, credential theft, silent data loss.
HIGH = private data in files people share, deletion without a way back, a crash a hostile file can trigger at will.
MEDIUM, LOW below that.

Result: **no CRITICAL finding.** 5 HIGH, 7 MEDIUM, 14 LOW.

---

## HIGH

### H1. An API key comes back after it was cleared, and is then sent to the new server address (RUN + READ)

`core/secrets.py:124` only writes to the credential store when the key is non-empty (`if key and not set_secret(...)`),
so saving a profile with an empty key leaves the old key in the store. `core/secrets.py:92` then reads it back
(`get_secret(...) or d.get("api_key", "")`).

Scenario: a profile "Cloud" holds a key for a cloud API. The user changes the URL to another machine. The tab clears
the key box, as audit 10 finding 7 asked (`tabs/caption.py:692-703`). The user saves the profile under the same name
(`tabs/caption.py:708-720`). `reload_profiles` (`:668-677`) loads the profile again, `apply_api_preset` (`:679-687`)
puts the old key back into the box and marks it as belonging to the new host. The next "List" or captioning run sends
`Authorization: Bearer <cloud key>` to the new address, over plain http if that is what the URL says. The same happens
without any URL change when a user simply wants a key gone: clearing the box and saving does not remove it; only
deleting the profile does.

Confirmed at the store level by running (key restored, credential store still holds it); the tab flow was read.

Smallest fix: in `ApiProfileStore.save` call `set_secret(f"api:{name}", key)` for every profile (an empty key deletes
the entry; `set_secret` already does that). Belt and braces: store the host the key was entered for
(`key_host`) in the profile file and do not load a key whose host differs from the profile's URL.
Test: save a profile with a key, save it again with an empty key and another URL, reload: the key is empty and the
fake store has no entry.

### H2. An API key with a trailing space ends up in `user_data/logs/tagscriber.log` (RUN)

The key is used exactly as typed (`tabs/caption.py:729`, `:976`; `inference/openai_api.py:141` at HEAD, `:174` in
the working copy). A key pasted with a
trailing space or line break is not a legal HTTP header value, the HTTP library refuses it with
`LocalProtocolError: Illegal header value b'Bearer <the key> '`, and `inference/openai_api.py:221` (HEAD; `:347`
in the working copy: `log.debug("API call failed", exc_info=True)`) writes the whole exception chain to the log file (the file handler is
at DEBUG, `core/log.py:30`). The message on screen is only "Could not connect to the API server", so the user has no
idea the key is now in a file that `start.bat` points people to when something goes wrong.

Confirmed by running the same client call and the same logging call against a loopback server: the logged traceback
contains the key.

Smallest fix: strip the key where it is read (the three places above and `ApiProfile` on save); refuse a key that
contains whitespace or non-printable characters with a plain message before any request; add a `logging.Filter` on
both handlers that replaces `Bearer \S+`, `sk-...`, `hf_...` and similar with `<redacted>` (so no later code path can
do this again); log `type(e).__name__` and the scrubbed message instead of `exc_info` for API failures.
Test: a provider built with `"key "` fails with the plain message, and a log capture never contains the key.

### H3. "Clean metadata" leaves identifying data in the file, and the audit then says "Nothing sensitive found" (RUN)

A JPEG was built with EXIF (GPS, Make, MakerNote, body serial, time-zone offset, an IFD1 thumbnail), XMP, extended
XMP, IPTC (city, by-line), a comment, an MPF segment, an APP11 (C2PA / JUMBF) segment and bytes after the end-of-image
marker. After `strip_metadata`:

| what | default options (location, device, AI data, XMP) | "everything" |
|---|---|---|
| GPS IFD, Make, body serial, EXIF thumbnail, main XMP, comment | removed | removed |
| MakerNote (tag 0x927C: camera-internal serial numbers, owner name on some makes) | **kept** | removed |
| OffsetTime / OffsetTimeOriginal (the time zone: a coarse location) | **kept** | removed |
| IPTC in APP13 (city, sub-location, country, by-line, contact) | **kept** | removed |
| Extended XMP (`http://ns.adobe.com/xmp/extension/` APP1 segments) | **kept** | **kept** |
| APP2 MPF (index of further pictures in the file) | **kept** | **kept** |
| APP11 (C2PA / JUMBF content credentials: signer, tool, edit history) | **kept** | **kept** |
| Everything after the end-of-image marker (motion-photo video, vendor trailers, MPF pictures) | **kept** | **kept** |

For PNG, bytes after `IEND` survive both modes; text chunks under keys the AI list does not know (`Description`,
`Source`, ...) survive the default mode by design.

Where: `core/metadata.py:32-33` (`DEVICE_TAGS` has no MakerNote, `TIME_TAGS` no offset or sub-second tags),
`:387-405` (`_rewrite_jpeg` matches only the main XMP namespace, drops APP13 only for "everything", keeps every
other APPn), `:338-355` (`_jpeg_segments` returns everything from start-of-scan to the end of the file as one piece),
`:408-417` (`_png_chunks` keeps trailing bytes). `privacy_findings` (`:142-170`) does not look at any of these, so
after a default clean the Metadata tab reports "Nothing sensitive found" for a file that still names the city, the
photographer and the camera body. The confirmation text for "everything" promises "all metadata except orientation
and color profile" (`tabs/metadata.py:511-520`).

Scenario: a user cleans photos before publishing a dataset, sees the green tick, and uploads files that still carry
the above.

Smallest fix:
* "everything" on JPEG: keep an allow-list (APP0 JFIF, the ICC APP2 unless `remove_icc`, APP14 Adobe because it tells
  decoders the colour transform, the tables, frame and scan segments) and drop every other APPn and COM; cut the file
  after the last end-of-image marker. On PNG: stop at `IEND`.
* default mode: `device` also drops MakerNote; `location` also drops the OffsetTime tags; `xmp` also drops the
  extended-XMP segments.
* `privacy_findings`: report MakerNote, IPTC, extended XMP, APP11, MPF and trailing bytes, so the audit never says
  clean while they are there.
* Question for the owner: should `location` remove IPTC as a whole (it also holds copyright and by-line, which a user
  may want to keep), or only its location datasets? Recommendation: remove the whole block under `location`, and say
  so in the tick box's tooltip. Dropping an MPF block and trailing bytes also removes the extra pictures of a
  multi-picture file; that is what "everything" should mean, but it is a behaviour change to state in the Help text.

Test: the generated file above; after each mode none of the markers the mode covers is in the bytes, the picture
still decodes, and `privacy_findings` lists what is left.

### H4. Text from image files and from API servers is rendered as HTML (RUN for a local file, SUSPECTED for a network path)

Qt treats a string as rich text when the first `<` on its first line starts a known HTML tag (it does not have to be
at the start: `GIMP <img src="...">` qualifies; prompt syntax such as `<lora:x:0.8>` does not). Rich text in a
`QLabel`, a tooltip or `QTextEdit.append` loads `<img src>` from a file path. On Windows a path of the form
a UNC path (two backslashes, a server name, a share, a file) makes the system connect to that host over SMB and offer the user's log-in (user name and a
password hash that can be attacked offline). The local-file load was confirmed by running (the injected image changed
the label's and the log's size); the network form was not run.

Sinks that take untrusted text:

* `tabs/metadata.py:420` `item.setToolTip(f.detail)`: `detail` is the EXIF Software, Make / Model, owner or host
  computer string of the selected image (`core/metadata.py:150`, `:162`, `:166`). Scenario: a downloaded picture
  whose Software tag is an `<img>` element with a UNC path on the sender's server as its source; the user selects it in the Metadata tab (the place to
  inspect untrusted files) and rests the mouse on the finding.
* `tabs/caption.py:1199-1200` `self.log_box.append(msg)` (a `QTextEdit`), fed by `:748` (the server's answer to
  "List"), `:1175` (a failed image's reason: `inference/openai_api.py:111` and `:114` at HEAD, `:124-133` in the
  working copy, pass the server's own error message through), `:637` and `:1184`. Scenario: a hostile, compromised or (over plain http) impersonated API server
  answers with an error text that carries the tag.
* `tabs/metadata.py:409` (`lbl_file`, always rich, with the reader's error text), `tabs/train_dialogs.py:318-320`
  (the verdict string from a run folder's `problem_images.json`), `tabs/caption.py:640` (a download error in a
  message box).

Not affected: the image cards' tooltip (its first line is the file name, which cannot hold `<` on Windows), the
review dialog (escapes), the Train console and every `QPlainTextEdit`, item texts in lists and tables (drawn as plain
text), the Help Center (its HTML is the app's own).

Smallest fix: make the Auto Caption log a `QPlainTextEdit` (`appendPlainText`, and give it `setMaximumBlockCount`);
pass every tooltip that carries outside text through `html.escape` (or `Qt.convertFromPlainText`); set
`Qt.PlainText` on labels and message boxes that show outside text, and escape the parts inserted into deliberately
rich labels. A helper `tabs.common.plain_tip(text)` and a test that feeds `<img src=...>` through each sink keep it
fixed.

### H5. Files people share carry absolute paths (the Windows user name and folder layout) (READ, fields listed)

Checked first, and clean: **LoRA and fine-tune checkpoint metadata hold no path.** They hold base names only
(`ss_training_adapter`, `ss_context_lora`, `tagscriber_source_base`), the optimizer label with its arguments, the
LoRA name as title, the trigger phrase, the last preview prompt as `modelspec.description`, a preview picture as
`modelspec.thumbnail` (or the picture named in Thumbnail; "off" disables it), author / licence / tags only when
typed, and the date (`training/train.py:1259-1293`, `:780-783`; `training/metadata.py`). Preview PNGs carry no text
chunks. The Settings "Copy report" holds versions and GPU names only (`core/hardware.py:393-416`).

Files that do carry paths:

| file | fields |
|---|---|
| a saved training preset (`user_data/training_presets/<family>/<name>.json`, `training/presets.py:65-67`, `:359-365`) | `LORA_OUTPUT_DIR`, `FAMILY_EDIT_DIR`, `FAMILY_EDIT_EXTRAS_DIR`, `FAMILY_FT_REG_DIR`, `CONTEXT_LORA_PATH`, `METADATA_THUMBNAIL` (absolute paths); `LORA_NAME`, `METADATA_AUTHOR`, `_TITLE`, `_DESCRIPTION`, `_TAGS`, `_TRIGGER_PHRASE` |
| `run.log` in the run folder (`tabs/train.py:995`, `:1009`) | the run folder, the base model path (`training/train.py:1064`), state folders (`:277`, `:1219`), every saved file's full path (`:787`, `:1306`), the sample folder (`:582`), the dataset folder and image paths in dataset warnings (`training/dataset.py:75`, `:142`, `:404`), image file names in preflight and caption-fix lines, the text of automatic recaptions |
| `train_config.json` (`training/pipeline.py:946-949`) | `output_dir`, `dit_path`, `vae_path`, `te_path`, `training_adapter`, `speed_lora`, `context_lora_path`, `preview_checkpoint`, `sample_reference`, the paths of `dataset.json` and `sample_prompts.txt`, the captioner's model folder, and under `settings` every form value including the path keys above |
| `dataset.json` (`training/pipeline.py:723-736`) | `image_directory`, `cache_directory`, `control_directory`, `extra_reference_directory` |
| `.tagscriber_paused.json` (`tabs/train.py:1110`) | the state or checkpoint path, the dataset folder, every form value |
| `loss_log/*.json`, `per_image_loss.jsonl`, `sample/.prompts_<name>.json` | image file names (stems), caption texts, preview prompts |
| `user_data/logs/tagscriber.log` | dataset folders, image paths, model folders, and see M1 |
| `user_data/projects/<hash>.json`, `settings.json`, `.last_train_settings.json`, `training_queue.json` | dataset folders, model paths, recent folders (not normally shared) |

Scenario: a preset sent to a friend, or a `run.log` pasted into a GitHub issue, shows the home folder path with the
Windows user name in it and the names of private dataset folders.

Smallest fix, without changing what the files on disk hold (the run folder must keep working as it is):
* a "Copy for sharing" / "Export for sharing" action for the console and for presets that writes a scrubbed copy:
  the home folder, the app folder, the output folder and the dataset folder replaced by placeholders, the path keys
  and `LORA_NAME` / `METADATA_AUTHOR` left out of an exported preset;
* one function `core.privacy.scrub(text)` used by both, with a test on Windows and POSIX paths;
* a short "What is in the files TagScribeR writes" topic in the Help Center with the table above.
Question for the owner: should the trainer's own `run.log` lines print base names instead of full paths? It would make
a pasted log safe by default, at the cost of a log that no longer says which drive a file was on.

---

## MEDIUM

### M1. The HTTP library's own log lines reach `tagscriber.log`: response headers, 13 lines per request (RUN)

`core/log.py:42-43` quietens `httpx`, `httpcore` and `openai`, but the installed `openai` package (3.x) talks through
`httpx2` / `httpcore2`, which are not in the list. One request wrote 13 lines to the log file, among them the full
list of response headers. With a cloud API those name the account (`openai-organization`, `openai-project`), carry
cookies and request ids. The key itself is not in these lines. A batch of a thousand images also pushes everything
useful out of the 2 MB x 4 rotation, and the request URL goes to the console at INFO.

Fix: let the file handler record DEBUG only for the app's own loggers (`tagscriber`, `core`, `tabs`, `inference`,
`training`) and WARNING for everything else, instead of keeping a list of noisy names that goes stale with each new
dependency. Test: one request against a loopback server leaves no `httpcore2` line and no response header in the log.

### M2. Windows that are created per click and never destroyed; one keeps a timer running (RUN)

* `tabs/train.py:1350-1352` creates a `ProblemImagesDialog` on every click, parented to the tab, without
  `WA_DeleteOnClose`; its 4-second timer (`tabs/train_dialogs.py:263-265`) is never stopped. After six open / close
  cycles six dialogs were alive with six timers running; each re-reads the report and rebuilds its table whenever
  the trainer rewrites `problem_images.json`. Growth: one dialog per click, for the life of the app.
* `main.py:181-183`: a `CommandPalette` per Ctrl+K, never destroyed (10 opens, 10 alive), each holding the command
  list and its closures.
* Context menus built per right-click and never deleted: `tabs/workspace/browser.py:398` (its closures hold the list
  of selected paths), `tabs/workspace/panels.py:544`, `tabs/common.py:205`. Dialogs shown with `exec()` and left as
  children of their tab: `tabs/caption.py:821`, `tabs/common.py:354` and `:371` (every confirmation box),
  `tabs/datasets.py:245`, `tabs/gallery.py:255`, `tabs/train.py:1357` (READ).

Each object is small; the sum is bounded only by how long the app stays open, and the timers add work every epoch.
Fix: `setAttribute(Qt.WA_DeleteOnClose)` on the non-modal ones and stop the timer in `done()` (as
`SliderPromptsDialog` already does); `deleteLater()` after `exec()` for menus and modal dialogs.
Test: open and close N times, `findChildren` returns none.

### M3. A paused run's sidecar is trusted as it is (READ)

`tabs/train.py:1228-1255` (`resume_paused`) takes `values`, `folder`, `state` and `family` from
`.tagscriber_paused.json` in the run folder and launches with them. Unlike a preset, a queue item or the last-run
snapshot, they do not pass `presets.apply`. So a run folder received from someone else can set: any dataset folder;
any state path; `LORA_OUTPUT_DIR` (the run then writes elsewhere and the "name in use" question is skipped);
and `OPTIMIZER_TYPE`, which as a dotted name reaches `importlib.import_module(...)` and a call of the named attribute
(`training/optimizers.py:275-277`). Audit 08 judged that path safe because the form and presets only offer strict
choices; the sidecar is the way around them. No callable was found that turns this into code execution (the call
always passes `lr=`, and the first argument is the trainer's own parameter list), so this is rated MEDIUM as a
missing gate, not as an exploit.

Fix: validate the sidecar's values with the preset rules (strict choices refused, types coerced), require `state` to
be inside the run folder the sidecar was found in, and show the dataset folder in the Resume confirmation when it is
not the open one. Question for the owner: should a dotted optimizer name be limited to a list of module prefixes
(`torch.optim.`, `bitsandbytes.optim.`, the optimizer packages in `requirements.txt`)? Fizgig accepts any; a list
would be narrower than Fizgig, so it is the owner's call. Recommendation: yes, with a setting to add prefixes.

### M4. A picture of up to 600 megapixels is decoded whole for captioning, editing and export (READ + header test)

`Image.MAX_IMAGE_PIXELS = 300_000_000` (`inference/image_prep.py:11` / `:12`, `core/image_ops.py:25`) makes Pillow
warn above 300 Mpx and refuse only above 600: a generated 400 Mpx PNG header opened with a warning, 625 Mpx was
refused. Thumbnails and the health scan have their own 80 Mpx guard (audit 09 finding 6, still in place), but
`load_rgb` (`inference/image_prep.py:14-30` / `:35-57`) and `open_upright` (`core/image_ops.py:96-114`) decode the
full picture: about 1.8 GB for 600 Mpx RGB before the copies of the alpha flattening. The caption worker decodes a
whole chunk (`inference/worker.py:160-167` at HEAD, up to 16 with API concurrency; the working copy decodes on up to
four threads, `inference/pipeline.py:33-37`, and already reduces JPEGs with `Image.draft`, so there the finding is
about PNG, WebP and TIFF). A solid-colour PNG of that size is about a megabyte,
so a downloaded dataset can hold several. Result: memory exhaustion while captioning (reported per image if Python
gets a `MemoryError`, a stalled machine otherwise). Not measured here.

Fix (question for the owner, since pictures between 300 and 600 Mpx load today): treat the warning as the limit the
constant states (`warnings.simplefilter("error", Image.DecompressionBombWarning)` around these two opens), and in
`load_rgb` read the size first, use `Image.draft` for JPEG (the working copy does) and decode one picture at a time
when it is above, say, 100 Mpx.

### M5. Copies of pictures and captions stay in `user_data` for ever, with no way to clear them from the app (READ)

* Thumbnails: `tabs/workspace/thumbs.py:34-36`, `:61-63` writes a 384 / 768 px JPEG per image, per change of the
  file, never removed and never capped.
* Training caches: `training/pipeline.py:53-63`, one folder of latents and text embeddings per dataset folder ever
  trained (latents decode back to the pictures).
* Caption backups: `core/caption_io.py:130-146`, one copy per caption per day edited, under the dataset folder's
  name.
* `user_data/projects/<hash>.json` keeps each dataset folder's absolute path (`core/projects.py:79`).

A user who deletes a private dataset still has its thumbnails, its latents and its captions here. Growth is
unbounded in all three. Fix: a "Storage and privacy" group in Settings showing the size of each and a Clear button
(Recycle Bin for backups), an age or size cap for thumbnails applied in the background at start, and a paragraph in
the Help Center.

### M6. The legacy `api_presets.json` keeps its plain-text keys after they were imported (READ)

`core/secrets.py:103-116` imports the old file's keys into the credential store and leaves the file untouched
(`core/paths.py:39`: "read once for migration and otherwise left untouched"). It is git-ignored and was never
committed (history checked), but it sits in the app folder for good: a zip of the folder, a cloud backup or a shared
drive carries the keys. Fix: after a successful import rewrite the legacy file without the `api_key` values (keep a
note in it saying where they went), and say so once in the log. Anyone who upgraded from the old version should
check that file now.

### M7. The commit history shows a personal e-mail address (RUN: `git log`, address not reproduced here)

The author and committer of the owner's commits on `main` and on the work branch is a personal mailbox address, not
a GitHub `noreply` address. It is public with every push. `tools/check_release.py` prints the identities "for you to
judge" but does not flag it. Fix going forward: `git config user.email <id>+<login>@users.noreply.github.com` and
GitHub's "Block command line pushes that expose my email"; make `check_release.py` report any identity that is not a
`noreply` address as a finding. Removing it from the existing history needs a rewrite and a force-push, which is the
owner's decision alone.

---

## LOW

* **L1. `tools/check_release.py` gaps (RUN for the first).** A base ref that does not exist (`--base` mistyped,
  `origin/main` not fetched) gives "Commits not yet on ...: 0", scans no history and exits 0 (`:52-54`, `:111-119`:
  git's return code is ignored). Commit message bodies are not scanned (rule 6 of `CLAUDE.md` names them). Binary
  files are skipped silently and there is no check for model, archive or database extensions or for file size. The
  machine name is only caught when it looks like `DESKTOP-...`; `platform.node()` could be added to the terms as the
  user name is. Patterns miss URLs with embedded credentials and UNC paths.
* **L2. `.gitignore` gaps.** `Thumbs.db` (it holds thumbnails of pictures in a folder), `desktop.ini`, `.DS_Store`,
  `*.log`, `.env.*`, `*.pem`, `*.key`, and weight files outside the tracked family assets (`*.ckpt`, `*.pt`,
  `*.gguf`, `*.onnx`) are not ignored.
* **L3. The JPEG sent to an API server carries the source file's JPEG comment (RUN).**
  `inference/image_prep.py:33-36` (`:60-63` in the working copy): Pillow writes `info["comment"]` by default. EXIF, GPS and XMP are not sent.
  `training/metadata.py:127-131` (the LoRA thumbnail) does the same. Fix: `img.info.pop("comment", None)` before
  saving, or save from a fresh image.
* **L4. Header readers without a size limit (RUN).** `training/pipeline.py:161-171` (`ft_continuation`) reads
  `f.read(n)` for any claimed length: a header claiming 2^62 bytes raised `MemoryError`, which the `except` does not
  list, on the UI thread at Resume; `ft_source_unfit` beside it has the 256 MB guard, `base_check._header` has both
  guards. `training/modules/safetensors_io.py:104-107` has none (trainer child, and `ft.source_unfit_reason`), and
  `get_tensor` (`:109-126`) trusts `data_offsets` (a hostile span makes numpy allocate the claimed size). The result
  is an error, never code execution. Fix: one shared `read_header(f, limit)` that also checks `8 + n <= file size`,
  and a span check in `get_tensor`.
* **L5. The final LoRA file is written in place.** `training/lora.py:637-639` calls `save_file(path)` directly;
  `training/train.py:1651-1655` writes `<name>.safetensors` and copies the numbered file from it afterwards. A full
  disk or a kill during that write leaves a truncated final file and no numbered copy of the last epoch. Fix: write
  to `path + ".tmp"` and `os.replace`, as the fine-tune writer and the thumbnail refresh already do.
* **L6. The loss graph re-reads the whole step log on the UI thread (RUN).** `tabs/train.py:1369-1393` and
  `training/step_log.py:82-106`: 0.05 s at 20 000 steps, 0.45 s at 200 000, every time the file changes (every 20
  steps) while the Train tab is visible. Fix: remember the offset and read only the new lines, or read in a worker.
* **L7. Unbounded text.** The Auto Caption log (`tabs/caption.py:230`) has no block limit (the Train console has
  4000). Caption files are read whole with no size limit (`core/caption_io.py:55-56`, a 60 MB "caption" was read in
  full); cap at about 1 MB and flag the image.
* **L8. `core/metadata.py:274-285` `_atomic_write_bytes`** has neither the `fsync` nor the permission-bit fix that
  audit 09 finding 8 brought to `caption_io` and `image_ops`: after a metadata edit on Linux the picture is mode
  0600, and a power cut can leave an empty file under the real name.
* **L9. The plain-text fallback.** Without a credential store, keys go to `user_data/api_profiles.json`
  (`core/secrets.py:124-128`) with the default permission bits (world-readable on a shared Linux machine). Fix:
  mode 0600 for that file when it holds a key.
* **L10. A preset from someone else can set path keys with only a console line** (`training/presets.py:284-289`,
  `tabs/train.py:654-674`): `LORA_OUTPUT_DIR`, the edit folders, `CONTEXT_LORA_PATH`, and `METADATA_THUMBNAIL`
  (a picture on the user's disk would be embedded, downscaled, in the LoRA they later upload). Fix: a confirmation
  dialog listing the path keys when the preset came from Import, or drop them at import (question for the owner:
  Fizgig preset files carry these keys).
* **L11. The Hard images window trusts `problem_images.json` (RUN).** A key that is an absolute path makes
  `_image_path` (`tabs/train_dialogs.py:267-273`) resolve outside the open dataset, and "Save fix" writes that
  picture's caption (with the daily backup). Needs a run folder from someone else and a click.
* **L12. An API server can hold a request past the timeout.** The timeout is per read: a loopback server that sent a
  byte every half second kept a `timeout=3` request alive for 20 s (RUN), and the response size is not limited.
  The request runs on a worker thread and Abort works, so the app does not hang. Redirects are followed, but the
  `Authorization` header is dropped on a redirect to another origin (RUN: the second server saw none). Proxies come
  from the environment (`HTTP_PROXY` and friends), as with any `httpx` client.
* **L13. If the app dies while a run is going, the child keeps running without a reader (SUSPECTED).**
  `tabs/train.py:1021-1036` starts the stages with `QProcess` and no job object; `shutdown()` kills the tree only on a
  normal quit. After a crash of the app the trainer either runs on unseen or fails at its next console write, at a
  point that can be inside a save (see L5). Fix: a Windows job object with kill-on-close (or `prctl` on Linux), or
  let the trainer write its own log file and ignore a closed console.

* **L14. New, uncommitted code: `inference/server_probe.py` forwards the key on a redirect (RUN).**
  `_get_json` (`:59-69`) adds `Authorization: Bearer <key>` and uses `urllib`, which follows redirects and keeps
  that header for the new host: a loopback server that answered `/props` with a 302 to another origin made the
  probe send the key there. It is called for local URLs only (`inference/openai_api.py:197-202`), where keys are
  rare, and the `openai` client itself does not do this (L12). Fix before it is committed: an opener whose redirect
  handler refuses redirects (three status GETs need none), or one that drops `Authorization` when the origin
  changes. Test: the loopback pair of this audit. The reply size is capped (2 MB) and errors are swallowed without
  logging the URL or the key, which is right.

Smaller notes, no number: a file symlink used as a caption is read through (its target's text becomes the caption,
is copied by export and collections, and goes to the model as tag hints); writes replace the link itself and never
follow it. Interrupted atomic writes can leave `.edit-*`, `.meta-*` or `.<name>.*.tmp` files beside the pictures (the
Gallery hides dot files, the trainer's preflight counts an `.edit-*.png` as an image without a caption).
System tools are started by bare name (`taskkill`, `explorer`, `typeperf`, `nvidia-smi`, `powershell`): Windows looks
in the current folder before the system folders, the launchers set the current folder to the app folder, and Qt's
file dialogs do not change it, so this needs someone who can already write into the app folder.

---

## Earlier fixes checked again

| finding | state |
|---|---|
| 08-3 caption extension equal to an image extension | holds: `write_caption` refuses (`core/caption_io.py:153-155`), preflight checks (`training/pipeline.py:535-540`); `tests/test_caption_io.py` passes |
| 08-6 LoRA name as a path | holds: `run_dir_for` validates before any use (`training/pipeline.py:66-71`) |
| 08-8 `torch.load` on a resume state | holds: `weights_only=True` (`training/train.py:281-288`); the test passes |
| 09-1 EXIF format written by the metadata tools | holds: the stripped file reads back in Pillow, GPS gone |
| 09-2 overwrite + convert asks first | holds (`tabs/editor.py:361-378`) |
| 09-3 / 09-9 collection copies and names | holds (`core/fileops.py:23-26`, `:86-105`; `tabs/workspace/context.py:299-302`) |
| 09-6 pixel guard for thumbnails and the health scan | holds (`core/image_utils.py:62-72`, `core/health.py:53-58`); captioning and editing are not covered, see M4 |
| 09-8 atomic writers: mode and fsync | holds in `caption_io` and `image_ops`; **not applied** in `core/metadata.py`, see L8 |
| 10-1 custom model code is per model | holds (`inference/transformers_vlm.py:119-137`) |
| 10-2 bitsandbytes wheel checked by SHA-256 | holds (`tools/install.py:224-279`); `tests/test_install_bnb_hash.py` passes |
| 10-4 API calls can be cancelled | holds (`inference/openai_api.py:163-184`); the timeout itself is per read, see L12 |
| 10-7 key cleared when the server address changes | holds in the box, **undone by Save**, see H1 |

---

## Fix list (in this order)

1. **H1** `core/secrets.py` (`save`, `load`), `tabs/caption.py` (key host). Test in `tests/test_secrets.py`:
   an emptied key is gone from the store; a key is not loaded for another host.
2. **H2** `inference/openai_api.py`, `tabs/caption.py`, `core/log.py` (redacting filter), `core/secrets.py` (strip
   on save). Test: a key with whitespace is refused with a plain message; a log capture never holds a key.
3. **M1** `core/log.py`. Test: one loopback request leaves no third-party DEBUG line in the log file.
4. **H4** `tabs/caption.py` (log widget), `tabs/metadata.py`, `tabs/train_dialogs.py`, `tabs/common.py` (helper).
   Test: `<img src=...>` through each sink stays text (the widget's size does not change, the document has no image).
5. **H3** `core/metadata.py` (`_rewrite_jpeg`, `_jpeg_segments`, `_rewrite_png`, `_filter_exif`, `privacy_findings`),
   `tabs/metadata.py` (wording), `tabs/help_content.py`. Test: the generated JPEG and PNG of this audit, per mode.
6. **M6** `core/secrets.py` (`_import_legacy`). Test: after import the legacy file holds no key.
7. **H5** new `core/privacy.py` (`scrub`), `tabs/train.py` (copy the console for sharing, export a preset for
   sharing), `tabs/help_content.py`. Test: `scrub` on Windows and POSIX paths; an exported preset has no path key.
8. **M3** `tabs/train.py` (`resume_paused`, `_restore_paused`), optionally `training/optimizers.py`. Test: a sidecar
   with a dotted optimizer, a state outside the run folder or another dataset folder is refused or asked about.
9. **M2** `tabs/train.py`, `tabs/train_dialogs.py`, `main.py`, `tabs/workspace/browser.py`,
   `tabs/workspace/panels.py`, `tabs/common.py` and the `exec()` call sites. Test: N open / close cycles leave no
   child of that class and no active timer.
10. **M4** `inference/image_prep.py`, `core/image_ops.py`, `inference/worker.py` (after the owner's answer). Test:
    a generated header above the limit is refused with a per-image message; a large JPEG is draft-reduced.
11. **M5** `tabs/settings.py`, `tabs/workspace/thumbs.py`, `tabs/help_content.py`. Test: the cap removes the oldest
    thumbnails; Clear empties the folder.
12. **M7, L1, L2** `tools/check_release.py`, `.gitignore`. Test: a missing base ref exits non-zero; a commit message
    with a home path is found; a tracked `.safetensors` outside the family assets is found.
13. **L4, L5** `training/modules/safetensors_io.py`, `training/pipeline.py`, `training/lora.py`. Tests: a hostile
    header length is an error within the limit; a save that fails half-way leaves the old file.
14. **L14** `inference/server_probe.py`, before that file is committed. Test: the redirecting loopback pair.
15. **L3, L6, L7, L8, L9, L10, L11, L13** as described, each with the test named in its line or a plain unit test of
    the new behaviour.

## What was checked and found sound

* **Nothing listens.** No server socket, no local web server, no IPC endpoint anywhere in the app or the trainer.
* **Network use is what the user asks for.** The API URL the user typed, the Hugging Face Hub for a download the
  user started (and the fixed tokenizer / tagger repos on first use), three browser links
  (`core/support.py`, the "Get" buttons). No update check, no fonts or icons from the network, no start-up call.
  Hub telemetry and onnxruntime telemetry are switched off before either library loads (`core/hardware.py:84-87`).
  TLS verification is not disabled anywhere.
* **Keys.** Stored through `keyring`; the profile file holds none when a store works. No key in `ProviderSpec`
  logging, window titles, status lines, the diagnostics report, run folders, `train_config.json` (the recaption
  captioner is a local model or the tagger only: `tabs/train.py:253-256`, `:764-769`), presets, the child
  processes' arguments, or URL query strings. The child environment is the app's own environment plus
  `PYTHONPATH` and the GPU pin (`training/pipeline.py:1115-1126`); the app puts no secret into it.
* **Process execution.** Every `subprocess` / `QProcess` call uses an argument list; the one `shell=True`
  (`training/compile.py:274`) runs a `vcvars64.bat` path found through `vswhere`. Child stages run with the app
  folder as working directory and this interpreter. `openUrl` is only called with `QUrl.fromLocalFile` on image
  paths, folders and run folders; `webbrowser.open` only with `https://huggingface.co/...` built from family
  constants.
* **No unsafe loaders.** No `pickle`, `eval`, `exec`, `yaml.load`; `torch.load` only with `weights_only=True`;
  models and LoRAs through safetensors; `trust_remote_code` only per model on an explicit tick.
* **The fine-tune writer.** `save_checkpoint` writes to `.tmp` and renames, removes the `.tmp` on any failure and
  refuses to write over its own source (`training/ft.py:729-774`). The scratch master is
  `<run folder>/.tagscriber_ft_master` and nothing else (`training/train.py:688`, `:1112`), so the `rmtree` at
  `training/ft.py:208-211` and `:284` can only ever remove that folder; it is deleted only after the final
  checkpoint is written (`training/train.py:1646-1647`), and each trained weight is replaced atomically
  (`training/ft.py:264-275`). A full disk during a checkpoint raises and leaves the previous checkpoints.
* **`base_check.py`** caps the header at 256 MB and the fingerprint tensor at 64 MB and checks both against the file
  size; it never reads a model-sized tensor in the UI process.
* **Run-folder names.** `validate_output_name` refuses separators, `..`, a trailing dot and illegal characters before
  anything opens or writes in the run folder; state pruning is anchored to the LoRA name and keeps at least one.
* **Caption and image writes.** Atomic, fsynced, with the daily backup for captions; deletes go to the Recycle Bin;
  collection copies and exports never overwrite; the editor asks before overwriting originals; a caption write
  replaces a symlink or hard link instead of writing through it.
* **Export with "strip metadata".** The exported JPEG held none of the test markers (EXIF, GPS, MakerNote, XMP, IPTC,
  comment, trailer). An in-place edit keeps EXIF, XMP and comments, as its radio button says, and drops the embedded
  EXIF thumbnail (so a crop never leaves the uncropped preview behind).
* **Installer and launchers.** `https` indexes only, pinned torch builds, the bitsandbytes wheel verified by
  SHA-256 before install, quoted paths, `git pull --ff-only`, nothing run from a temp path.
* **Repository.** 264 tracked files: source, docs, the generated screenshots, the logo, one small model asset. No
  settings, logs, caches, datasets or `.env` tracked now or ever added in history (`git log --all --diff-filter=A`);
  `tools/check_release.py` reports no findings on the current branch.
* **Bounded by design.** The pixmap cache (400 MB), the Train console (4000 blocks), the samples list (60), the
  rotating log (2 MB x 4), the loss graph's points (4000 after thinning).

## Not checked

* Real training, real model files, real captioning models, GPU memory behaviour, and anything only a long run shows
  (memory growth of the trainer over many hours was read for, not measured).
* The SMB step of H4 (it would be a network request to another machine); the crash path of L13.
* The numerics of the model families and of `training/modules/rotation.py` (planner and schedule were read for file
  and process safety only); Fizgig parity.
* NVIDIA and Linux paths, macOS, network drives, the Recycle Bin on removable media.
* The installer end to end (no download was made) and known vulnerabilities of the pinned third-party packages.
* `tabs/theme.py` line by line (it was being edited; a scan found file writes only under `user_data/cache/theme`
  and no outside input).
* Every widget's lifetime: the ones in M2 were sampled from the per-click creation sites, not an exhaustive census.

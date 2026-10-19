# Audit 09 - adversarial review of the app core (`core/`, `main.py`, `tabs/` except the Train tab)

Read-only review for data-loss, correctness and "hostile dataset folder" problems. Scope: `core/`, `main.py`, every
`tabs/` module except `tabs/train.py` / `tabs/train_dialogs.py`, and `inference/`, `tools/install.py`, network and
secrets only where this code calls into them (those areas have their own parallel reviews).

Line numbers refer to commit `98a48a2`. While this review ran, somebody else had uncommitted work in the tree
(a "Camera data" feature in `core/metadata.py`, `tabs/metadata.py`, `core/camera_presets.py`, a wheel guard in
`tabs/common.py`). None of the findings below depend on that work, but finding 1 applies to it as well (its
`apply_camera` writes EXIF through the same `_filter_exif` -> `ex.tobytes()` path).

Every finding marked REPRODUCED was shown with a throwaway script on generated images (Pillow 12.3.0, PySide6
offscreen, Linux); nothing was run on real datasets or real models. The commands are inline so they can be rerun.
Findings marked REASONED were read in the code but not executed.

Summary, most severe first:

| # | Severity | Where | What |
|---|----------|-------|------|
| 1 | high | `core/metadata.py` EXIF writers | Every EXIF block the metadata tools write carries a doubled `Exif\0\0` header (JPEG) or a header the container forbids (PNG eXIf). Pillow still reads it; ImageMagick reads nothing. Orientation and authorship are invisible to other software after "Clean metadata", "Apply authorship" or a field edit. |
| 2 | medium-high | `tabs/editor.py` run() | "Overwrite originals" + "Convert" to the file's own format re-encodes the originals in place (JPG at the quality slider, e.g. 50) with no confirmation, then shows a dialog saying converted files were created "next to the originals". Thumbnails are not refreshed. |
| 3 | medium | `core/fileops.py` copy_image_with_caption | A copy into a collection overwrites an existing caption when only the image extension differs (`a.png` copied next to an existing `a.jpg` + `a.txt`): the collection's `a.txt` is replaced, reported as "1 copied". |
| 4 | medium | `core/image_ops.py` / `tabs/editor.py` | In-place edits silently drop PNG text chunks (A1111 `parameters`, ComfyUI `prompt`/`workflow`), XMP and JPEG comments, and re-encode JPEGs at a fixed quality 92 (the slider only applies to Convert). The tooltip says "Color profile and EXIF data are kept", which is true but not the whole story: the data the Prompts tab relies on is gone. |
| 5 | medium-low | `core/image_utils.py`, `core/image_ops.py`, `core/health.py` | 16-bit PNGs (`I;16`) are clipped, not scaled, when converted to 8-bit: thumbnails and previews are black/white, Convert to JPG/WEBP produces a blown-out image, blur detection runs on clipped data. |
| 6 | medium-low | thumbnails / previews / health | A 1.1 MB PNG whose header says 20000x20000 (400 Mpx, under the raised 600 Mpx bomb limit) is decoded in full by every thumbnail worker that paints it (about 2.5 GB each). No dimension check before decoding; `Image.MAX_IMAGE_PIXELS` is raised process-wide to 300 M. |
| 7 | low | `core/caption_io.py` read_text_file | UTF-16 caption files (Notepad "Unicode") decode as cp1252 mojibake with NULs; an edit then saves the mojibake as UTF-8. |
| 8 | low | `core/caption_io.py`, `core/image_ops.py`, `core/metadata.py` | Atomic writers create the replacement with `mkstemp` (mode 0600): captions in a shared folder lose group/other permissions on first save. Image and metadata writers do not `fsync` before `os.replace`, so a power cut during a batch can leave zero-length images. |
| 9 | low | assorted | Inspector keeps showing a trashed image; "Copy to Collection" accepts any text as a folder name; `.json` sidecars go to the bin with the image; export and collection counts ignore `caption.extension`. |

Checked and found no problem (section 10): settings/preset/project loaders (JSON only, no `eval`/`pickle`/`yaml`),
HTML in captions and file names (Qt's auto-rich-text heuristic does not fire on `<lora:...>`-style text; QLabel/tooltip
links are not clickable), `QDesktopServices`/`subprocess` use (local files and fixed argument lists only), symlinks,
every caption write path (all go through `caption_io.write_caption`), the undo/merge logic, selection/proxy index
handling, and Qt threading (all cross-thread results travel over queued signals).

---

## 1. EXIF written by `core/metadata.py` is unreadable to other software (high, REPRODUCED)

**Where.** `core/metadata.py:327` (`_filter_exif` returns `ex.tobytes()`), `:355-357` (`_jpeg_app1` prepends
`EXIF_HEADER` again), `:446-447` (PNG `eXIf` chunk body = `new_exif`), `:479` (WebP `EXIF` chunk body = `new_exif`),
`:546-557` (`set_exif_text`: same `ex.tobytes()` into `_assemble_jpeg` / `eXIf`).

**Why.** `PIL.Image.Exif.tobytes()` already returns `b"Exif\x00\x00" + TIFF`. The JPEG path wraps it in
`EXIF_HEADER + exif`, so the APP1 payload becomes `Exif\0\0Exif\0\0MM\0*...`. The PNG `eXIf` chunk and the WebP `EXIF`
chunk must hold the bare TIFF structure (Pillow's own writers strip the 6-byte header before writing those chunks);
here they get the header. Pillow 12's `Exif.load` strips the header in a `while` loop, so TagScribeR and the test suite
read everything back fine, which is why the existing tests pass. A spec-conformant reader finds `Exif\0\0` where the
TIFF byte-order mark must be and gives up.

**Scenario.** Owner runs Privacy -> Clean metadata (defaults) over a folder of phone photos, all EXIF-rotated. The
module promises "The EXIF Orientation tag is always preserved so photos keep displaying upright". After cleaning, any
viewer, trainer or browser other than Pillow sees no EXIF at all: the photos display rotated, and authorship written
with the Authorship tab is invisible to the people it is meant for. Same for a single field edit in Inspect.

**Reproduction.**

```python
from PIL import Image; from core import metadata as md; import shutil
im = Image.new("RGB", (32, 32)); ex = im.getexif(); ex[md.ORIENTATION] = 6; ex[md.ARTIST] = "Someone"
im.save("orig.jpg", exif=ex); shutil.copy("orig.jpg", "stripped.jpg"); shutil.copy("orig.jpg", "authored.jpg")
md.strip_metadata("stripped.jpg", md.StripOptions()); md.apply_authorship("authored.jpg", md.Authorship(artist="Me"))
seg = next(s for m, s in md._jpeg_segments(open("stripped.jpg", "rb").read()) if m == 0xE1)
print(seg[4:16])   # b'Exif\x00\x00Exif\x00\x00'
```

```
$ identify -format '%[EXIF:Orientation] %[EXIF:Artist]' orig.jpg      -> 6 Someone
$ identify -format '%[EXIF:Orientation] %[EXIF:Artist]' stripped.jpg  -> (nothing)
$ identify -format '%[EXIF:Orientation] %[EXIF:Artist]' authored.jpg  -> (nothing)
```

For PNG, the `eXIf` chunk written by `apply_authorship` starts with `Exif\x00\x00MM\x00*`; the one Pillow writes
for the same image starts with `MM\x00*`. (ImageMagick does not surface PNG eXIf even for Pillow's file, so the PNG
and WebP cases are by inspection against the container specs, not by a second reader.)

**Fix (S).** Normalise once: `raw = ex.tobytes(); return raw[6:] if raw.startswith(EXIF_HEADER) else raw` at the end
of `_filter_exif`, and the same in `set_exif_text`. `_jpeg_app1` then correctly adds the one header JPEG needs, and
the PNG/WebP chunks get the bare TIFF. Add a test that asserts the APP1 payload has exactly one header and the eXIf
body starts with `MM\0*` or `II*\0`, and, where ImageMagick/exiftool is available, that it reads the orientation back.
Existing files already cleaned with the current code can be repaired by running the fixed writer over them once (the
TIFF data is intact behind the extra header).

## 2. Overwrite + Convert to the same format re-encodes originals without asking (medium-high, REPRODUCED)

**Where.** `tabs/editor.py:357-363` (confirmation skipped when `convert`), `:372-375` (`dest = srcp` unless the
extension differs), `:390-391` (`_last_overwrite` false for convert, so `_finished` never calls
`ctx.images_modified`), `:420-424` (dialog "converted file(s) were saved next to the originals").

**Scenario.** "Overwrite originals" selected, Convert format = JPG, quality slider at 50, images are `.jpg`. The code
assumes Convert never touches an original (`not convert` gates the confirmation; the hint text says "originals are
never deleted"). But `FORMATS["JPG"][1] == ".jpg" == srcp.suffix.lower()`, so `dest = srcp` and `image_ops.save`
replaces the original with a q50 re-encode. No confirmation, the only dialog afterwards claims new files were
created, the status line and summary say "done", and because `_last_overwrite` is false the cached thumbnails and
`Entry.mtime` are not refreshed, so the grid keeps showing the old pixels. Same for `.png` -> PNG (harmless
re-encode) and `.webp` -> WEBP (lossy).

**Reproduction** (offscreen, `tabs.editor.confirm` patched to record the dialogs it asks):

```
D. confirmations asked: ['Converted files created']
D. original photo.jpg overwritten in place: True | size 18692 -> 6916 bytes | files now: ['photo.jpg', 'photo.txt']
```

**Fix (S).** Treat a same-extension convert like any other overwrite: ask the destructive confirmation whenever
`overwrite` is on and `dest == srcp` (compute the destination before deciding), or always write converted files as
new unique files in the overwrite mode too, as the hint promises. Either way call `ctx.images_modified` for every
`dest == src` result regardless of the operation kind.

## 3. Collection copies can overwrite an existing caption (medium, REPRODUCED)

**Where.** `core/fileops.py:81-92`. The collision check is `(dest_dir / src.name).exists()`: only the image name.
`unique_stem_for_pair` (which checks the sidecars too) runs only when the image name collides. The sidecar copy on
line 92 then writes `dest_dir / f"{stem}{ext}"` unconditionally.

**Scenario.** Collection holds `a.jpg` + `a.txt` ("ORIGINAL caption in collection"). User adds `a.png` + `a.txt`
("NEW caption") from another dataset via Datasets -> Add Selected or Gallery -> Copy to Collection. `a.png` does not
exist in the collection, so the stem stays `a` and `a.txt` is overwritten. The collection now has two images sharing
one caption, the first image's caption is gone, and the report says "1 copied" with no rename note. Mixed `.jpg`/`.png`
datasets are common. Caption backups do not apply (the copy does not go through `write_caption`).

**Reproduction.**

```python
fileops.copy_image_with_caption(src / "a.png", dst, report=rep)
# dst/a.txt == 'NEW caption from source'; rep.summary() == '1 copied'
```

**Fix (S).** Decide the stem with `unique_stem_for_pair(dest_dir, src.stem, src.suffix)` whenever the image *or* any
sidecar with that stem exists (not only when the image exists), and record the rename in the report. The same
one-sided check exists in `tabs/editor.py:380-383` (caption copied only if `dcap` is absent, so a stale caption from an
earlier edit of a same-stem image is kept for the new file); it does not lose data but mismatches captions.

## 4. In-place edits drop PNG text, XMP and comments; JPEG quality fixed at 92 (medium, REPRODUCED)

**Where.** `core/image_ops.py:67` and `:82` keep only `icc_profile`, `exif`, `dpi`; `save()` (`:141-181`) passes only
those to Pillow. `tabs/editor.py:366` uses `quality = op.params.get("quality", 92)`, so the slider affects Convert only.

**Scenario.** Folder of A1111/ComfyUI renders. User rotates or crops them with "Overwrite originals" (the UI says
colour profile and EXIF are kept). Afterwards Metadata -> Prompts -> "Use embedded prompts as captions" finds
nothing: the `parameters` / `prompt` / `workflow` text chunks are gone, as are XMP and JPEG COM segments. For JPEGs
every in-place rotate is also a full re-encode at q92 / 4:4:4, so repeated edits accumulate loss; the quality slider
the user can see does not apply.

**Reproduction.**

```python
image_ops.save(image_ops.apply(image_ops.open_upright("gen.png"), Operation("rotate", {"direction": "cw"})), "gen.png")
md.read_metadata("gen.png").text.get("parameters")   # None (was 'masterpiece, 1girl\nSteps: 20')
```

**Fix (M).** For PNG, carry `img.text` into a `PngInfo` on save (keep `info` keys that are `str`); for JPEG keep
`info["comment"]` and `info["xmp"]`; for WebP keep `xmp`. Pass the slider's quality for every lossy save in
overwrite mode (or use Pillow's `quality="keep"` for JPEG -> JPEG). State in the tooltip what is and is not kept. This
is also the right place to decide whether in-place edits should strip anything on purpose (they currently do so by
accident).

## 5. 16-bit PNGs are clipped to black/white (medium-low, REPRODUCED)

**Where.** `core/image_utils.py:42-43` and `:28-29` (`convert("RGBA")` from `I;16`), `core/image_ops.py:152-158`
(`convert("RGB")` before JPEG/BMP save), `core/health.py:51` (`convert("L")`).

**Why.** Pillow's `I;16` -> 8-bit conversion clips at 255 instead of scaling, so anything brighter than 255/65535
becomes white. A 16-bit gradient came back as only `{0, 255}` in the thumbnail and in the JPG conversion; the blur
metric is computed on the clipped image.

**Scenario.** Scans or renders saved as 16-bit PNG (Krita, darktable, some upscalers). Thumbnails and previews are
unusable, Convert to JPG/WEBP produces a white image (and with "Overwrite originals" the original is replaced by it,
see finding 2), health flags them as blurry or sharp at random.

**Fix (S).** One helper used by all three places: if `im.mode in ("I;16", "I;16B", "I;16L", "I")`, scale with numpy
(`(arr >> 8).astype(uint8)` for I;16) or `im.point(lambda v: v // 256, "L")` before converting. Export already
re-encodes through `image_ops.save`, so it is fixed by the same change.

## 6. Header-only check missing before thumbnail/preview/health decodes (medium-low, REPRODUCED)

**Where.** `core/image_ops.py:18` (`Image.MAX_IMAGE_PIXELS = 300_000_000`, process-wide, imported by the Editor tab
at startup); `tabs/workspace/thumbs.py:60` -> `core/image_utils.py:48-55` (`load_qimage`: `draft` is a no-op for PNG,
`thumbnail` decodes the whole image); `tabs/workspace/panels.py:185` (inspector preview), `core/health.py:47-54`.

**Scenario.** A dataset downloaded from somewhere contains a 1.1 MB PNG whose IHDR says 20000x20000 (400 Mpx of
zeros compresses to that). It is under the 600 Mpx `DecompressionBombError` threshold that the raised limit implies, so
every thumbnail worker (`cpu_count/2` threads) that paints such a file decodes it fully: RGB + RGBA copies, roughly
2.5 GB per worker. With a handful of such files on screen the app allocates tens of GB; on Windows the per-job
`MemoryError` is caught and the card reads "Unreadable image", but only after the allocation storm; on Linux with
overcommit the OOM killer may take the app and its unsaved edits. The header scan (`read_image_info`) already knows the
dimensions before any decode and nothing uses that.

**Reproduction.** Hand-built PNG (IHDR 20000x20000, one zlib stream of zero rows, 1,166,428 bytes).
`read_image_info` -> `20000x20000`; `load_qimage(path, (384, 384))` under a 3 GB address-space cap raised
`MemoryError` after 2.4 s, i.e. it attempted the full decode.

**Fix (S).** In `load_qimage` / `load_thumbnail` / `health.analyze` check `im.size` (available right after
`Image.open`, no decode) against a sane ceiling (e.g. 80 Mpx for thumbnails, configurable) and return a placeholder
with a log line instead of decoding; use `im.reduce()` / `draft` where the format supports it. Set
`MAX_IMAGE_PIXELS` only around the editor/export operations that legitimately need big images rather than globally.

## 7. UTF-16 captions read as mojibake (low, REPRODUCED)

**Where.** `core/caption_io.py:36-43`. cp1252 decodes almost any byte sequence, so it accepts UTF-16 files.

**Scenario.** Captions written by Windows Notepad with "Unicode" encoding, or exported by a tool that writes UTF-16.
`read_caption` returns `'ÿþ1\x00g\x00i\x00r\x00l\x00, ...'`; the Gallery shows that, "missing:caption" is false, and
if the user edits and saves, the mojibake is written back as UTF-8 (the original is backed up once that day, so it is
recoverable, but the dataset now has a wrong caption).

**Fix (S).** Check for the UTF-16 BOMs (`\xff\xfe`, `\xfe\xff`) before the cp1252 fallback and decode with `utf-16`;
optionally treat a high proportion of NUL bytes as UTF-16LE without BOM.

## 8. Atomic writers: file mode and missing fsync (low, REPRODUCED for the mode, REASONED for fsync)

**Where.** `core/caption_io.py:64-78` (`atomic_write_text`), `core/metadata.py:270-281` (`_atomic_write_bytes`),
`core/image_ops.py:169-180` (`save`).

- `tempfile.mkstemp` creates the temp file with mode 0600; after `os.replace` the caption/image has that mode. A caption
  that was `0664` became `0600` after one save. Matters on Linux/macOS for datasets shared between users or served
  over SMB/NFS; Windows inherits the folder ACL so is unaffected.
- `atomic_write_text` does `flush` + `fsync` before `replace`; `_atomic_write_bytes` and `image_ops.save` do not. With
  delayed allocation a crash or power cut after the rename but before the data hits disk leaves a zero-length image in
  place of the original. "Clean metadata" and in-place edits over a large folder are exactly the long batches where a
  power cut happens.

**Fix (S).** After writing: `os.fsync`; copy the original's mode (`shutil.copymode(path, tmp)` when the target
exists) before `os.replace`. Same three functions.

## 9. Smaller items (low)

- **Inspector keeps a trashed image** (REPRODUCED). `WorkspaceContext.remove_entries` (`tabs/workspace/context.py:273-276`)
  resets the model; a model reset clears the selection without emitting `selectionChanged`, so
  `GalleryTab._on_selection_changed` never runs and `InspectorPanel.entries` still holds the deleted `Entry`
  (`inspector still shows: ['photo.jpg']` after the trash, session empty). Typing into it pushes commands for a key
  the session no longer has (no-ops, but they land on the undo stack and the preview stays). Fix: emit
  `selection_changed([])` from `DatasetBrowser` on model reset, or have `remove_entries` emit `entries_changed` and
  let the inspector drop entries that are gone.
- **Collection name unvalidated** (REASONED). `WorkspaceContext.copy_to_collection` (`context.py:292-303`) uses the
  editable `QInputDialog.getItem` text as `root / name` with no character check; `../x` or an absolute path copies
  outside the collections root. `tabs/datasets.py:162-166` has the `INVALID_CHARS` check; share it.
- **`.json` sidecars are trashed and copied with the image** (REASONED). `CAPTION_SIDECAR_EXTS` includes `.json`
  (`core/fileops.py:20`), so Shift+Del on `cover.png` also bins `cover.json` (a ComfyUI workflow, a tagger's output,
  anything). Recycle Bin, so recoverable; the confirmation says "and their captions". Mention it or drop `.json`.
- **`caption.extension` is honoured only by the session** (REASONED). `core/export.py:68` reads captions with the
  default `.txt`, `tabs/datasets.py:40` counts `.txt`, while `DatasetSession` uses `cfg["caption.extension"]`. The
  setting has no UI, so this only bites people who edit `settings.json`; either wire it through or drop it.
- **Tag bubbles flatten prose and newlines** (REASONED). `core/widgets.py:124-157` splits on `,` only and rebuilds the
  caption with `join_tags`, so removing a bubble from a multi-line or prose caption rewrites the whole text onto one
  comma-separated line. Visible and undoable, so not data loss, but the comment at `panels.py:213` describes a guard
  (`looks_like_tags`) that is not implemented.
- **Editor skips reload of `_modified` thumbnails for same-extension converts**: part of finding 2.

## 10. Checked, no finding

- **Caption writes.** Every writer goes through `caption_io.write_caption`: `DatasetSession.save`, the AI
  `BatchWorker._save` (`inference/worker.py:221-225`), and `export_one` (`atomic_write_text` to a stem that
  `unique_stem_for_pair` has just verified free). Backups are once per day, earliest version; unchanged text is not
  rewritten; empty text never creates a file. Dirty tracking (`text != saved_text`) survives undo/redo, external
  reloads skip dirty entries, and the Caption tab saves dirty entries before a merge-mode job. Typing merges into one
  undo command per image (`ChangeCommand.mergeWith` keeps the first `old`).
- **Deletes.** Only `fileops.trash` (send2trash) and `datasets.delete_collection` -> `fileops.trash`. No `unlink`
  of user files anywhere in scope; the only `rename` is a collection rename guarded by an exists check and an
  "is it open" check.
- **Loaders.** `settings.json`, project files, preset files, API profiles: `json.loads` with type coercion; corrupt
  files are quarantined, not deleted. No `eval`, `pickle`, `yaml.load`, `exec`. The theme name from settings goes to
  `qt_material.apply_stylesheet`, which only parses an XML colour file.
- **HTML / links in untrusted text.** Card tooltips, inspector labels and the metadata findings list carry caption
  text and file names through `QLabel`/`QToolTip` auto-text. `Qt.mightBeRichText` returned False for
  `<lora:foo:0.8>, 1girl`, `<segment:yolo>`, `1girl, <3` and `a < b, c > d`, so real prompt syntax renders as plain
  text. A caption that *starts* with a real tag (`<b>`, `<img src=file://...>`) would render as rich text in a
  tooltip, but no label has `setOpenExternalLinks` and no `QTextBrowser` shows caption text (`ReviewDialog` escapes
  with `html.escape`), so the worst case is a mangled tooltip. The Help Center's `setOpenExternalLinks(True)` only
  shows text from `help_content.py`.
- **URLs and subprocesses.** `QDesktopServices.openUrl(QUrl.fromLocalFile(...))` on image paths (opened by the OS by
  extension; `is_image` only admits image extensions), on folders, and on the run folder. `subprocess.Popen(["explorer",
  "/select,", path])` with a list, no shell. `core/vram_monitor.py` runs fixed `typeperf` / `nvidia-smi` / `amd-smi`
  argument lists.
- **Hostile names and symlinks.** `scan_images` follows file symlinks but `os.walk` does not follow directory
  symlinks; caption paths are derived with `with_name`, so they stay beside the (link) file. Reserved Windows names
  cannot exist on Windows; very long paths fail the `mkstemp` with an `OSError` that is reported, not swallowed.
  `ExportOptions.target_dir` strips `<>:"/\|?*` from the concept.
- **Model/view indices.** All actions resolve keys through `KEY_ROLE` on proxy indexes at the time of the action;
  `select_matching`/`select_keys` iterate the proxy; `navigate` wraps on the proxy row count. Dynamic re-sorting after
  an edit moves rows but selection follows persistent indexes.
- **Threading.** `run_in_background`, `FileJobHandle`, `_Emitter`, `_InfoSignals` all deliver results through signals
  owned by GUI-thread objects (queued connections); workers touch only Pillow/numpy and plain Python data. Health
  results are written into shared dicts from the worker but the GUI reads them only after `finished`.
- **Thumbnail cache.** Keyed by normalised path, mtime, size and tier; `images_modified` refreshes mtime after in-place
  edits (except the case in finding 2). Disk cache is JPEG q88 in `user_data`, which stores thumbnails of whatever
  dataset was opened (expected; worth a line in the privacy docs).

## 11. Not checked

Real training, real models, Windows-specific behaviour (ACLs, `explorer /select`, send2trash on network drives), the
AI providers themselves, and the uncommitted camera-data feature beyond noting that it shares finding 1.

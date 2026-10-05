"""Help Center content. Keep this describing the *current* app.

Each topic: id -> (title, group, html). Hotkeys are listed in HOTKEYS and
rendered as a table.
"""
from __future__ import annotations

HOTKEYS: list[tuple[str, str, str]] = [
    # scope, keys, action
    ("Global", "Ctrl+1 … Ctrl+7", "Switch tab (Gallery, Auto Caption, Editor, Datasets, Metadata, Train, Settings)"),
    ("Global", "F1", "Help for the current tab"),
    ("Global", "Ctrl+K  (or Ctrl+Shift+P)", "Command palette — type to find and run any action, filter or help topic"),
    ("Image grid", "Click / Ctrl+Click / Shift+Click", "Select / add to selection / select a range"),
    ("Image grid", "Arrow keys", "Move between images"),
    ("Image grid", "Ctrl+A", "Select all shown images"),
    ("Image grid", "Ctrl+O", "Open a folder (opens in every tab)"),
    ("Image grid", "Ctrl+F", "Jump to the filter box"),
    ("Image grid", "Ctrl+S", "Save changed captions"),
    ("Image grid", "Ctrl+Z / Ctrl+Y", "Undo / redo caption changes"),
    ("Image grid", "Ctrl+Shift+C / Ctrl+Shift+V", "Copy a caption / paste it onto all selected images"),
    ("Image grid", "Del", "Clear captions of selected images (undoable)"),
    ("Image grid", "Shift+Del", "Move selected images (and captions) to the Recycle Bin"),
    ("Image grid", "Ctrl + mouse wheel, Ctrl+= / Ctrl+-", "Bigger / smaller thumbnails"),
    ("Image grid", "Ctrl+0", "Reset thumbnail size"),
    ("Gallery", "Double-click / F2", "Edit the selected image's caption"),
    ("Gallery", "Ctrl+Up / Ctrl+Down", "Previous / next image (works while typing)"),
    ("Auto Caption", "Ctrl+Enter", "Caption the selected images"),
    ("Auto Caption", "Esc", "Abort (images already finished stay saved)"),
    ("Image Editor", "Ctrl+R / Ctrl+Shift+R", "Rotate selected right / left"),
    ("Datasets", "Ctrl+N", "New collection"),
    ("Train", "Ctrl+Enter", "Start training (queues it while a run is going)"),
    ("Datasets", "F5", "Refresh collections"),
]

FILTER_SYNTAX = """
<table cellpadding="4">
<tr><td><code>smile</code></td><td>caption or file name contains “smile”</td></tr>
<tr><td><code>tag:"long hair"</code></td><td>has exactly that tag (underscores and spaces are treated alike)</td></tr>
<tr><td><code>-tag:blurry</code></td><td>does <b>not</b> have the tag (any term can be negated with <code>-</code>)</td></tr>
<tr><td><code>tag:*hair</code></td><td>wildcards: any tag ending in “hair”</td></tr>
<tr><td><code>missing:caption</code> / <code>has:caption</code></td><td>no caption / has a caption</td></tr>
<tr><td><code>is:unsaved</code></td><td>edited but not saved yet</td></tr>
<tr><td><code>is:tagged</code> / <code>is:prose</code></td><td>tag-list captions / sentence captions</td></tr>
<tr><td><code>res:&lt;768</code></td><td>shorter side under 768 px (also <code>w:</code> <code>h:</code> <code>mp:</code> for megapixels)</td></tr>
<tr><td><code>ar:&gt;1.5</code></td><td>aspect ratio (width ÷ height) — wide images</td></tr>
<tr><td><code>tags:&gt;40</code> / <code>len:&lt;20</code></td><td>number of tags / caption length in characters</td></tr>
<tr><td><code>name:img_*</code> / <code>ext:png</code></td><td>file name pattern / file type</td></tr>
<tr><td><code>flag:blurry</code></td><td>Health scan results: duplicate, similar, blurry, lowres, crop, or any</td></tr>
</table>
<p>Combine terms with spaces — all must match. Example: <code>tag:1girl -tag:blurry missing:caption res:&lt;1024</code></p>
"""

TOPICS: dict[str, tuple[str, str, str]] = {
    "start": ("Getting started", "Basics", """
<h2>Getting started</h2>
<ol>
<li><b>Open a folder</b> of images with <i>📂 Open Folder</i> (Ctrl+O) in the Gallery. The folder opens in
every tab, so you can switch between tagging, AI captioning and editing without reopening it.</li>
<li><b>Look around.</b> Each card shows the image, its file name and caption. Cards marked
<span style="color:#d63031">NO CAPTION</span> still need one; a yellow dot means unsaved changes.</li>
<li><b>Caption with AI</b> in the <i>Auto Caption</i> tab (natural-language descriptions with a vision model), or
<b>auto-tag</b> in <i>Gallery → Batch</i> (booru-style tags with a WD tagger).</li>
<li><b>Refine</b> captions in <i>Gallery → Inspect</i> (one image) or <i>Gallery → Batch</i> (many images at once).</li>
<li><b>Save</b> with Ctrl+S. Nothing is written to disk before you save, and every overwritten caption is backed up first.</li>
<li><b>Collect</b> your finished images into a training folder with <i>📦 Copy to Collection</i>.</li>
<li><b>Train</b> a LoRA on the open folder in the <i>Train</i> tab (Ctrl+6). See <i>Training a LoRA</i>.</li>
</ol>
<p>Captions are stored the way trainers expect: <code>image.png</code> + <code>image.txt</code> in the same folder.</p>
<p><b>Tip:</b> press <b>Ctrl+K</b> anywhere for the command palette — type a few letters (e.g. “miss cap”, “health”,
“save”) to run any action, apply a filter, open a recent folder or jump to a help topic.</p>
"""),
    "safety": ("Your data is safe", "Basics", """
<h2>How TagScribeR protects your data</h2>
<ul>
<li><b>Nothing is written until you save.</b> Edits in the Gallery are kept in memory, can be undone (Ctrl+Z), and are
marked with a yellow dot until saved.</li>
<li><b>Automatic caption backups.</b> The first time each day a caption file is overwritten, its previous version is
copied to <code>user_data\\caption_backups\\&lt;date&gt;\\</code>. (Settings → Open caption backups.)</li>
<li><b>Safe writes.</b> Captions are written to a temporary file and swapped in, so a crash can't leave a half-written file.
Empty captions never create empty .txt files.</li>
<li><b>Recycle Bin, not delete.</b> Deleting images, captions or collections moves them to the Recycle Bin.</li>
<li><b>No silent overwrites.</b> Copying into a collection never replaces an existing file — clashes get “ (1)” names.
The Image Editor saves copies by default and asks before changing originals.</li>
<li><b>Unsaved-changes prompts</b> appear before opening another folder or closing the app.</li>
<li><b>AI jobs save as they go.</b> If you abort or something crashes, finished images keep their captions.</li>
</ul>
"""),
    "gallery": ("Gallery overview", "Gallery", """
<h2>Gallery</h2>
<p>The Gallery is your dataset workspace: browse, filter, inspect and edit captions for thousands of images.</p>
<h3>Selecting</h3>
<p>Click to select, Ctrl+Click to add, Shift+Click for a range, Ctrl+A for everything shown. Right-click for
image actions (open, show in folder, copy/paste caption, copy to collection, Recycle Bin, smart selections).</p>
<h3>Thumbnail size</h3>
<p>Use the <b>Size</b> slider, <b>Ctrl + mouse wheel</b> over the grid, or <b>Ctrl+= / Ctrl+-</b> (Ctrl+0 resets).
Untick <b>Captions</b> to show images only. For larger text everywhere, see <i>Accessibility</i>.</p>
<h3>Sorting</h3>
<p>Sort by name, caption length, tag count, size, aspect ratio, date, or put unsaved / uncaptioned images first.
<b>↓</b> reverses the order.</p>
<h3>Side panels</h3>
<ul><li><b>Inspect</b> — preview and edit the selected image (see <i>Inspecting &amp; editing</i>).</li>
<li><b>Batch</b> — change many captions at once (see <i>Batch tools</i>).</li>
<li><b>Tags</b> — tag statistics (see <i>Tag statistics</i>).</li></ul>
<p>Captions written by the Auto Caption tab or by other programs appear automatically when you come back to the Gallery
(images you're editing are left alone).</p>
"""),
    "filter": ("Filtering images", "Gallery", "<h2>Filtering</h2><p>Type in the filter box (Ctrl+F) to show only matching "
               "images. The box turns red if part of the filter isn't understood — hover it for details.</p>"
               + FILTER_SYNTAX +
               "<p>Tip: in <i>Batch</i>, choose <b>All shown</b> to apply an operation to everything the filter shows.</p>"
               "<h3>Saved filters (★)</h3><p>The ★ button next to the filter box offers ready-made filters (missing "
               "captions, unsaved edits, health flags, small images…) and lets you <b>save the current filter</b> under a "
               "name for this folder. Saved filters belong to the dataset and come back whenever you open it.</p>"),
    "inspect": ("Inspecting & editing", "Gallery", """
<h2>Inspecting &amp; editing captions</h2>
<p>Select one image to see a large preview, its size, and its caption.</p>
<ul>
<li><b>Caption box</b> — edit freely; changes apply as you type and form one undo step per burst of typing.
The counter shows tags, words and characters.</li>
<li><b>Tag bubbles</b> — click × to remove a tag, or type in “Add tag” and press Enter.</li>
<li><b>◀ ▶</b> or <b>Ctrl+Up / Ctrl+Down</b> — previous / next image, even while typing.</li>
</ul>
<p><b>Several images selected?</b> The bubbles show only the tags <i>every</i> selected image shares. Removing a bubble
removes that tag from all of them; adding a tag adds it to all of them.</p>
"""),
    "batch": ("Batch tools", "Gallery", """
<h2>Batch tools</h2>
<p>Choose <b>Apply to: Selected</b> or <b>All shown</b> (everything the current filter shows), then:</p>
<ul>
<li><b>Add</b> tags at the start or end (skips tags already present).</li>
<li><b>Remove</b> tags (whole tags only — removing “hair” keeps “long hair”).</li>
<li><b>Replace</b> one tag with another, or with nothing to delete it.</li>
<li><b>Find &amp; Replace</b> any text, optionally with regular expressions.</li>
<li><b>Prefix / Suffix</b> — ideal for trigger words (e.g. <code>ohwx woman, </code>); works on sentence captions too.</li>
<li><b>Cleanup &amp; normalize</b> — underscores → spaces, lowercase, remove duplicates, sort, escape parentheses,
accents → ASCII, or clear captions.</li>
<li><b>Auto Tag</b> — run a WD tagger (see <i>Auto tagging</i>).</li>
<li><b>Quick tags</b> — click to add a favourite tag (or a group like <code>ohwx, 1girl</code>) to the selection.
Type a new one and press Enter to add it; right-click or Delete removes; drag to reorder. Saved instantly and shared with
Settings.</li>
</ul>
<p>Every batch operation is a single undo step and stays unsaved until you press Save.</p>
"""),
    "tags": ("Tag statistics", "Gallery", """
<h2>Tag statistics</h2>
<p>The <b>Tags</b> panel counts how many images use each tag — over the whole folder, the shown images, or the
selection. Spelling variants (<code>long_hair</code> / <code>Long Hair</code>) are counted together.</p>
<ul>
<li><b>Double-click</b> a tag to show only images that have it.</li>
<li><b>Right-click</b> to show images without it, select them, add it to the selection, <b>rename / merge</b> it
everywhere (type an existing tag's name to merge), or <b>delete</b> it everywhere.</li>
<li><b>Show only tags used once</b> finds typos and noise tags quickly.</li>
</ul>
<p>Why it matters: consistent tags train better. Merge spelling variants and remove one-off noise before training.</p>
"""),
    "health": ("Dataset health", "Gallery", """
<h2>Dataset health</h2>
<p><i>Gallery → Health</i> checks the open folder for problems that hurt training:</p>
<ul>
<li><b>Exact duplicates</b> — identical files. The first of each group is kept; the extra copies are flagged.</li>
<li><b>Near-duplicates</b> — the same picture resized, re-saved or slightly edited (perceptual hashing). The slider
sets how similar images must be (strict → loose).</li>
<li><b>Blurry / soft</b> — images much less sharp than the rest of the dataset.</li>
<li><b>Low resolution</b> — shorter side under half the training size.</li>
<li><b>Heavy crop</b> — images whose shape would lose more than 20% when fitted to the nearest training bucket.</li>
<li><b>Aspect-ratio buckets</b> — how your images spread over SDXL-style buckets (64 px steps) at the chosen training
size. Many tiny buckets with 1–2 images train less efficiently; consider cropping outliers to common shapes in the
Image Editor (<i>Crop to aspect ratio</i>).</li>
</ul>
<p>Click a result (or a group under it) to select those images. Compare them in <i>Inspect</i>, then move unwanted copies
to the Recycle Bin with <b>Shift+Del</b>. Cards show purple badges, and you can filter with <code>flag:duplicate</code>,
<code>flag:similar</code>, <code>flag:blurry</code>, <code>flag:lowres</code>, <code>flag:crop</code> or <code>flag:any</code>.</p>
<p>Scans run in the background and are cached — changing the training size or similarity updates the results instantly,
and rescanning only analyzes new or changed images.</p>
"""),
    "caption": ("Auto Caption", "AI", """
<h2>Auto Caption</h2>
<ol>
<li>Pick a <b>model</b> (Local model tab) or a <b>server</b> (API / Server tab).</li>
<li>Select images in the grid — <i>Select Uncaptioned</i> picks every image without a caption.</li>
<li>Choose <b>Instructions</b>: a preset (training caption, booru tags, character, clothing, composition…) or write your own
(see <i>Caption presets</i>).</li>
<li>Choose what happens to <b>existing captions</b>: overwrite, skip, append or prepend.</li>
<li>Press <b>Caption</b> (Ctrl+Enter). Cards show QUEUED / AI WORKING / FAILED. Esc aborts.</li>
</ol>
<p>Results are saved to disk as each image finishes (with backups), and appear in the Gallery immediately. If some images
fail, a report lists them with the reason; everything else is kept.</p>
<h3>Review before applying</h3>
<p>Tick <b>Review before applying</b> (Generation section) to collect AI captions in a review queue instead of writing
them. Cards show <span style="color:#00897b">REVIEW</span>; when the run finishes the review window opens (or click
<b>📝 Review proposals</b>). For each image you see the current caption, the highlighted changes and the caption you'll get
(editable). <b>Accept</b> (Ctrl+Enter), <b>Reject</b> (Ctrl+Backspace), <b>Skip</b> (Ctrl+Right), or accept/reject all.
Accepted captions become unsaved, undoable edits — press Save to write them. Unreviewed proposals are discarded if you
open another folder.</p>
<h3>Subject &amp; tag hints (recipes)</h3>
<ul>
<li><b>Subject</b> — type your trigger word or character name (e.g. <code>ohwx woman</code>). The model then refers to
the subject that way instead of “a woman”. <b>Start every caption with it</b> is guaranteed: if the model forgets,
TagScribeR adds it.</li>
<li><b>Use each image's current tags as hints</b> — the <i>tag → caption</i> recipe. First run <i>Gallery → Batch →
Auto Tag</i>; then caption here. The tags ground the model on details (eye color, clothing, objects) and it writes a
natural caption. Choose <i>Append</i> under “If captioned” to keep the tags and add the caption after them.</li>
</ul>
<p><b>Per-folder memory:</b> the subject, “start with it”, tag hints and the chosen preset are remembered for each
dataset folder separately — opening another folder loads its own settings (an empty subject if none was set), so one
dataset's trigger word never ends up in another's captions. These settings live in <code>user_data\\projects\\</code>;
nothing is written into your dataset folders.</p>
<h3>Caption length</h3>
<p>The Gallery's caption editor shows an estimated <b>CLIP token</b> count. SD1.5 and SDXL read captions in 75-token
chunks, so very long captions matter less there; Flux, Qwen-Image and other models with T5/LLM text encoders handle
long natural captions well. Use a shorter preset (e.g. Concise Caption) for SDXL-family training if needed.</p>
<h3>Speed tips</h3>
<ul>
<li>The model stays loaded between runs — only the first run pays the loading time.</li>
<li>The very first image after loading is slower (~20 s on AMD) while the GPU picks its kernels.</li>
<li><b>Batch size</b> 2–4 (Model options) is usually faster than 1. If memory runs out, TagScribeR retries one at a time.</li>
<li><b>Max image size</b> (Advanced sampling) trades detail for speed and memory.</li>
<li><b>🧹 Free VRAM</b> unloads the model, e.g. before using ComfyUI. Settings can unload it automatically when idle.</li>
</ul>
"""),
    "presets": ("Caption presets", "AI", """
<h2>Caption presets</h2>
<p>A preset stores caption instructions so you can reuse them. Built-in presets come first in the list;
<b>★</b> marks your own.</p>
<ul>
<li><b>Use one</b> — pick it from the list. Your own presets can also restore their saved generation settings.</li>
<li><b>Edit freely</b> — change the text; the hint shows <b>● Edited</b>. Your working text is remembered even if you don't save it.</li>
<li><b>💾 Save</b> — store the current instructions under a name. Tick <i>Also save generation settings</i> to include tokens,
temperature, sampling and what to do with existing captions. Built-in presets can't be overwritten — save your version
under a new name.</li>
<li><b>⋯ menu</b> — rename or delete your preset, revert edited text, and <b>import / export</b> presets as a .json file to
share them or move them to another PC.</li>
<li><b>Output is a tag list</b> — tick for tag-style presets so Append/Prepend merges tags without duplicates.</li>
</ul>
<p>Your presets live in <code>user_data\\caption_presets.json</code> and stay until you delete them.</p>
"""),
    "models": ("Choosing a model", "AI", """
<h2>Choosing a captioning model</h2>
<p>The list shows <b>installed</b> models first (from <code>models\\</code>, your Stability Matrix <code>Models\\LLM</code>
folder, and folders added in Settings), then models you can <b>download</b> (size shown before downloading).
“✓ fits your GPU” compares the model size with your VRAM.</p>
<table cellpadding="4" border="0">
<tr><td><b>Qwen3-VL 4B / 8B</b></td><td>Detailed and fast; the most AMD-friendly. Recommended default.</td></tr>
<tr><td><b>Qwen3.5 2B / 4B / 9B</b></td><td>Newest Qwen generation; strong detail. Currently slower on AMD (no ROCm kernels
for its linear attention yet — output is correct).</td></tr>
<tr><td><b>Gemma 4 E2B / E4B</b></td><td>Fluent natural-language captions. Use <i>Use model's recommended settings</i>.</td></tr>
<tr><td><b>JoyCaption Beta One</b></td><td>Built for diffusion-training captions, including uncensored and artistic content.</td></tr>
<tr><td><b>27B+ models, GGUF files</b></td><td>Run them in LM Studio or llama.cpp server and use the API / Server tab.</td></tr>
</table>
<h3>Model options</h3>
<ul><li><b>Precision</b> — Auto picks bfloat16 on GPUs that support it (RX 7000+, RTX 30+), else float16.</li>
<li><b>Quantization</b> — 8-bit/4-bit need bitsandbytes (installed by default; run <code>update.bat</code> if it is missing).</li>
<li><b>Allow reasoning mode</b> — lets Qwen3.5+ “think” first. Slower; rarely better for captions.</li>
<li><b>Allow custom model code</b> — only for models from sources you trust.</li></ul>
"""),
    "api": ("API & local servers", "AI", """
<h2>API / Server</h2>
<p>Connect to any OpenAI-compatible server: <b>LM Studio</b>, <b>Ollama</b>, <b>llama.cpp server</b>, KoboldCpp, vLLM, or cloud APIs.</p>
<ol><li>Enter the <b>URL</b> (LM Studio default: <code>http://127.0.0.1:1234/v1</code>).</li>
<li>Click <b>List</b> to test the connection and fetch the model names.</li>
<li>Enter a <b>key</b> if the service needs one, and <b>💾 save a profile</b> — keys are stored in Windows Credential Manager,
not in a file.</li>
<li><b>Parallel</b> sends several images at once — good for cloud APIs, keep 1 for a single local GPU.</li></ol>
<p>Reasoning is switched off automatically for local servers that support it (Qwen3.5-style templates).</p>
"""),
    "autotag": ("Auto tagging (WD)", "AI", """
<h2>Auto tagging</h2>
<p><i>Gallery → Batch → ✨ Auto Tag</i> runs a WD (SmilingWolf) booru tagger. The model downloads once (0.3–1.2 GB).</p>
<ul>
<li><b>Existing tags:</b> skip tagged images, append new tags (no duplicates), or replace.</li>
<li><b>General threshold</b> — lower gives more tags and more mistakes (0.35 is a good start).</li>
<li><b>Character threshold</b> — kept high so the tagger doesn't invent character names.</li>
<li><b>Blacklist</b>, <b>Force prepend</b> (e.g. your trigger word) and <b>Force append</b>.</li>
<li>Formatting: underscores → spaces, escape parentheses, include a rating tag.</li>
<li><b>Presets</b> — pick a built-in (Balanced, Precise, Broad, No character names) or <b>Save…</b> your own settings
(model, thresholds, blacklist, trigger words) under a name. Your presets stay until you delete them.</li>
</ul>
<p>Results stay unsaved and undoable — review them, then Save.</p>
"""),
    "editor": ("Image Editor", "Tools", """
<h2>Image Editor</h2>
<p>Works on the folder open in the workspace. Select images in the grid, then:</p>
<ul>
<li><b>Rotate / flip</b> — buttons apply immediately to the selection (Ctrl+R / Ctrl+Shift+R rotate).</li>
<li><b>Resize</b> — by longest side, shortest side, exact size or percent. Images already smaller than the target are
<b>skipped</b> unless <i>Allow upscaling</i> is ticked, so nothing gets blurry by accident.</li>
<li><b>Crop to aspect ratio</b> — keeps as much of the image as possible at 1:1, 2:3, 3:2, 16:9… (handy for training
buckets). <b>Crop to exact size</b> cuts a fixed rectangle. <i>Keep</i> chooses which part survives (center, top, left…).</li>
<li><b>Convert</b> — JPG / PNG / WEBP / BMP / TIFF with a quality setting for JPG and WEBP.</li>
</ul>
<h3>Preview</h3>
<p>Changing any Resize / Crop setting shows a before → after preview of the selected image with the resulting size.</p>
<h3>Where results go</h3>
<ul><li><b>Save edited copies</b> (default): <code>Image Edits\\&lt;source folder&gt;\\</code>. Originals are untouched, names never
collide, and each copy gets its caption file. <i>Open</i> shows the folder.</li>
<li><b>Overwrite originals</b> asks first. Color profiles and EXIF data are kept; photo rotation is applied to the pixels.
Converting with overwrite selected writes the new file next to the original — originals are never deleted.</li></ul>
<p>Edits run in the background with a progress bar and <b>Cancel</b>. Skipped images are listed with the reason.</p>
"""),
    "datasets": ("Dataset Collections", "Tools", """
<h2>Dataset Collections</h2>
<p>Collections are folders (in <code>Dataset Collections\\</code>, configurable) where you gather finished images and
captions for training. The panel shows how many images each holds and how many still lack captions.</p>
<ul><li><b>New</b> (Ctrl+N), <b>Rename</b>, <b>Delete</b> (Recycle Bin), <b>Show in Explorer</b>.</li>
<li>Select images in the grid and press <b>Add to Collection</b> (Ctrl+Enter). Images and captions are copied; name clashes are
renamed, never overwritten. Unsaved caption edits are saved first so the copy matches what you see.</li>
<li><b>Double-click</b> a collection (or <i>Open in workspace</i>) to open it in every tab — e.g. to review its captions in the
Gallery or caption the rest in Auto Caption.</li></ul>
<h3>Export for training</h3>
<p><b>🚀 Export for Training</b> (also in Ctrl+K) makes a training-ready copy of the selected images — or of everything shown
if nothing is selected. Your originals are never changed and nothing in the output folder is overwritten.</p>
<ul>
<li><b>kohya-style subfolder</b> <code>&lt;repeats&gt;_&lt;concept&gt;</code> (e.g. <code>10_ohwx</code>) for sd-scripts and compatible trainers.</li>
<li><b>Fit training buckets</b> scales each image to cover its nearest bucket at the chosen resolution and center-crops it;
or limit the longest side; or keep sizes. Images that would need more than ~10% enlargement are skipped (and listed)
unless you allow upscaling.</li>
<li>Optional format conversion, <b>metadata stripping</b> (colour profile kept), and sequential renaming.</li>
<li>Captions are copied next to each image (.txt or .caption). A <b>trigger word</b> can be added to the start of every
caption (pre-filled from this folder's subject); images without captions are skipped by default.</li>
</ul>
"""),
    "metadata": ("Metadata", "Tools", """
<h2>Metadata</h2>
<p>Image files can carry hidden information: where a photo was taken, which device took it, the prompt and
workflow that generated it, editing history. The Metadata tab shows it, audits it, removes it, and lets you credit your work.</p>
<h3>Inspect</h3>
<p>Select an image to see everything it stores, with a privacy summary on top
(<span style="color:#ff7675">red</span> = sensitive, <span style="color:#fdcb6e">yellow</span> = worth knowing).
Text fields and artist / copyright / description / software can be edited (double-click), then <b>Save field edits</b>.</p>
<h3>Privacy</h3>
<ul>
<li><b>Audit</b> counts what the selected (or all shown) images contain, without changing anything.</li>
<li><b>Clean metadata</b> removes what you tick: GPS location, device make/model/serials/owner, AI generation data
(prompts, seeds, models, ComfyUI workflows), XMP history, timestamps, software — or <i>Everything</i>.</li>
<li>Cleaning is <b>lossless</b>: image data isn't re-encoded, so quality is identical. Photo orientation is always kept,
and the color profile is kept unless you tick the option to remove it.</li>
<li>Supported: JPEG, PNG and WebP. Caption .txt files are separate and never affected.</li>
</ul>
<h3>Authorship</h3>
<p>Write your artist name, copyright / license, description and the software you used to the standard EXIF and PNG
fields — exactly the values you enter. Save them as a <b>template</b> to reuse. EXIF can only hold plain ASCII, so “©” is
written as “(c)” there; PNG keeps the exact text.</p>
<h3>Prompts → captions</h3>
<p>Images from A1111 / Forge, ComfyUI, NovelAI or InvokeAI often contain the prompt that generated them. Use it as a
starting caption for images without one (or replace existing captions). Results are unsaved and undoable until you Save.</p>
"""),
    "access": ("Accessibility & display", "Settings", """
<h2>Accessibility &amp; display</h2>
<ul>
<li><b>Interface scale</b> (Settings → Appearance) enlarges all text, buttons and inputs — 100% to 200%.
Restart TagScribeR to apply.</li>
<li><b>Thumbnail size</b> — Size slider, Ctrl + mouse wheel, Ctrl+= / Ctrl+- in any image grid. Larger cards load sharper
thumbnails and slightly larger caption text.</li>
<li><b>Themes</b> — light and dark themes in Settings → Appearance.</li>
<li>Every tool can be driven from the keyboard — see <i>Keyboard shortcuts</i>.</li>
</ul>
"""),
    "train": ("Training a LoRA", "Train", """
<h2>Training a LoRA (Train tab)</h2>
<p>TagScribeR trains LoRAs natively, using the same training engine, presets and adaptive learning rate as the
Fizgig trainer. It trains the dataset folder that's open in the workspace, so the captions you just fixed are the
ones it learns.</p>
<ol>
<li><b>Open and caption your dataset</b> (Gallery / Auto Caption). Every image needs a caption file, so filter
<code>missing:caption</code> to find gaps. Put your trigger word in the captions (Auto Caption → Subject).</li>
<li><b>Pick a family</b> (Krea 2, Qwen Image 2.1, MiniMax H3, FLUX.2 Klein, SDXL / Pony / Illustrious / NoobAI, Anima) and set its <b>Model files</b> once. <i>Get</i> opens the download page;
the paths are remembered.</li>
<li><b>Load a preset.</b> ✨ presets are measured recipes; the first one is applied on your first visit.</li>
<li>The tab starts in a <b>simple view</b>: name, epochs, resolution and preview prompts, with the preset deciding
the rest. Tick <b>Show all settings</b> for full control of every option.</li>
<li>Set the <b>LoRA name</b> (and output folder), check the <b>preview prompts</b> (Samples section), then
<b>Start Training</b> (Ctrl+Enter).</li>
<li>Watch the <b>Loss</b> chart and the <b>Samples</b>. Every saved epoch is a usable LoRA: pick the best-looking one,
not necessarily the last.</li>
</ol>
<h3>What happens when you press Start</h3>
<ul>
<li>The settings are checked first (missing captions or model files stop it with a clear message).</li>
<li>They are frozen into a run folder (<code>&lt;output&gt;/&lt;LoRA name&gt;/</code>), so changing the tab during a run
can't affect it.</li>
<li>Training runs as separate processes, so the app stays responsive and a crash can't take it down.</li>
<li>Stages: <b>caching latents</b> → <b>caching captions</b> → <b>training</b>. Caching only encodes what
changed since the last run.</li>
</ul>
<h3>Pause, Stop, Resume, Queue</h3>
<ul>
<li><b>Pause</b> finishes the current epoch, saves a resumable state and exits cleanly; <b>Resume</b> continues exactly
there. <b>Stop</b> ends the run immediately (saved epochs stay).</li>
<li>To train more epochs on a finished LoRA: raise <i>Epochs</i>, click <i>Latest</i> under Resume, Start.</li>
<li>Start while a run is going <b>queues</b> the current settings; queued runs start one after another. The queue is
never started automatically when the app opens.</li>
<li><b>Last run</b> reloads the settings of the most recent run.</li>
</ul>
<p>See also: <i>Training presets</i>, <i>Adaptive learning rate</i>, <i>Problem images</i>, <i>Training previews</i>,
<i>Training memory &amp; speed</i>.</p>
"""),
    "train_presets": ("Training presets", "Train", """
<h2>Training presets</h2>
<ul>
<li><b>✨ Built-in presets</b> come with each family. They are tested recipes (rank, learning rate or Adaptive LR range,
epochs, resolution) and can't be changed or deleted.</li>
<li><b>Save…</b> stores the current settings as your own preset, in
<code>user_data/training_presets/&lt;family&gt;/</code>. Presets never contain the model family, so loading one can't
switch your model. Preview settings and Resume aren't part of presets either.</li>
<li><b>Import…</b> adds a preset file. Fizgig preset files work as they are (same setting names).</li>
<li>When a preset holds a value this family doesn't offer (an optimizer, a learning-rate bound, a precision), that
setting is kept as it was and the console says so. Settings this version doesn't know are ignored.</li>
</ul>
"""),
    "train_lr": ("Adaptive learning rate", "Train", """
<h2>Adaptive learning rate</h2>
<p>Instead of one fixed learning rate, you give a <b>Min</b> and <b>Max</b>. The run starts in the geometric middle
(√(min × max), e.g. 2.83e-4 for 2e-4–4e-4) and adjusts once per epoch:</p>
<ul>
<li><b>Probe up ×1.25</b> after two epochs in a row with a new best loss.</li>
<li><b>Reduce ×0.5</b> when the loss stops improving (after 1 epoch in epochs 2–3, otherwise 2).</li>
<li><b>Reduce + rollback</b> on instability: more than half the steps had their gradients clipped, or the LoRA's weights
grew more than 30% in one epoch. The weights are also blended 70/30 back toward the previous epoch, and the optimizer
is restored.</li>
<li>It never leaves the Min–Max range. While it's on, the Learning rate box and the scheduler are ignored. The Loss
chart marks its decisions (↑ ↓ !).</li>
</ul>
<p>Optimizers that set their own rate (Automagic v3) switch Adaptive LR off. Lion needs about a tenth of an AdamW
rate.</p>
"""),
    "train_watch": ("Problem images (loss watch)", "Train", """
<h2>Problem images</h2>
<p><b>Detect problem images</b> follows each image's loss through the run. The loss is corrected for how noisy each
step was, so images are compared fairly. Open <b>Problem Images</b> during a run to see the verdicts:</p>
<ul>
<li><b>stuck</b>: consistently harder than the rest and not improving. Usually a wrong or misleading caption, or an
image that doesn't fit the set.</li>
<li><b>suspect</b>: unusually high loss. <b>exhausted</b>: learned early, then stopped improving.
<b>easy</b> / <b>mid</b> / <b>learning</b>: fine.</li>
<li>Edit a caption there and click <b>Save fix</b>. The caption file is saved (with a backup), and the run re-encodes
it at the next epoch.</li>
<li><b>Per-image adaptive LR</b> trains stuck images more gently (×0.5 down to ×0.1) and easy ones slightly harder
(×1.1).</li>
<li><b>Auto-recaption stuck images</b> re-captions them between epochs with the captioner you choose (a local vision
model or the WD tagger). After two failed attempts an image is set aside.</li>
<li>When nothing has improved for a while, the window says training has <b>plateaued</b> and estimates the best
epoch.</li>
</ul>
<p>These need <b>batch size 1</b>.</p>
"""),
    "train_samples": ("Training previews", "Train", """
<h2>Training previews</h2>
<ul>
<li>Previews are rendered with the LoRA as it is after each epoch (Samples section: prompts one per line, size, steps,
seed, CFG). <i>Preview before training</i> shows the base model for comparison.</li>
<li>They use the averaged (EMA) weights and leave the family's training adapter off, so you see what your saved file
will do.</li>
<li>On cards under 20 GB the preview size is capped at 768 px and the model steps aside for decoding.</li>
<li>A failed preview (e.g. out of memory) turns previews off for the rest of the run. Training continues.</li>
<li><b>Preview override…</b> renders a different prompt at the next preview round without restarting.</li>
<li>Each saved checkpoint carries its own epoch's preview as its thumbnail (shown by ComfyUI's model browser).</li>
<li>A family's <b>Turbo LoRA</b> (if you set its file) makes previews much faster. Strength 0 turns it off; -1 uses the
family default.</li>
</ul>
"""),
    "train_memory": ("Training memory & speed", "Train", """
<h2>Memory &amp; speed</h2>
<ul>
<li><b>Base precision: Auto</b> chooses between bf16, fp8, INT8 and 4-bit NF4 from your free VRAM and resolution,
in the order each family's recipe prefers, and skips ones your machine can't run (INT8 needs GPU support, 4-bit
needs bitsandbytes). fp8 checkpoints are loaded as they are.</li>
<li><b>Blocks to swap</b> streams parts of the model through system RAM to fit smaller cards (25–65% slower). Auto
uses as few as fit.</li>
<li><b>Target megapixels</b> sets the training resolution. Higher is sharper but slower and needs more memory.</li>
<li><b>Gradient accumulation</b> gives a bigger effective batch without more memory.</li>
<li><b>AdamW 8-bit</b> and the 4-bit base need the <code>bitsandbytes</code> package, which the installer adds by
default. If it is missing (an older install), run <code>update.bat</code>. Without it, runs fall back to AdamW with a
warning.</li>
<li>Close other apps holding VRAM (ComfyUI, Forge) before training, and use <b>Free GPU memory</b> in Auto
Caption.</li>
<li>The <b>VRAM</b> and <b>RAM</b> bars at the bottom of the window (under every tab) show memory in use out of the
total, updated every second, so you can also see what a captioning model takes. The white mark and the <i>peak</i>
figure are the highest point since the last training run started (or since the app opened); they reset when a run
starts. Both bars cover the whole machine, so other apps are included. If VRAM is pinned at the top of the bar,
training is likely spilling into system memory and slowing down: lower Target megapixels, pick a smaller Base
precision or swap more blocks. <b>Hide stats</b> hides the bars (remembered). A card with no readable counter
shows "VRAM stats unavailable".</li>
<li>Photos that are rotated only by EXIF are trained as stored, so the pre-start check warns about them. Edit and save
them in the Image Editor first.</li>
</ul>
<h3>Krea 2</h3>
<ul>
<li><b>Pick a RAW checkpoint</b> as the DiT, not the Turbo file. Either kind works: the bf16 file (about 26 GB) or
an <b>fp8 / fp8-scaled RAW</b> file (about 13 GB). An fp8 file stays fp8 in memory.</li>
<li><b>Base precision: Auto</b> follows Fizgig's order: INT8 if your GPU supports it and it fits, then 4-bit (needs
bitsandbytes), then fp8, and finally fp8 with block swap. INT8 and 4-bit are made from whichever file you picked.
Choose <b>fp8</b> yourself to train on the fp8 weights exactly as they ship.</li>
<li>The <b>Turbo LoRA</b> is optional and only for previews: the RAW model plus the LoRA renders in 8 steps without a
second model. Leave it empty to preview the RAW model at 28 steps.</li>
<li>The text encoder file can be the fp8_scaled or the bf16 Qwen3-VL-4B. Each cached caption is about 30 MB, so
leave room on the cache drive.</li>
<li>Measured by Fizgig at 0.25 MP: INT8 about 16 GB, 4-bit about 11 GB, fp8 about 19 GB (each swapped block saves
about 0.4 GB). bf16 holds 26 GB of weights alone. Each extra image in the batch adds about 2.4 GB.</li>
<li>The <b>fp8-scaled text encoder</b> also stays fp8 in memory, which saves VRAM while captions are cached.</li>
<li><b>Automagic v3</b> (Optimizer) sets its own learning rate for each group of layers: text fusion, attention, MLP
and input/output. The Learning rate box is only its starting value, and Adaptive LR, the scheduler and per-image LR
are ignored with it.</li>
<li><b>Compile blocks</b> (Memory &amp; Precision, under Show all settings) mainly speeds up NVIDIA cards. Auto leaves
it off on AMD, with block swap, on fp8 and bf16 bases, and on short runs. On Windows it needs Triton and the C++ Build
Tools; without them the run trains normally and says so.</li>
</ul>
<h3>MiniMax H3</h3>
<ul>
<li><b>Images only.</b> TagScribeR trains H3 LoRAs from still images. Clips, sound and voice items aren't
supported here.</li>
<li><b>Files:</b> the pruned INT8 DiT (about 21 GB), the Qwen3-VL-32B text encoder, the H3 video VAE, and the optional
training adapter. <i>Get</i> opens each download page.</li>
<li><b>Memory:</b> loading the DiT needs about 22 GB of free system RAM. Caching captions needs about 15 GB of free
VRAM; with less, the text encoder streams from system RAM and is slower.</li>
<li>The default mode trains blocks 20 to 49. <b>Low-noise %</b> and the training mode are under <i>Show all
settings</i>.</li>
<li>The presets use the Automagic v3 optimizer, which sets its own learning rate, so Adaptive LR is off for them.</li>
<li><b>Medium to High Noise LR %</b> (Timesteps, under <i>Show all settings</i>) scales the learning rate of the
steps drawn from the noisy half of the range, where pose, framing and face shape are decided. Leave it at 100 unless
you are experimenting; lower values bias a run toward surface detail. It only acts with the AdamW optimizer:
Automagic v3 sets its own rate and ignores it.</li>
<li>Previews run at 20 steps on the training model. Fizgig's Turbo-LoRA previews and HQQ 4-bit aren't available
yet.</li>
</ul>
<h3>FLUX.2 Klein Base 9B</h3>
<ul>
<li><b>Files:</b> the Base 9B checkpoint (the fp8 file, as Fizgig uses, or the bf16 one), the FLUX.2 <code>ae.safetensors</code>
autoencoder, and the Qwen3-8B text encoder. <i>Get</i> opens each download page; the Klein and autoencoder repos are
gated, so accept their terms on Hugging Face first.</li>
<li><b>Model area</b> chooses which blocks the LoRA trains: <i>Full Model</i>, <i>Identity</i> (a subject's
likeness), <i>Style</i> and <i>Style+Composition</i>, <i>Details</i>, or <i>Custom</i> (list the blocks yourself).
The built-in presets set it for you.</li>
<li><b>Timestep sampling</b> defaults to <code>flux2_shift</code>, which adapts to the image size. The other modes
are for experiments.</li>
<li>Previews render on the training model at 40 steps. The 4-step Distilled preview model isn't supported.</li>
<li><b>Attention mechanism</b> (under Show all settings): <code>sdpa</code> runs on any GPU and is the default.
<code>flash3</code> is listed because Fizgig lists it, but Fizgig has no working flash3 path, so choosing it stops the
first training step with an error. Use sdpa.</li>
<li>On cards under about 16 GB, Auto picks the 4-bit base (about 8.5 GB at 0.5 MP).</li>
</ul>
<h3>SDXL, Pony, Illustrious, NoobAI</h3>
<ul>
<li><b>One file:</b> pick the model's <code>.safetensors</code> checkpoint. The VAE and text-encoder rows are
optional: leave them empty to use the ones inside the checkpoint, or point the VAE row at a fixed VAE.</li>
<li><b>Pick the right family for the checkpoint.</b> NoobAI comes as two kinds: <i>eps</i> and <i>v-pred</i>.
Training a v-pred checkpoint as eps (or the reverse) gives noise.</li>
<li>Captions are cut at <b>225 tokens</b>. Tag captions work well; consider <i>Shuffled tag variants</i> with
<i>Keep first N tags</i> for your trigger word (Caption augmentation, off by default).</li>
<li>Batch size above 1 works for these models. The loss watch needs batch size 1.</li>
<li>Three optional extras, all off by default: <b>Min-SNR gamma</b>, <b>Noise offset</b>, and <b>Also train
convolutions (LoCon)</b>.</li>
<li>The first run downloads the small CLIP tokenizer files. To stay offline, put them in a
<code>clip_tokenizer</code> folder next to the checkpoint.</li>
<li>These presets are community starting points, not recipes measured by Fizgig. Expect to tune them.</li>
</ul>
<h3>Anima</h3>
<ul>
<li><b>Files:</b> the Anima DiT, the Qwen-Image VAE and the Qwen3-0.6B text encoder (all from the Anima repo).</li>
<li>Anima is tag-trained: lowercase tags with spaces, artists as <code>@name</code>. The preview negative prompt is
tag-style for the same reason.</li>
<li>The model's author recommends a light touch: the presets use low learning rates and are community starting
points, not recipes measured by Fizgig.</li>
</ul>
"""),
    "settings": ("Settings", "Settings", """
<h2>Settings</h2>
<ul>
<li><b>Appearance</b> — theme and interface scale.</li>
<li><b>AI defaults</b> — unload an idle model after N minutes; run the tagger on GPU or CPU.</li>
<li><b>Model folders</b> — extra folders to scan for vision models (subfolders included). Stability Matrix is detected automatically.</li>
<li><b>Quick tags</b> — manage the Gallery's quick-tag list.</li>
<li><b>System &amp; diagnostics</b> — detected GPU, PyTorch, ROCm/CUDA and library versions; open logs and caption backups;
copy a report for bug reports.</li>
</ul>
<p>Captioning choices (model, prompt, tokens, temperature, save mode) are remembered automatically.</p>
"""),
    "trouble": ("Troubleshooting", "Help", """
<h2>Troubleshooting</h2>
<ul>
<li><b>Where are the logs?</b> <code>user_data\\logs\\tagscriber.log</code> (Settings → Open log folder).</li>
<li><b>I need an old caption back.</b> Look in <code>user_data\\caption_backups\\&lt;date&gt;\\</code>.</li>
<li><b>“Not enough GPU memory”.</b> Lower Max image size, use a smaller model or quantization, and close apps that keep models
in VRAM (ComfyUI, Forge).</li>
<li><b>First image takes ~20 seconds (AMD).</b> One-time GPU kernel selection after loading; later images take about a second.</li>
<li><b>Qwen3.5 logs “falling back to its reference PyTorch implementation”.</b> Expected on AMD; results are correct.</li>
<li><b>“Model has not been downloaded yet”.</b> Use Download (the size is shown first) or pick an installed model.</li>
<li><b>“Could not connect to the API server”.</b> Is LM Studio / the server running? Does the URL end in <code>/v1</code>?
Use <i>List</i> to test.</li>
<li><b>A model needs custom code.</b> Enable “Allow custom model code” in Model options — only for trusted sources.</li>
</ul>
"""),
}

# Which topic F1 opens for each main tab index.
TAB_TOPICS = ["gallery", "caption", "editor", "datasets", "metadata", "train", "settings"]

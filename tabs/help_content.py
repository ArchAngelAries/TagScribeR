"""Help Center content. Keep this describing the *current* app.

Each topic: id -> (title, group, html). Hotkeys are listed in HOTKEYS and
rendered as a table.
"""
from __future__ import annotations

HOTKEYS: list[tuple[str, str, str]] = [
    # scope, keys, action
    ("Global", "Ctrl+1 … Ctrl+6", "Switch tab (Gallery, Auto Caption, Editor, Datasets, Metadata, Settings)"),
    ("Global", "F1", "Help for the current tab"),
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
</ol>
<p>Captions are stored the way trainers expect: <code>image.png</code> + <code>image.txt</code> in the same folder.</p>
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
               "<p>Tip: in <i>Batch</i>, choose <b>All shown</b> to apply an operation to everything the filter shows.</p>"),
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
<p>Your presets live in <code>user_data\caption_presets.json</code> and stay until you delete them.</p>
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
<li><b>Quantization</b> — 8-bit/4-bit need bitsandbytes (<code>install.bat --with-bnb</code>).</li>
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
<p>Batch rotate, resize, crop and convert the selected images.</p>
<ul><li><b>Save copies</b> (default) writes to <code>Image Edits\\&lt;source folder&gt;\\</code> — originals are untouched,
names never collide, and each copy keeps its caption file.</li>
<li><b>Overwrite originals</b> changes the files in place and asks for confirmation first.</li>
<li>Photo orientation (EXIF rotation) is respected.</li></ul>
"""),
    "datasets": ("Dataset Collections", "Tools", """
<h2>Dataset Collections</h2>
<p>Collections are folders (in <code>Dataset Collections\\</code>) where you gather finished images and captions for training.</p>
<ul><li><b>New</b> creates a collection; double-click one to browse it.</li>
<li>Load a source folder, filter and select images, then <b>Add to Collection</b> — images and captions are copied;
name clashes are renamed, never overwritten.</li>
<li>Deleting images or collections moves them to the Recycle Bin.</li></ul>
"""),
    "metadata": ("Metadata", "Tools", """
<h2>Metadata</h2>
<p>View and edit EXIF (JPEG) and text chunks (PNG), including Stable Diffusion generation parameters.</p>
<ul><li>Saving keeps color profiles and other binary data intact and writes safely.</li>
<li><b>Strip All Metadata</b> removes everything (asks first).</li></ul>
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
TAB_TOPICS = ["gallery", "caption", "editor", "datasets", "metadata", "settings"]

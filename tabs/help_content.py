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
    ("Image grid", "Click a selected image", "Deselect it (the rest stay selected)"),
    ("Image grid", "Arrow keys", "Move between images"),
    ("Image grid", "Ctrl+A", "Select all shown images; press it again to deselect them all"),
    ("Image grid", "Ctrl+Shift+A", "Deselect all images"),
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
    ("Auto Caption", "Ctrl+P", "Pause / resume the running job (nothing is lost)"),
    ("Image Editor", "Ctrl+R / Ctrl+Shift+R", "Rotate selected right / left"),
    ("Datasets", "Ctrl+N", "New collection"),
    ("Train", "Ctrl+Enter", "Start training (queues it while a run is going)"),
    ("Train", "Ctrl+Shift+S", "Sample now: render a preview round after the current step (while Training runs)"),
    ("Train", "Ctrl+Shift+D", "Save now: save the LoRA after the current step as <name>-stepNNNNNNN"),
    ("Train", "Ctrl+Shift+L", "Slider prompts: change a prompt slider's prompts while it trains"),
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
<p>Caption files in other encodings (UTF-16 as Notepad writes it, with a byte-order mark; Windows-1252) are read
correctly; when you edit one it is saved back as UTF-8, which every trainer reads.</p>
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
    "credits": ("Credits and licences", "Basics", """
<h2>Credits and licences</h2>
<p>TagScribeR is free, open-source software (GPL-3.0) and an independent project. It is not affiliated with,
endorsed by or a version of any other trainer or tool.</p>
<h3>Training engine</h3>
<p>Parts of TagScribeR's native training code are adapted from the open-source Fizgig trainer
(<a href="https://github.com/shootthesound/Fizgig">https://github.com/shootthesound/Fizgig</a>, Apache License 2.0,
Copyright Peter Neill), which in turn builds on work by kohya-ss (sd-scripts, musubi-tuner), Ostris (ai-toolkit),
the diffusers team and others. TagScribeR follows that project for which models and training features are worth
supporting; some speed and memory figures in this Help come from it and are marked "measured upstream". Every
adapted file names its source in its header, and the full list of sources and licences is in
<code>THIRD_PARTY_NOTICES.md</code> in the application folder.</p>
<h3>Preset files</h3>
<p>Preset files written by Fizgig use the same setting names and import as they are (Train tab, Import...).</p>
<h3>Models</h3>
<p>Model weights are never included. Each model has its own licence, shown or linked on its row under Model files.</p>
<h3>Support TagScribeR</h3>
<p>TagScribeR is free for everyone. If it saves you time, you can support its development on
<a href="https://www.patreon.com/c/ArchAngelAries">Patreon</a> (https://www.patreon.com/c/ArchAngelAries) or
<a href="https://ko-fi.com/archangelaries">Ko-fi</a> (https://ko-fi.com/archangelaries). The two small buttons at the
bottom of the sidebar open the same pages in your browser; nothing is sent from the app.</p>
"""),
    "gallery": ("Gallery overview", "Gallery", """
<h2>Gallery</h2>
<p>The Gallery is your dataset workspace: browse, filter, inspect and edit captions for thousands of images.</p>
<h3>Selecting</h3>
<p>Click to select, Ctrl+Click to add, Shift+Click for a range, Ctrl+A for everything shown. Press Ctrl+A again
(or Ctrl+Shift+A at any time) to deselect everything. Click a selected image
again to deselect it; the rest of the selection stays. Right-click for
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
<p>PNG, WebP and TIFF images above 80 megapixels are listed as unreadable ("too large to analyse") and left out of the
sharpness, resolution, crop and similarity checks (exact duplicates are still found), so a huge or hostile file can't
exhaust memory; thumbnails show them as unreadable too. Large JPEGs are read at reduced size and checked as usual. The
Image Editor still opens everything.</p>
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
<li><b>Allow this model to run its own code</b> — enabled only for a model that needs it, asked for and remembered per
model (and revision). Models built on Transformers' own architectures never run code from their repository. Only for
sources you trust.</li>
<li>A download that stopped partway shows <i>incomplete</i>; <b>Download (resume)</b> continues it.</li>
<li><b>Gated models</b> (Gemma): accept the licence on the model page, then run <code>hf auth login</code> or set
<code>HF_TOKEN</code>.</li>
<li><b>Captioning faster.</b> Three things decide how long a large job takes: how many requests the server works on
at once, how large each picture is, and how long each caption is.
<ul>
<li><i>Parallel requests</i> (API server): with <b>Find the best number of parallel requests</b> ticked, TagScribeR
measures as it goes and settles on the number that really is fastest for your server: 1 for a server that answers
one request at a time, more for a server with parallel slots (never more than <i>At most</i>). It backs off by
itself when the server reports overload.</li>
<li><i>Picture detail</i> (Advanced generation): a pixel limit on top of Max image size. For models of the Qwen
family fewer pixels mean fewer image tokens and a faster answer; keep <i>Full size</i> or <i>Detailed</i> when small
text or fine detail matters. <i>Full size</i> changes nothing.</li>
<li><i>Max tokens</i> bounds a reply that will not stop. <b>Stop a reply that repeats itself</b> (off by default)
ends such a reply early and keeps the caption up to the first repeat.</li>
<li>The line under the progress bar shows pictures per minute and the time left. <b>Pause</b> (Ctrl+P) holds the job
without losing anything: captions being written finish and are saved; Resume continues.</li>
<li><b>In LM Studio:</b> load the model before you start, set GPU Offload to all layers, set Max Concurrent
Predictions to 4, and give the model a Context Length of at least 8192 (16384 for full-size pictures).</li>
<li><code>tools/bench_caption.py</code> measures your own server on generated test pictures and prints the best
settings it found. These options were tested here against a simulated server only; the gain on a real model is for
that benchmark to show.</li>
</ul></li>
<li><b>API servers:</b> requests time out after 180 seconds and retry automatically: a local server is not retried
after a timeout, so a slow reply is never captioned twice; a cloud service is retried twice. Changing the server address clears the API key, and a key that
would travel over plain http to another machine is warned about. A saved key only works for the server address it was
saved for; clearing the key box and saving really deletes the stored key. A key with a line break or another
character that cannot be sent is refused when you type it. Keys never appear in the log file: anything that looks
like a key is blanked before a line is written. The Auto Caption log is plain text (it never loads images or links
from a server's reply) and keeps the last 4000 lines. A reply the app can't read fails only that image;
the batch continues.</li></ul>
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
<li><b>Overwrite originals</b> asks first whenever an original will be replaced, including <i>Convert</i> to the
file's own format. Color profiles and EXIF are kept; for PNG, JPG and WebP so are XMP, JPEG comments and PNG text
(A1111 / ComfyUI prompts). Photo rotation is applied to the pixels (and the orientation tags reset to match). Converting to another format writes the new file next to the original (originals
are never deleted); a PNG converted to JPG or WebP loses its prompt text, which those formats can't hold.</li></ul>
<p>The <b>Quality</b> slider applies to every JPG and WebP save, edits in place included. 16-bit PNGs are scaled to
8 bits for JPG, WebP and BMP (they stay 16-bit as PNG).</p>
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
<li><b>Clean metadata</b> removes what you tick. <i>Location</i> and <i>Camera / device</i> each remove GPS, time-zone
offsets, camera-internal MakerNote data, the whole IPTC / Photoshop block (city, country, by-line, contact, copyright),
embedded preview pictures (including the extra pictures of multi-picture files), content credentials, comments,
extended XMP, unknown vendor blocks, PNG text fields other than software, creation time, author, copyright,
description and title, and any bytes appended after the picture (an appended video or archive); an XMP block that
names a place or a person goes too. Device also removes make, model, lens, serial numbers and owner. Other choices:
AI generation data (prompts, seeds, models, ComfyUI workflows), XMP history, timestamps, software, or
<i>Everything</i> (only the pixels, orientation and, unless you choose otherwise, the color profile stay).</li>
<li><b>Checked after cleaning.</b> TagScribeR reads each cleaned file again and lists any file that still holds
something you asked to remove, or that could not be checked. The privacy audit never says "nothing found" about a file
it could not read. Artist and copyright you wrote yourself stay unless you choose Everything.</li>
<li>Cleaning is <b>lossless</b>: image data isn't re-encoded, so quality is identical. Photo orientation is always kept,
and the color profile is kept unless you tick the option to remove it.</li>
<li>Supported: JPEG, PNG and WebP. Caption .txt files are separate and never affected.</li>
</ul>
<h3>Authorship</h3>
<p>Write your artist name, copyright / license, description and the software you used to the standard EXIF and PNG
fields — exactly the values you enter. Save them as a <b>template</b> to reuse. EXIF can only hold plain ASCII, so “©” is
written as “(c)” there; PNG keeps the exact text.</p>
<h3>Camera data</h3>
<p>Write EXIF that mimics a real camera. Pick one of the built-in presets (Canon, Sony, Nikon, Fujifilm, Panasonic,
OM System, Leica, Ricoh, iPhone, Pixel, Galaxy, a DJI drone ...), adjust any field, and <b>Apply camera data</b> to the
selected or all shown images. It writes Make, Model, firmware, lens, aperture, shutter, ISO, focal length (real and
35&nbsp;mm equivalent), exposure mode, metering, flash, white balance, exposure bias and the date taken. Blank fields
are left unchanged. Your own presets can be saved and deleted; built-in ones can't be changed.</p>
<ul>
<li><b>Per-image variation</b> (off by default) picks shutter, ISO and aperture on the usual third-stop scales inside
the ranges you set. It is seeded from the file name, so applying again gives identical values.</li>
<li><b>Timestamp series</b> gives the first image your start time and every following image one more step, in grid order.</li>
<li>Lossless like the other tools: pixels, orientation and color profile are untouched. JPEG, PNG and WebP (a plain
WebP without a metadata header must be converted first).</li>
<li>Serial numbers, owner name, image IDs and GPS are never written. <i>Privacy → device</i> removes what this adds.</li>
<li>The built-in strings are typical values written from general knowledge; entries marked "verify" in
<code>core/camera_presets.py</code> have not been checked against a file from that camera.</li>
<li>The values describe a camera that did not take the picture. Use it for style or testing, not to pass an image off as a photograph.</li>
</ul>
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
<li><b>Themes</b> — light and dark themes in Settings → Appearance → Theme. A theme applies at once, without a
restart. <i>TagScribeR dark</i> and <i>TagScribeR light</i> are the new themes with larger, clearer controls;
<i>dark_teal</i> is the classic look and stays the default, and the other classic themes stay available.</li>
<li><b>Scrolling never changes a setting.</b> The mouse wheel over a dropdown, number box or slider scrolls the
panel it sits in; click the control first (or Tab to it) to change it with the wheel.</li>
<li>Every tool can be driven from the keyboard — see <i>Keyboard shortcuts</i>.</li>
</ul>
"""),
    "train": ("Training a LoRA", "Train", """
<h2>Training a LoRA (Train tab)</h2>
<p>TagScribeR trains LoRAs natively, with built-in presets and an adaptive learning rate (see <i>Credits and
licences</i> for where the training code comes from). It trains the dataset folder that's open in the workspace, so the captions you just fixed are the
ones it learns.</p>
<ol>
<li><b>Open and caption your dataset</b> (Gallery / Auto Caption). Every image needs a caption file, so filter
<code>missing:caption</code> to find gaps. Put your trigger word in the captions (Auto Caption → Subject).</li>
<li><b>Pick a family</b> (Krea 2, Qwen Image 2.1, MiniMax H3, FLUX.2 Klein, MOSS-Image, SDXL / Pony / Illustrious / NoobAI, Anima) and set its <b>Model files</b> once. <i>Get</i> opens the download page;
the paths are remembered.</li>
<li><b>Load a preset.</b> ◆ presets are the built-in recipes; the first one is applied on your first visit.</li>
<li>The tab starts in a <b>simple view</b>: name, epochs, resolution and preview prompts, with the preset deciding
the rest. Tick <b>Show all settings</b> for full control of every option.</li>
<li>Set the <b>LoRA name</b> (and output folder), check the <b>preview prompts</b> (Samples section), then
<b>Start Training</b> (Ctrl+Enter).</li>
<li>Watch the <b>Loss</b> chart and the <b>Samples</b> (see <i>Watching and steering a run</i>: per-step loss, Sample
now, Save now). Every saved epoch is a usable LoRA: pick the best-looking one, not necessarily the last.</li>
</ol>
<h3>What happens when you press Start</h3>
<ul>
<li>The settings are checked first (missing captions or model files stop it with a clear message).</li>
<li>They are frozen into a run folder (<code>&lt;output&gt;/&lt;LoRA name&gt;/</code>), so changing the tab during a run
can't affect it.</li>
<li>Training runs as separate processes, so the app stays responsive and a crash can't take it down.</li>
<li>Stages: <b>caching latents</b> → <b>caching captions</b> → <b>training</b>. A picture (or its edit / slider partner)
edited in place since it was cached is re-encoded on the next run (by its saved modified time and size). Caching only
encodes what
changed since the last run.</li>
<li><b>Edit LoRA</b> (Qwen Image 2.1 and FLUX.2 Klein): tick it and pick the <i>Originals folder</i>. The dataset
folder holds the edited ("after") images, the originals folder the matching "before" images with the same file names.
The LoRA learns to turn one into the other, for example a colour grade. Previews apply the edit to the first original,
or to the photo you set under Samples.</li>
<li><b>Slider LoRA</b> (every family): the LoRA's strength becomes a dial. At +1 it pushes one way, at -1 the
other, and 0 is the plain model. Two kinds (<i>Slider from</i>):
<ul>
<li><i>pairs</i>: the dataset folder is the +1 end; the <i>Slider -1 end folder</i> holds the same pictures changed
the other way, with the same file names and the same framing (for example a photo and its darker version). The
LoRA learns what differs between the two ends, not what they share. Partners may also be named
<code>photo_anything.png</code> for <code>photo.png</code>.</li>
<li><i>prompts</i>: no dataset. Fill in the base prompt (what the picture is) and what each end adds (for example
"a photo of a woman", "smiling", "frowning"). The model first renders 16 practice pictures of the base prompt, then
learns to push them toward one prompt at +1 and the other at -1. <i>Slider guidance</i> sets how far.</li>
</ul>
A slider is always a plain LoRA; Adaptive LR, EMA, hard-image tracking and gradient accumulation are off for it, and it
trains at batch size 1. Previews show the dial at -1, 0 and +1 side by side. Krea 2, Qwen, SDXL, Anima and MiniMax H3 have a
Slider preset (tried upstream on Qwen, Krea 2 and MiniMax H3, not by TagScribeR; starting points on SDXL and Anima). Krea 2 adds <i>Train composition blocks only</i>
(for sliders that hold up at higher strengths). The file records that it is a slider
(<code>ss_slider</code>).</li>
<li><b>Fine-tune the whole model</b> (Show all settings; Krea 2, Qwen Image 2.1, FLUX.2 Klein, MOSS-Image, the SDXL
family and Anima): the base model itself trains instead of a LoRA, and the result is a full model file. See <i>Full
fine-tune</i>. A run is one kind of training: ticking Edit, Slider or Fine-tune unticks the other two.</li>
<li>A <b>Companion LoRA</b> (Show all settings) stays frozen and active while you train, so the new LoRA learns on top
of it; it is never saved into the output. LoRA, LoKR and LoHa files load, in kohya, diffusers, OneTrainer or
AI-Toolkit naming. A companion LoRA that matches none of the model's layers (a LoRA made for another model) stops
the run with a message, instead of training on as if it were not there.</li>
<li>Tokenizers are read from the local Hugging Face cache first, so once the model downloader has fetched them, a
run starts without an internet connection.</li>
</ul>
<h3>Pause, Stop, Resume, Queue</h3>
<ul>
<li><b>Pause</b> finishes the current epoch, saves a resumable state and exits cleanly; <b>Resume</b> continues exactly
there. <b>Stop</b> ends the run immediately (saved epochs stay). Only state folders written by TagScribeR's trainer
can be resumed (they are loaded as plain tensors, never as arbitrary Python objects). If you resume with a
different <i>Optimizer</i> than the run used, the LoRA's weights and the epoch carry on and the optimizer starts
fresh (the console says so); to carry its state on too, resume with the optimizer the run used.</li>
<li>To train more epochs on a finished LoRA: raise <i>Epochs</i>, click <i>Latest</i> under Resume, Start. The
finished LoRA is also saved under its epoch number (for example <code>name-000030.safetensors</code>), so extending a
run never overwrites its last epoch.</li>
<li>A <b>fine-tune</b> has no state folders: Pause waits for the rotation in progress to finish, saves a full
checkpoint and stops; Resume continues from that checkpoint for the rotations that are left.</li>
<li>Start while a run is going <b>queues</b> the current settings; queued runs start one after another. The queue is
never started automatically when the app opens.</li>
<li><b>Last run</b> reloads the settings of the most recent run.</li>
</ul>
<p>See also: <i>Training presets</i>, <i>Adaptive learning rate</i>, <i>Hard images</i>, <i>Training previews</i>,
<i>Full fine-tune</i>, <i>MOSS-Image</i>, <i>Training memory &amp; speed</i>, <i>Credits and licences</i>.</p>
"""),
    "train_presets": ("Training presets", "Train", """
<h2>Training presets</h2>
<ul>
<li><b>◆ Built-in presets</b> come with each family. They are ready-made recipes (rank, learning rate or Adaptive LR range,
epochs, resolution) and can't be changed or deleted.</li>
<li><b>Save…</b> stores the current settings as your own preset, in
<code>user_data/training_presets/&lt;family&gt;/</code>. Presets never contain the model family, so loading one can't
switch your model. Preview settings and Resume aren't part of presets either.</li>
<li>A preset that sets a folder or file (output folder, originals folder, slider -1 end folder, Companion LoRA, thumbnail)
is named in the console when you load it: check those paths, especially in a preset someone else made.</li>
<li><b>Caption extension</b> (Dataset, Show all settings) must start with a dot, like <code>.txt</code> or
<code>.caption</code>; an image extension is refused, since a caption written there would replace the picture.</li>
<li><b>Import…</b> adds a preset file. Preset files from other trainers that use the same setting names import as they are (see <i>Credits and licences</i>). If an imported file sets a folder or a file on your disk (output folder, edit folders, companion LoRA, thumbnail), TagScribeR asks first and by default applies the preset without those paths.</li>
<li><b>Export for sharing…</b> saves the current settings to a file you can send to someone: without any folder or
file paths and without personal text (LoRA name, author, title, description, tags, trigger phrase).</li>
<li><b>Copy log for sharing</b> (under the console) copies the console text with your user name and your folders
replaced by placeholders such as &lt;home&gt; and &lt;path1&gt;, so a log pasted into a bug report does not show
where your files live. The files in the run folder themselves (<code>run.log</code>, <code>train_config.json</code>,
<code>dataset.json</code>) keep full paths, because the run needs them: use the copy button when you share.</li>
<li>When a preset holds a value this family doesn't offer (an optimizer, a learning-rate bound, a precision), that
setting is kept as it was and the console says so. Settings this version doesn't know are ignored.</li>
<li>The <b>Slider</b> and <b>Fine-tune</b> presets work on the families that offer those kinds of training.
On a family that does not (a Fine-tune preset on MiniMax H3, whose fine-tune is not ported yet) the preset is refused
with a message and changes nothing: applied anyway it would train an ordinary LoRA.</li>
</ul>
"""),
    "train_finetune": ("Full fine-tune", "Train", """
<h2>Full fine-tune (training the model itself)</h2>
<p>A LoRA is a small file that sits on top of a model. A <b>full fine-tune</b> changes the model itself: the result
is a complete model file, as large as the one you started from, that you load <i>in place of</i> the base model. Use
it when a LoRA is not enough (a large or varied dataset, a new base for further LoRAs). For one person, one style or
one concept a LoRA is faster, far smaller and usually all you need.</p>
<p><b>Status:</b> adapted from open-source training code and tested here only on a tiny made-up model on the CPU. It has not been run
on real weights by TagScribeR, on AMD or on NVIDIA; the upstream project documents its fine-tune as tested on NVIDIA cards. Treat
your first run as a trial and keep an eye on the console.</p>
<h3>How to start one</h3>
<ol>
<li>Pick a family that offers it: Krea 2, Qwen Image 2.1, FLUX.2 Klein, MOSS-Image, SDXL / Pony / Illustrious /
NoobAI or Anima. (MiniMax H3's fine-tune needs its own back end, which is not ported yet.)</li>
<li>Tick <b>Show all settings</b>, then <b>Fine-tune the whole model</b> (Training Parameters), or load a ◆
Full fine-tune preset where the family has one (SDXL, Anima, MOSS-Image). Ticking it sets the learning rate to the
family's fine-tune rate (1e-5) and turns Adaptive LR off.</li>
<li>Set <b>Rotations</b> (how long it trains) and check the preview prompts. Start.</li>
</ol>
<h3>What is different from a LoRA run</h3>
<ul>
<li><b>The model file must be 16-bit or fp32</b> (bf16 / fp16 / fp32). An fp8 or pre-quantised file is refused with a
message: there are no full-precision weights in it to train.</li>
<li><b>Windows and rotations.</b> The frozen model sits on the card in 4-bit (NF4). One <i>window</i> of its parts
at a time (for example every block's attention, then every block's MLP) is brought up to full precision, trained and
written back. A <b>rotation</b> trains every part once. On a card that holds the whole model at once there is one
window and nothing rotates. The console says how many windows your card needs and how many epochs one rotation is.</li>
<li><b>Rotations replace Epochs.</b> <i>Checkpoint + preview every N rotations</i> replaces <i>Save every N
epochs</i>. Saves only happen at whole rotations, so no saved file has some parts trained more than others.</li>
<li><b>Every save is a full model</b> (several GB to tens of GB). Keep the output folder on a roomy drive.</li>
<li><b>System memory:</b> a full-precision master copy of the trained weights lives in RAM, or on disk in the run
folder when RAM is short (the console says which).</li>
<li><b>Not used:</b> rank, alpha, network type, Model Area, the timestep range, Adaptive LR, EMA, the LR scheduler,
torch.compile, the Companion LoRA, auto-recaption, identity blocks only, Sample now / Save now and Resume from a state
folder. The boxes that do not apply are hidden.</li>
<li><b>Optimizer:</b> Adafactor (its state is about a tenth of AdamW's, which is what keeps a fine-tune on the
card), whatever the Optimizer box says.</li>
<li><b>Step each weight as its gradient arrives</b> (on by default) steps each weight the moment its gradient arrives,
so only one gradient is held at a time. Gradient clipping and gradient accumulation are off with it.</li>
<li><b>Window size</b> (Auto by default): choose a smaller cap if Auto runs too close to your card's limit.</li>
<li><b>Regularisation images (optional):</b> a folder of ordinary, well-captioned real photos of the broader class
(men, women, people). They train at a fraction of the learning rate (<i>Regularisation LR x</i>, 0.2 by default) and
keep the model's general sense of that class from drifting. Use real photos, not generated ones.</li>
<li><b>Previews</b> render on the model being trained, after each saved checkpoint.</li>
<li><b>Pause</b> finishes the rotation in progress, saves a checkpoint and stops. <b>Resume</b> continues from that
checkpoint for the rotations that are left.</li>
<li>The checkpoint is written in the source file's own layout and names, so it is meant to load wherever the base
model loads (ComfyUI included); that has not been tried here on a real model yet. Its metadata keys start with <code>tagscriber_</code> (models saved with the older key names are still read).</li>
<li><b>Krea 2 note:</b> the fp8-scaled RAW file many people train LoRAs on cannot be fine-tuned (it is refused
before the run starts); a fine-tune needs the bf16 model file.</li>
<li><b>Never under the same name as its source:</b> to train a finished fine-tune further, set its file as the
model file and give the new run another LoRA name. A run refuses to start from one of its own checkpoints.</li>
<li><b>Batch size 1</b> is the path the memory plan is measured for; a larger batch gets a warning.</li>
</ul>
"""),
    "train_moss": ("MOSS-Image", "Train", """
<h2>MOSS-Image</h2>
<p>MOSS-Image (OpenMOSS) is a text-to-image model with strong text rendering, a Qwen3.5-9B text encoder and the
FLUX.2 autoencoder. The family is adapted from open-source training code (upstream version 8.7.3). <b>Status:</b> tested here only on tiny made-up
models; it has not been run on real weights by TagScribeR. The upstream project has changed its MOSS-Image recipes since that
version (a newer optimizer, a low-noise schedule that changes during the run, a different preview recipe); those
are on the list to follow.</p>
<ul>
<li><b>Train on Base, use the LoRA on SFT.</b> Base is the model to train on; SFT is the sharper model people
generate with. Previews therefore render on the SFT model by default (set its file under Model files), so they
show the LoRA where you will use it.</li>
<li><b>Another Base</b> (a fine-tune of Base, or SFT itself): the previews tick switches off once, because the model
you train on is then the model you will use. You can tick it again.</li>
<li><b>Train attention only</b>: recommended when the model you train on is SFT or a fine-tune of it (training its
feed-forward layers breaks anatomy). On Base leave it off.</li>
<li><b>Low-noise share (% of steps)</b> (Timesteps): how much of the training happens on nearly clean images, where
faces and fine detail are refined. 30 is the default and an experimental figure; empty uses the model's own
schedule (about 11%).</li>
<li><b>Epochs:</b> the presets use 70. MOSS-Image learns slowly and steadily; small datasets were still improving
late in the upstream project's runs. Adaptive LR is off in the presets (it walks down to its floor on this model's flat loss).</li>
<li><b>Captions</b> may be long (up to 4096 tokens) and are read through the model's chat template.</li>
<li><b>No Edit LoRAs:</b> the released checkpoints are text-to-image only.</li>
<li><b>Full fine-tune:</b> offered (see <i>Full fine-tune</i>); with Train attention only it trains attention alone
and fits a smaller card.</li>
<li><b>First run on your card:</b> the Qwen3.5 text encoder runs in plain PyTorch on every GPU. It has not been
checked on an AMD card here, so compare your first previews with a known-good render before a long run.</li>
<li><b>Previews with the SFT tick on</b> use the SFT model's own recipe (30 steps, CFG 4) and your negative prompt;
the Turbo LoRA and its steps apply only when previews render on the training model (tick off).</li>
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
    "train_watch": ("Hard images (per-image loss)", "Train", """
<h2>Hard images</h2>
<p><b>Track hard images</b> follows each image's loss through the run. The loss is corrected for how noisy each
step was, so images are compared fairly. Open <b>Hard images</b> during a run to see the verdicts:</p>
<ul>
<li><b>stuck</b>: consistently harder than the rest and not improving. Usually a wrong or misleading caption, or an
image that doesn't fit the set.</li>
<li><b>suspect</b>: unusually high loss. <b>exhausted</b>: learned early, then stopped improving.
<b>easy</b> / <b>mid</b> / <b>learning</b>: fine.</li>
<li>Edit a caption there and click <b>Save fix</b>. The caption file is saved (with a backup), and the run re-encodes
it at the next epoch.</li>
<li><b>Per-image learning rate</b> trains stuck images more gently (×0.5 down to ×0.1) and easy ones slightly harder
(×1.1).</li>
<li><b>Recaption images that stay hard</b> re-captions them between epochs with the captioner you choose (a local vision
model or the WD tagger). After two failed attempts an image is set aside.</li>
<li>Set-aside images are recorded per model family in <code>tagscriber_excluded.json</code> in the dataset folder.
A later run trains them normally; if one gets stuck again it is set aside at once, without new recaptions. Editing
its caption clears the record.</li>
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
<li>On cards under 20 GB the model steps aside while a preview is decoded. Krea 2 previews are also rendered
on a 768 px canvas there; every other model keeps the preview size you set (shrunk previews made some models look
soft or broken).</li>
<li>A failed preview (e.g. out of memory) turns previews off for the rest of the run. Training continues.</li>
<li><b>Preview override…</b> renders a different prompt at the next preview round without restarting.</li>
<li>Each saved checkpoint carries its own epoch's preview as its thumbnail (shown by ComfyUI's model browser),
unless you set a thumbnail yourself in the Metadata section.</li>
<li>With a <b>Turbo / speed LoRA</b> the preview uses the LoRA's own CFG (usually 1). Set CFG above 1 to guide it
more; the negative prompt then applies too.</li>
<li>Preview time is left out of the speed shown on the progress bar.</li>
<li>A family's <b>Turbo LoRA</b> (if you set its file) makes previews much faster. Strength 0 turns it off; -1 uses the
family default.</li>
</ul>
"""),
    "train_live": ("Watching and steering a run", "Train", """
<h2>Watching and steering a run</h2>
<p>These came with upstream version 8 of the training code and are unit-tested on tiny test models only. They have <b>not yet been run on real
hardware</b>: if one misbehaves on a real run, please tell us.</p>
<h3>The loss graph</h3>
<ul>
<li>The <b>Loss</b> tab now draws the loss of <b>every step</b> as it trains (a faint line), with a bright
<b>smoothed</b> line over it, the <b>average of each epoch</b> (green dots), the <b>learning rate</b> (dashed orange,
its own scale) and a faint line where each epoch starts (e1, e2...). Point at the graph to read a step: its number,
epoch, loss and learning rate.</li>
<li>A smooth line going down is what you want, but loss alone doesn't tell you when the LoRA is best. Judge by the
sample previews.</li>
<li>The graph updates every few seconds while a run trains, and shows the curve of a finished, paused or earlier run
with the same LoRA name too. A resumed run carries on its own curve.</li>
<li>Very long runs are thinned to a few thousand points so the graph stays quick ("every 4th drawn").</li>
</ul>
<h3>Sample now and Save now</h3>
<ul>
<li><b>Sample now</b> (Ctrl+Shift+S) renders a preview round straight away, after the current step, with the LoRA as
it is now. The pictures show in <b>Samples</b> under the epoch in progress.</li>
<li><b>Save now</b> (Ctrl+Shift+D) saves the LoRA right after the current step, as
<code>&lt;name&gt;-step0000123.safetensors</code> beside the epoch files. It shows in the <b>Checkpoints</b> tab. It is
a LoRA file only, not a resume state: use Pause to be able to continue.</li>
<li>Both are only available while the <b>Training</b> stage runs (not while latents and captions are cached).
Sample now also needs previews to be set up: <i>Every N epochs</i> above 0 and a prompt.</li>
<li>The console confirms the request, and the trainer logs <code>[sample now]</code> / <code>[save now]</code> when it
acts.</li>
</ul>
<h3>Prompt sliders: change the prompts while it trains</h3>
<ul>
<li>While a <b>prompt-pair slider</b> (Slider LoRA from prompts) trains, <b>Slider prompts...</b> (Ctrl+Shift+L)
opens its base prompt, what each end adds and the slider guidance. Apply hands them to the run, which takes them at the
end of the epoch. A new base prompt makes the model render its practice pictures again (a short pause).</li>
<li>The window shows what the trainer last applied: from which epoch, and whether the practice pictures were rendered
again. The saved LoRA records the history of its prompts.</li>
<li>Image-pair sliders have no live prompts. A slider run <b>ignores the sample override</b> (Preview override...): its previews are
always the dial at -1, 0 and +1 on the base prompt.</li>
</ul>
<h3>Which prompt made this preview?</h3>
<ul>
<li>Each picture in <b>Samples</b> shows the prompt it was rendered from under its label, and in full in its tooltip,
so a picture from before a Preview override or a slider prompt change still shows the prompt that made it.</li>
</ul>
<h3>Klein: the Distilled tick and Steps / CFG</h3>
<ul>
<li>With Klein's <b>Preview with the Distilled model</b> ticked and the Distilled file set under Model files, the
<b>Steps</b> and <b>CFG</b> boxes move to the Distilled's own recipe (4 steps, CFG 1); unticked, they go back to the
training model's. A value you typed yourself is never replaced.</li>
<li>A Steps or CFG you typed is kept, unless it equals one of the two defaults (the plain model's or the
checkpoint's), which the tab and the trainer treat as untouched. CFG above 1 brings the negative
prompt.</li>
</ul>
<h3>Edit LoRAs with extra pictures</h3>
<ul>
<li>For families whose edit takes several pictures (<b>Klein</b>: up to 4, <b>Qwen Image 2.1</b>: up to 10), the Edit
card (Show all settings, Edit LoRA on) asks <b>Edit inputs</b>. <b>The photo only</b> is an edit that changes how
a photo looks (a grade, a relight, a retouch). <b>The photo + extra pictures</b> brings something from other pictures
into it (a person, an object, a place).</li>
<li>Your <b>edited photos and their captions</b> are the dataset folder: <code>photo.png</code>,
<code>photo.txt</code>. The <b>Originals folder</b> holds <code>photo.png</code> at the same crop. The <b>Extra
pictures folder</b> holds <code>photo_1.png</code>, <code>photo_2.png</code> ... (any size or shape; each is resized to
the target's pixel count in its own shape). Up to <b>3 extras for Klein, 9 for Qwen Image 2.1</b>. A photo can have
none (it trains as a normal pair). Extras may also sit in the Originals folder.</li>
<li>The number is the order the model sees them in: say so in the edit instruction ("Put the man from the second
picture into this scene").</li>
<li>For previews, the edit preview photo's own extras are <code>test_1.png</code>, <code>test_2.png</code> ... in the
Extra pictures folder (for <code>test.png</code>); without a test photo, the first original's extras are used.</li>
<li>File names are not case-sensitive.</li>
</ul>
<h3>For power users: the files a run writes</h3>
<ul>
<li><code>.tagscriber_loss_&lt;name&gt;.jsonl</code> in the run folder: one line a step,
<code>{"s": step, "e": epoch, "l": loss, "lr": rate}</code> (the rate only now and then). A resumed run keeps the lines
up to its resume step and carries on.</li>
<li><code>&lt;name&gt;-step0000123.safetensors</code>: what Save now writes.</li>
<li><code>sample/.prompts_&lt;name&gt;.json</code>: the prompts each preview round used, by epoch
(<code>{"0": [...], "3": [...]}</code>; a Sample now uses the epoch in progress).</li>
<li><code>.sample_now</code>, <code>.save_now</code>, <code>.slider_prompts.json</code> (answered in
<code>.slider_prompts_applied.json</code>), <code>.pause_requested</code> and <code>.sample_override.json</code>: the
control files. The trainer takes each once, between steps or at an epoch boundary.</li>
</ul>
<h3>Also new in this trainer</h3>
<ul>
<li>A failed <b>resume-state save</b> no longer ends a run: it is tried twice, a half-written state folder is
removed, and the run warns and trains on. A Pause whose save fails still stops loudly, because you asked for a state
to resume from.</li>
<li>With <b>Compile Blocks</b> on, a quick test compile runs first, and previews are rendered <b>uncompiled</b>
(eager), so the first preview no longer waits minutes for the compiler; training keeps its compiled graphs.</li>
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
<li>With <b>Blocks to swap</b> on an INT8 base, the swapped blocks' weights wait in system RAM, so the whole INT8
model never has to fit on the card first.</li>
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
<li><b>Base precision: Auto</b> tries INT8 first if your GPU supports it and it fits, then 4-bit (needs
bitsandbytes), then INT8 with block swap. On a machine that runs neither INT8 nor 4-bit, Auto uses fp8. INT8 and
4-bit are made from whichever file you picked. Choose <b>fp8</b> yourself to train on the fp8 weights exactly as they
ship.</li>
<li>The <b>Turbo LoRA</b> is optional and only for previews: the RAW model plus the LoRA at strength 0.7 renders in 10
steps at CFG 1 without a second model (the built-in setup; the Samples Turbo strength box changes it). Leave it empty to
preview the RAW model at 28 steps and CFG 4.5 (the built-in RAW recipe).</li>
<li>Captions are encoded one at a time, so each caption's cache never depends on its neighbours.
Krea 2 text caches made by older versions are re-encoded once on the next run (the image caches are kept).</li>
<li>The text encoder file can be the fp8_scaled or the bf16 Qwen3-VL-4B. Each cached caption is about 30 MB, so
leave room on the cache drive.</li>
<li>Measured upstream on NVIDIA cards, not by TagScribeR, at 0.25 MP: INT8 about 16 GB, 4-bit about 11 GB, fp8 about 19 GB, bf16 about 26 GB (each
swapped block saves about 0.4 GB, 0.8 GB on bf16). At 1 MP: INT8 about 19 GB, 4-bit about 15 GB. So on a 20 GB card
Auto picks 4-bit for 1 MP training and INT8 for 0.25 MP; on a 16 GB card at 1 MP it picks INT8 with block swap. Each extra image in the batch adds about 2.4 GB.</li>
<li>The <b>fp8-scaled text encoder</b> also stays fp8 in memory, which saves VRAM while captions are cached.</li>
<li><b>Automagic v3</b> (Optimizer) sets its own learning rate for each group of layers: text fusion, attention, MLP
and input/output. The Learning rate box is only its starting value, and Adaptive LR, the scheduler and per-image LR
are ignored with it.</li>
<li><b>Compile blocks</b> (Memory &amp; Precision, under Show all settings) mainly speeds up NVIDIA cards. Auto leaves
it off on AMD, with block swap, on fp8 and bf16 bases, and on short runs. On Windows it needs Triton and the C++ Build
Tools; without them the run trains normally and says so.</li>
</ul>
<h3>Qwen Image 2.1</h3>
<ul>
<li>Keep the <b>training adapter</b> on: it is frozen during training, off in previews and never in your saved
file. Without it Qwen 2.1 LoRAs tend to collapse or wobble. It is now <b>version 2</b> (the current file): put the
v2 file next to the v1 file and TagScribeR switches to it on its own. A LoRA trained on v2 is not directly
comparable with one trained on v1.</li>
<li><b>Previews</b> use the Comfy-Org <b>Turbo LoRA</b> by default (0.91 GB, set it under Model files): 8 steps, CFG 1.
Without it (Samples: Turbo strength 0) previews render on the plain model at 25 steps and CFG 3, with a general
negative prompt.</li>
<li><b>Train identity blocks only</b> (the "Identity blocks only" preset, or the checkbox under Training Parameters) trains only
blocks 10-14, where a character's identity was measured upstream (not by TagScribeR) to sit. About 1.5x faster per step, with likeness very close
to a full-model LoRA. Use it for characters and faces, not styles. It is not used for Edit LoRAs.</li>
<li><b>Compile blocks</b> works on Qwen too: an upstream figure is 1.43x per step on INT8 and 1.15x on bf16 with no extra
memory, on NVIDIA. Auto compiles runs of 300+ steps on INT8 and 800+ on bf16, and leaves it off on AMD and with block
swap. This has not been run on real hardware in TagScribeR yet.</li>
</ul>
<h3>MiniMax H3</h3>
<ul>
<li><b>Images only.</b> TagScribeR trains H3 LoRAs from still images. Clips, sound and voice items aren't
supported here.</li>
<li><b>Files:</b> the pruned INT8 DiT (about 21 GB), the Qwen3-VL-32B text encoder, the H3 video VAE, and the optional
training adapter. <i>Get</i> opens each download page.</li>
<li><b>Memory:</b> loading the DiT needs about 22 GB of free system RAM. Caching captions needs about 15 GB of free
VRAM; with less, the text encoder streams from system RAM and is slower.</li>
<li><b>Older NVIDIA cards</b> (before the RTX 30 series, e.g. V100 or T4): long captions are encoded in chunks (same
result, lower peak) and very long training items use fp16 attention instead of bf16, both automatically. Nothing
changes on RTX 30 and newer, AMD or CPU. Adapted from open-source training code and unit-tested here, not yet run on such a card: please
report results.</li>
<li><b>Training base</b>: the standard First/last frame (fl2va) model, or <i>Reference (ref2va)</i> if the LoRA will
be used in ComfyUI's Reference-to-Video workflow (set the optional reference DiT file).</li>
<li><b>Training adapter</b>: <i>Circlestone</i> (the default, best for photos; one file for both bases),
<i>Ostris</i> (learns a video look faster; TagScribeR picks its fl2va or ref2va file to match the base), or
<i>Off</i>. It is on for every training step, off for previews and never saved into your LoRA.</li>
<li><b>Noise emphasis</b>: <i>Likeness and Style</i> puts 60% of the steps on nearly clean images (the tuned
default); <i>Model default</i> is the reference trainer's 8%, weighted to movement; <i>Custom</i> uses your own
<i>Low-noise share</i>.</li>
<li><b>Blocks preset</b>: <i>Default</i> trains blocks 20 to 49 (best likeness, quickest steps); <i>More Blocks</i>
6 to 49; <i>Off</i> trains the blocks you type in <i>Blocks to train</i> (numbers and ranges, or the
<code>h3blk_N</code> ids).</li>
<li>Under <i>Show all settings</i>: <b>High-noise learning rate</b> (the learning rate of the noisy-half steps; leave
at 100), <b>LoRA-size-relative learning rate</b> (the learning rate becomes a ceiling the run climbs toward as the LoRA grows; set
the rate where you want to end up) and <b>Train the text token refiner</b> (recommended off; it trains only in mode
Off with every block). The first two do nothing with Automagic v3, which sets its own rate.</li>
<li>The presets use Automagic v3 (it sets its own learning rate, so Adaptive LR is off), caption dropout
0.05 and EMA 0.98. EMA also offers <b>Short run</b>, which averages about the last quarter of a short run. Other
optimizers get an H3 weight decay of 1e-4. The LoRA trains in bf16.</li>
<li>Previews run at 20 steps on the training model, 768 x 768. Set the optional <b>Turbo LoRA</b> under Model files
and they render in 6 steps with it at 75% (Samples: <i>Turbo steps</i> and <i>Turbo strength</i>; 0% turns it off).
It is on only while previews render and is never saved into your LoRA. Its time-modulation rows, and those of an older
H3 LoRA used as a Companion LoRA, are applied at run time.</li>
<li>Not available yet: HQQ 4-bit, the 66 GB bf16 DiT, clips, sound and distillation. Sliders from still pairs or
prompts work (the Slider preset is the upstream recipe).</li>
</ul>
<h3>FLUX.2 Klein Base 9B</h3>
<ul>
<li><b>Files:</b> the Base 9B checkpoint (the fp8 file or the bf16 one), the FLUX.2 <code>ae.safetensors</code>
autoencoder, and the Qwen3-8B text encoder. <i>Get</i> opens each download page; the Klein and autoencoder repos are
gated, so accept their terms on Hugging Face first.</li>
<li><b>Model area</b> chooses which blocks the LoRA trains: <i>Full Model</i>, <i>Identity</i> (a subject's
likeness), <i>Style</i> and <i>Style+Composition</i>, <i>Details</i>, or <i>Custom</i> (list the blocks yourself, as
<code>double_0</code>-<code>double_7</code> and <code>single_0</code>-<code>single_23</code>; the older
<code>double_blocks.0</code> spelling still works). The built-in presets set it for you. The blocks a LoRA trained are
written into its file (<code>ss_train_blocks</code>).</li>
<li><b>Network type</b> can be LoKR as well as LoRA, and <b>EMA</b> is available (Off by default). The
Optimizer list offers every optimizer TagScribeR has.</li>
<li><b>Timestep sampling</b> defaults to <code>flux2_shift</code>, which adapts to the image size. The other modes
are for experiments.</li>
<li><b>Previews</b>: set the optional <i>Klein 9B Distilled DiT</i> under Model files and previews render on it in 4
steps, as ComfyUI renders your LoRA (Samples: <i>Preview with the Distilled model</i>, on by default). The training
model steps aside while they render and comes back exactly as it was. Without the file, or with the box unticked,
previews render the Base model at 40 steps and CFG 4.5. <i>Keep the sample model in RAM</i> saves reloading it each
epoch on 24 GB+ cards; <i>INT8 sample model</i> makes it smaller and faster on NVIDIA. Not run on real hardware in
TagScribeR yet.</li>
<li><b>Attention mechanism</b> (under Show all settings): <code>sdpa</code> runs on any GPU and is the default.
<code>flash3</code> is listed because the upstream project lists it, but there is no working flash3 path here, so choosing it stops the
first training step with an error. Use sdpa.</li>
<li><b>Base precision: Auto</b> picks INT8 when it fits (about 12.6 GB at 0.25 MP, 17.1 GB at 1 MP), else
4-bit (7.9 / 9.9 GB). Auto never block-swaps Klein. When the run is not compiled and your Base file is fp8, Auto
trains the fp8 file as it is instead of INT8: uncompiled that is faster (0.77 vs 1.14 s/step, measured upstream on NVIDIA cards, not by TagScribeR) and
smaller (11.1 GB). <b>fp8</b> is that same choice by hand (a preset file may call it "As the file").</li>
<li><b>Compile blocks</b> works on Klein too (an upstream figure is 2.1x per step on INT8 with less memory, 1.4x on 4-bit,
on NVIDIA). Auto compiles INT8 runs of 200+ steps and 4-bit runs of 400+ when the compiled run fits your free VRAM,
never the fp8 file, and leaves it off on AMD and with block swap. Not run on real hardware in TagScribeR yet.</li>
<li><b>Edit LoRAs</b> work on Klein too: the "before" images reach the model next to the image being made (Klein's
text encoder never sees them). About 40 pairs is a good start, 20 at least.</li>
<li>A <b>Companion LoRA</b> made for Klein by other trainers loads too, including diffusers-format files whose
attention is split into q / k / v (they are fused into Klein's layers).</li>
</ul>
<h3>SDXL, Pony, Illustrious, NoobAI</h3>
<ul>
<li><b>One file:</b> pick the model's <code>.safetensors</code> checkpoint. The VAE and text-encoder rows are
optional: leave them empty to use the ones inside the checkpoint, or point the VAE row at a fixed VAE.</li>
<li><b>Pick the right family for the checkpoint.</b> NoobAI comes as two kinds: <i>eps</i> and <i>v-pred</i>.
Training a v-pred checkpoint as eps (or the reverse) gives noise.</li>
<li>Captions are cut at <b>225 tokens</b>. Tag captions work well; consider <i>Shuffled tag variants</i> with
<i>Keep first N tags</i> for your trigger word (Caption augmentation, off by default).</li>
<li>Batch size above 1 works for these models. Hard-image tracking needs batch size 1.</li>
<li>Three optional extras, all off by default: <b>Min-SNR gamma</b>, <b>Noise offset</b>, and <b>Also train
convolutions (LoCon)</b>.</li>
<li>The first run downloads the small CLIP tokenizer files. To stay offline, put them in a
<code>clip_tokenizer</code> folder next to the checkpoint.</li>
<li><b>Presets:</b> <i>High capacity</i> (rank 32, alpha 16) and <i>Balanced</i> (rank 16, alpha 8) are a recipe
measured upstream, not by TagScribeR: a flat 5e-5 learning rate, fused AdamW, Adaptive LR off (it reacts to SDXL's noisy per-epoch loss). They were
measured on a photographic checkpoint, so on Pony, Illustrious and NoobAI they are a good start rather than a
measurement. The three <i>Community</i> presets (fast, cosine, style) are community starting points.</li>
<li><b>Base precision:</b> bf16, INT8 or 4-bit. About 10.3 GB peak (upstream figure, not measured by TagScribeR) on bf16, 8.2 GB on INT8 and 7.4 GB
on 4-bit (including the 1024 px preview), so SDXL trains on 8 GB cards with 4-bit.</li>
<li><b>LoKR</b> works too; its file uses the same layer names ComfyUI reads for SDXL.</li>
<li>SDXL 1.0 previews use <b>DPM++ 2M SDE Karras</b> (in ComfyUI: dpmpp_2m_sde / karras); the other
variants keep their own sampler.</li>
</ul>
<h3>Anima</h3>
<ul>
<li><b>Files:</b> the Anima DiT, the Qwen-Image VAE and the Qwen3-0.6B text encoder (all from the Anima repo).</li>
<li>Anima is tag-trained: lowercase tags with spaces, artists as <code>@name</code>. The preview negative prompt is
tag-style for the same reason.</li>
<li>The model's author recommends a light touch. <b>Presets:</b> <i>Character</i> (rank 16, 1e-4, 50 epochs),
<i>Style</i> (rank 16, 5e-5) and <i>Model card recipe</i> (the model card's rank 32 at 2e-5) come from the upstream project; the three
<i>Community</i> presets (fast, standard, style) are TagScribeR's earlier ones. All are community starting points, not
measured recipes.</li>
<li>The <b>Qwen3 text encoder runs in full precision</b> (float32) while captions are cached: in bf16
its output drifts noticeably. It is small, so this costs little. Captions cached by an older version are re-cached
once.</li>
<li>The T5 tokenizer comes from the Anima repo (one small file, downloaded once). To stay offline, put
<code>tokenizer.json</code> in a <code>t5_tokenizer</code> folder next to the text-encoder file.</li>
<li><b>Base precision:</b> bf16, INT8 or 4-bit. Upstream figures (not measured by TagScribeR): about 8.9 GB peak on bf16, 7.3 GB on INT8 and 6.6 GB
on 4-bit.</li>
<li><b>LoKR</b> works too. LoRAs made with AI-Toolkit load as frozen adapters.</li>
<li><b>Previews:</b> with the optional <b>Anima Turbo LoRA</b> file set, previews render on it by default: strength 1,
10 steps, CFG 1 (the default since October). Without it, or with Samples Turbo strength 0, they render on the
plain model at 20 steps and CFG 4.5 with the model card's negative prompt.</li>
</ul>
"""),
    "settings": ("Settings", "Settings", """
<h2>Settings</h2>
<ul>
<li><b>Appearance</b> — theme and interface scale. Settings → Appearance → Theme changes the look at once, without a
restart. <i>TagScribeR dark</i> and <i>TagScribeR light</i> are the new themes with larger, clearer controls;
<i>dark_teal</i> is the classic look and stays the default, and the other classic themes stay available.</li>
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
<li><b>A model needs custom code.</b> Tick “Allow this model to run its own code” in Model options (it is enabled
only for models that need it) — only for trusted sources.</li>
</ul>
"""),
}

# Which topic F1 opens for each main tab index.
TAB_TOPICS = ["gallery", "caption", "editor", "datasets", "metadata", "train", "settings"]

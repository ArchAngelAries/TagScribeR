# TagScribeR UI/UX research and proposals

Status: research and proposals only. Nothing here is implemented. It is written from the README's annotated
screenshots, `docs/ARCHITECTURE.md`, a skim of `main.py` and `tabs/*.py`, a local checkout of Fizgig's docs, and the
public sources listed at the end. Several of the web sources are practitioner blogs rather than primary standards; they
are marked as such. Nothing was run on real hardware and no real datasets were used. All "current state" statements
come from the screenshots and source, not from a live session.

## 1. Summary

TagScribeR already has the hard parts of a modern pro tool: a shared workspace, a fast virtualised grid, a Ctrl+K
palette, a searchable context-aware Help Center, tooltips everywhere, a simple view versus "Show all settings", a live
loss chart with Adaptive LR markers, a samples gallery, a run queue and a VRAM/RAM strip with a peak marker. What it
lacks is a **visual system**. Colour, spacing, type and button treatment are decided widget by widget, with raw hex
values scattered through `main.py`, `tabs/caption.py`, `tabs/datasets.py`, `tabs/metadata.py`, `tabs/review.py` and
others, on top of the stock `qt_material` dark teal theme. The result is functional but reads as "a themed form
toolkit" rather than "a studio".

The five biggest opportunities, in order of value for effort:

1. **One design-token module** (colours, type scale, spacing, radii) feeding one generated stylesheet. This removes the
   scattered hex values and unlocks everything else. Quick win, high leverage.
2. **A calmer button hierarchy.** Nearly every button is an outlined teal "SHOUTING CAPS" button (Browse, Get, Load,
   Save, Delete, Import, Problem Images ...), so the one primary action (Start Training, Save, Scan Folder) does not stand
   out and destructive actions (Delete, Stop) look identical to harmless ones.
3. **First-run and empty states.** The app opens on an empty Gallery; a beginner is not led to "open a folder, caption,
   check health, train". Fizgig ships a welcome tour and numbered workflow tabs, so TagScribeR should at least match it.
4. **A guided Train tab for beginners** (a pre-flight checklist and a "ready to train" card) and **a run dashboard
   that is readable at a glance** (large step/ETA/loss tiles, a pinned run strip visible from every tab, a desktop
   notification when a run ends).
5. **Comparison views** (Compare/Survey for the Gallery, before/after slider in the Editor and Review window, a
   per-epoch sample scrubber in Train). These are the signature features of Lightroom, Fizgig's run visualiser and
   LoRA Royale, and they are precisely what dataset curation and LoRA evaluation need.

Almost all of it is achievable in PySide6 with QSS, `QPainter` custom widgets, `QPropertyAnimation` and the existing
`qtawesome` icons. Section 9 lists the few things that would need a different toolkit (QML or a web view) and
recommends not doing them.

## 2. Current strengths (keep these)

- **One shared dataset workspace.** The folder you open is open everywhere, with its own selection and filter per tab.
  This is the "library" mental model of Lightroom and Eagle, and it is better than the kohya_ss / OneTrainer pattern of
  retyping a path per tool.
- **A filter language in the toolbar** (`tag:scenery res:>=800`, `missing:caption`). Search-as-filter is what taggui and
  Eagle users expect, and saved filters per folder are a power-user touch.
- **Data-safety affordances are visible**: Undo/Redo and Save in the toolbar, "NO CAPTION" badges, "unsaved" dot,
  Recycle Bin deletes, "Save edited copies (originals untouched)" as the default in the Editor.
- **Progressive disclosure on the Train tab.** "Show all settings", collapsible groups, presets that only change what
  they contain, "Native resolution about 1 MP" hints. This matches the guidance that two levels is the practical maximum
  ([NN/g](https://www.nngroup.com/articles/progressive-disclosure/)). Keep it at two levels.
- **A good live-run panel**: status line with epoch, step, s/it, ETA and loss; loss and LR on one chart with Adaptive
  LR decision arrows; samples newest first; a console that is also saved as `run.log`; Pause that saves a resumable
  state.
- **Memory strip under every tab** with a peak marker. Few trainers show this; it is a genuinely useful "premium"
  detail for people on 16-24 GB cards.
- **Help is built in** (F1 per tab, `?` buttons, Ctrl+K lists help topics) and every screenshot in the README carries
  plain-language annotations, which is the right voice for the app.
- **Honest, plain copy.** Tooltips and hints explain consequences ("Cleaning is lossless: pixels are never re-encoded").
  Keep this tone; it is a differentiator.

## 3. What polished tools do well (research digest)

Evidence quality is noted. "Vendor" means the tool's own documentation; "blog" means practitioner writing.

**Task-shaped navigation with a visible workflow order.** DaVinci Resolve splits the app into task pages (Media, Cut,
Edit, Fusion, Color, Fairlight, Deliver) so each keeps only the tools for that stage, and the left-to-right order is the
intended flow ([third-party guide](https://www.tourboxtech.com/en/news/davinci-resolve-pages-explained.html), blog).
Fizgig numbers its core tabs 1 to 5 (Start, Image Prep, Captions, Samples, Training) and puts unnumbered tools after
them ([Fizgig README](https://github.com/shootthesound/Fizgig)). Blender uses workspaces as saved task layouts
([Blender manual](https://docs.blender.org/manual/en/latest/interface/index.html)). Lesson: the order of the sidebar
should teach the pipeline.

**Culling and comparison as first-class views.** Lightroom Classic has Grid, Loupe, Compare (C, two images, can zoom in
sync) and Survey (N, several candidates, remove the weak ones); P/X flags, 1-5 ratings and "auto advance" make a first
pass fast ([Lightroom Killer Tips](https://lightroomkillertips.com/grid-loupe-compare-survey-view-one-use/),
[Picture Correct](https://www.picturecorrect.com/lightroom-culling-mastery-tips-for-choosing-your-best-photos/), blogs;
Adobe's own help page was not reachable during this research). Lesson: dataset curation is culling. Near-duplicate
groups, blurry images and "which of these two captions is better" all want Compare/Survey.

**Keyboard-first design with a palette that complements, not replaces, menus.** Command palettes should be one
consistent shortcut, fuzzy, grouped, show recent commands and shortcuts beside commands, always show an empty state,
and never be the only route to an action ([summary of palette guidance](https://ux.hubtel.com/design-system/components/command-palette/),
[another](https://www.techinterview.org/post/3233475212/build-command-palette-cmd-k/); blogs, no authoritative source
found). taggui is a good domain example: Ctrl+Up/Down between images, Ctrl+J to jump to the first untagged image,
Alt+letter to focus each pane, F2 rename, Ctrl+R find/replace, and a filter syntax with AND/OR/NOT and wildcards
([taggui README](https://github.com/jhc13/taggui)).

**Progressive disclosure, with limits.** NN/g: defer advanced options to a secondary level, keep to two levels, make
the path to the second level clearly labelled, and make sure frequently needed features are on the first screen;
wizards (staged disclosure) fit steps that do not depend on each other and break down when users must jump back and
forth ([NN/g](https://www.nngroup.com/articles/progressive-disclosure/)).

**Empty states teach.** NN/g's three guidelines for empty states in complex apps: communicate system status (do not show
"No records" while loading), teach how to populate the area, and offer a direct button to the key task, optionally with
demo data ([NN/g](https://www.nngroup.com/articles/empty-state-interface-design/)).

**Long-running jobs.** The classic limits: 0.1 s feels instant, 1 s keeps flow, 10 s is the attention limit; beyond
that show percent-done and a clear way to interrupt, and expect users to leave and come back
([NN/g](https://www.nngroup.com/articles/response-times-3-important-limits/)). Training takes hours, so the interface
must be glanceable from another room, from another tab, and after the window was minimised. Fizgig answers this with a
sidebar that shows what is training and queued on every tab, a phone view (QR code) with progress, loss and time
remaining, a run visualiser to scrub through epochs, and a live sample gallery that scores likeness
([Fizgig README](https://github.com/shootthesound/Fizgig), [Fizgig web UI doc](https://github.com/shootthesound/Fizgig/blob/main/docs/WEB_UI.md)).
OneTrainer offers TensorBoard integration, sampling without leaving the app, and periodic backups to resume
([OneTrainer README](https://github.com/Nerogar/OneTrainer)).

**Dark theme systems.** Practitioner guidance converges: avoid pure black as the base (halation, no visible shadows),
use a dark grey base with progressively lighter surfaces for elevation, desaturate accent colours, and check contrast per
element ([SEP](https://sep.com/blog/6-tips-for-designing-a-dark-mode-version-of-your-app/),
[Uxcel](https://uxcel.com/blog/12-principles-of-dark-mode-design-627),
[Accessibility Checker](https://www.accessibilitychecker.org/blog/dark-mode-accessibility/); blogs). The normative
anchor is WCAG 2.2: 4.5:1 for body text, and 3:1 for the visual information needed to identify a control and its state,
and for focus indicators ([WCAG 1.4.11](https://www.w3.org/WAI/WCAG22/Understanding/non-text-contrast.html)). Image
tools add a twist: darktable's default theme is a deliberate mid grey because a dark surround makes images look
clearer and more saturated than they are, and the manual recommends a neutral mid-grey surround for judging tone
(darktable manual and [community discussion](https://discuss.pixls.us/t/neutral-theme-for-darktable-and-gimp/9504);
users also report eye fatigue, so it is a choice not a rule). Lesson: keep the chrome neutral (no strong tint) around
images, and offer a second surround option for people judging photos.

**What "premium" feels like.** Not a single source, but a consistent pattern across the tools above: a restrained
palette with one accent; consistent radii and spacing; quiet secondary actions and one obvious primary action; fast
(under 100 ms) feedback and subtle motion that confirms a state change; thoughtful empty, loading and error states;
density that can be changed; keyboard shortcuts visible in tooltips and menus; nothing jumps around when data loads.

## 4. Findings per screen

Each finding names the element, the user feeling, and who benefits. "B" = beginner, "P" = power user.

### 4.1 Global chrome (`main.py`, `tabs/memory_bar.py`, `tabs/common.py`)

- **Sidebar.** Seven equal rows, 15 px font, solid teal selection, divider lines between every row. The numbering
  Ctrl+1..7 exists but is not shown. The order (Gallery, Auto Caption, Image Editor, Datasets, Metadata, Train,
  Settings) is not the pipeline order and mixes dataset tools with a final-stage Train tab. B: no sense of "what do I do
  first?". P: does not show shortcuts. Suggest: group the rows (WORKSPACE: Gallery, Health-in-Gallery; PREPARE:
  Caption, Editor, Metadata; OUTPUT: Datasets/Export, Train; then Settings and Help pinned at the bottom), show a
  subtle numeral or shortcut on hover, use a left accent bar plus a tinted background for the selected row instead of
  a full teal block (the block is the loudest thing on every screen, and the same teal is also the primary-action
  colour, so selected navigation and call-to-action compete).
- **Folder context is invisible in the chrome.** The dataset path appears only in a clipped status line in the Gallery
  ("D:\Datasets\demo..."). A persistent header strip with folder name, image count, unsaved count and missing-caption
  count (click to filter) would make the "shared workspace" idea visible on every tab.
- **Memory strip.** Good idea, but the VRAM and RAM bars are very wide and partly empty, the gradient fill (green to
  red) and the white peak marker are clear, but the text is small and low contrast, and the "HIDE STATS" label is
  in teal caps at the far right. Suggest a compact two-pill version with a tooltip that explains what is using VRAM
  ("loaded caption model", "training run") and an "Free VRAM" shortcut.
- **No ambient job indicator.** If a caption batch or training run is going on, nothing outside that tab says so. Add a
  small status chip in the sidebar footer or the strip ("Training - epoch 9/30 - ETA 21:27") that jumps to the Train
  tab on click. Fizgig does the equivalent on every tab.
- **Help button.** A grey box at the bottom of the sidebar with an emoji in code (`❓ Help / Manual`). Fine, but the
  Help Center is a feature to be proud of: add "Search or run a command  Ctrl+K" as a visible search field at the top of
  the window so the palette is discoverable.
- **Inconsistent icon/emoji use.** `main.py` builds the sidebar with emoji text, while the rest of the app uses
  `qtawesome` icons (and offscreen screenshots show icons, so there is a fallback). Pick one family (qtawesome only)
  so Windows emoji rendering and DPI scaling cannot change the look.
- **No toast layer.** Success/failure messages are modal `QMessageBox` or status text. A small non-blocking toast
  ("Saved 12 captions", with Undo) is the modern expectation and is cheap in Qt.

### 4.2 Gallery (`tabs/gallery.py`, `tabs/workspace/*`)

- **Toolbar density.** Two rows with Open Folder, history, Subfolders, filter, star, kebab, sort, direction, Undo, Redo,
  Save, Copy to Collection, Captions toggle, Size slider. All are the same outlined teal style, so the eye has no
  starting point. Suggest: Open Folder as a quiet button; Save as the only filled primary (and filled only when there
  are unsaved changes); Undo/Redo as icon-only with tooltips showing Ctrl+Z; view options (Captions, Size) grouped on
  the right as a "view" cluster.
- **Cards.** The thumbnail, file name and a three-line caption are legible, with resolution badge and "NO CAPTION"
  badge. Selected state is a teal outline with a faint teal fill; at 100% zoom it is subtle and identical to the
  "Similar/Blurry" badge colours in Health. Suggest: stronger selected state (2 px accent border plus check mark in the
  corner), consistent badge colours with a legend (red = problem, amber = warning, blue/grey = info), hover state with
  quick actions (caption icon, trash icon), and a fixed height so the grid does not reflow when captions are toggled.
- **Selection model.** Shows "1 selected" only in the status text. Add a selection bar that appears when more than one
  image is selected (count, "Add tag...", "Caption selected", "Copy to collection", "Clear") as Eagle and Lightroom
  do. This turns batch editing, a TagScribeR strength, into something discoverable. B and P.
- **Inspect panel.** Clear. The tag bubbles are a good idea. Issues: the caption box and the bubbles show the same
  data stacked (a lot of vertical space); the "9 tags - 11 words - ~22 CLIP tokens" counter is tiny (7B font at
  grey on grey) though it is the most useful number for a training author. Suggest making the token count a small
  meter that turns amber near 75 tokens (the CLIP limit it already computes), and a toggle to hide either view.
- **Right panel tabs** (INSPECT, BATCH, TAGS, HEALTH) are small caps text with a thin underline; the selected tab is
  teal text only. They are the main sub-navigation of the most-used screen and deserve larger hit areas and an icon.
- **Empty state.** With no folder open there is nothing to teach. Add a centred card: "Open a folder of images to
  begin" with Open Folder, "Open recent", "Try with demo images" (generated demo data already exists for
  screenshots), and a three-step strip "Caption - Check health - Train". This is NN/g's empty-state recipe applied.
- **Compare/Survey.** Missing and very valuable here (see 5.3). A Health group ("Group 1: alpine_dawn.jpg,
  alpine_dawn_crop.jpg") should open a Compare view with keep/reject keys, not just select the images.
- **Keyboard.** Arrow-key navigation and Del exist; missing are rating/flag keys, "next uncaptioned" (taggui's Ctrl+J
  is a loved feature), "focus filter" (Ctrl+F / "/"), and Space for a large preview (Loupe). P.

### 4.3 Caption (`tabs/caption.py`, `tabs/review.py`)

- **Right panel is a long scroll of form** (Local model / API, model options, instructions, preset, system prompt,
  subject and tag hints ...). The primary action (run) is not visible without scrolling in the screenshot. Suggest a
  sticky footer with "Caption 12 selected images" (a big primary button), a model chip and an estimate ("about 4 s
  per image, model loaded") so the action is always one click away.
- **Stateful run button**: it switches between red "run" and stop colours with inline styles (`#d63031`, `#ff4757`).
  Red for "Run" reads as "danger". Use the accent colour for run, red only for stop.
- **Progress.** Not visible in the screenshots; per-card "REVIEW" badges and a review window exist. Add a per-card
  spinner/progress overlay while a card is being captioned and a job chip with "7 / 40 - 3.2 s/image - ETA 1:45"
  that survives tab switches. Long jobs need percent-done and a clear interrupt (Esc exists; show it on the button:
  "Stop (Esc)").
- **Review window.** The word-level diff is a standout feature. Improve with: a before/after split with synchronised
  scroll, `A` accept / `R` reject / `Space` next shown as on-screen key caps (Lightroom-style cull), accept-all with
  confidence filter, and a clear counter ("5 of 40 reviewed").
- **Presets** look like a combo plus two icon buttons, one of which is only a "..." glyph. Tooltips exist; labels do not.
  Beginner: they do not know what the diskette icon does. Use a text button "Save as preset".
- **Subject/trigger word** is a strong feature but buried in a collapsible section; it is the single most important
  setting for LoRA authors. Pull "Subject / trigger word" up as a labelled field near the top.

### 4.4 Editor (`tabs/editor.py`)

- **Before/after is side by side in a small preview** and, in the screenshot, the right half is cropped by the panel
  edge. A large central canvas with a draggable before/after divider (as in every photo editor) would be more
  satisfying and more informative than two small thumbnails; the thumbnail grid on the left occupies the most space
  but is not the object being edited.
- **Operations are a long scroll of boxes** (Output, Rotate/Flip, Resize, Crop, Convert ...) with "APPLY RESIZE" etc.
  Each operation has its own apply button, so users can apply two operations in a row and forget which is pending.
  Suggest a visible "pending operations" stack with a single Apply (recipe) button and "Reset", like a mini history.
- **Output mode** ("Save edited copies" vs "Overwrite originals (asks first)") is the most important safety setting
  and is presented as two radio buttons mid-panel. Keep default; show it in a persistent pill at the top of the panel.
  Destructive choice gets a red outline.
- **Crop has no on-image handles** (preview shows "Crop to 3:2 (Center)" as text). On-canvas crop handles with aspect
  presets are the expected modern interaction; feasible with a `QGraphicsView` overlay.
- **Zoom and pan** (scroll to zoom, space-drag pan, 100%/fit keys) are missing and are expected in an image tool.

### 4.5 Metadata (`tabs/metadata.py`)

- The **audit list** is the most visual-design-ready piece: colour-coded counts ("GPS location: 1 of 16"). It uses
  hollow radio dots and tiny text. Make it a card list with a severity icon, a coloured left rule, a count chip and a
  "Show these images" link that filters the Gallery. Red/amber/grey is already used; add icons so it does not rely on
  colour alone (accessibility).
- The big blue "CLEAN METADATA..." button uses a different accent (`#0984e3`) than the rest of the app's teal
  (`#00b894`) and purple (`#6c5ce7`) on other tabs. Metadata, Datasets and Review each introduced their own primary
  colour. One accent plus semantic colours (danger, warning, success, info) would calm the whole app.
- **Before/after summary.** After cleaning, show a result card ("Removed GPS from 1 image, serial numbers from 1,
  prompts from 3. Pixels unchanged. Undo not available - backups in ...") rather than only a dialog.
- Tab bar (INSPECT / PRIVACY / AUTHORSHIP / PROMPTS) is clear and good; keep, with icons.

### 4.6 Health (a tab of the Gallery panel, `core/health.py` UI in `tabs/workspace/panels.py`)

- Great content, plain presentation: a tree of groups plus a bucket table. Scan is on the CPU and cached; show a
  progress bar with "412 / 2,000 images" and the ability to keep working.
- **Add a summary header**: a dataset score card ("Looks good: 14 of 16 ok - 2 blurry - 3 near-duplicate groups - 1 low
  resolution"), each chip clickable to filter. The slider "Near-duplicates: balanced" needs end labels (strict - loose).
- **Bucket table to a chart.** A small horizontal bar chart of images per bucket (a `QPainter` widget) with a warning
  for buckets that have fewer images than the batch size is much more informative than a table, and speaks directly to
  the failure mode training authors fear (a bucket with one image). Link the warning to the Train tab's batch size.
- **Duplicate groups should open Compare** with Keep/Remove (see 5.3) and an explicit "Keep best (largest, sharpest)"
  suggestion with reasons.

### 4.7 Train (`tabs/train.py`, `tabs/train_dialogs.py`)

Setup (left):

- **Beginner path.** The simple view is good, but a first-time user still faces Family, Preset, Load/Save/Delete/Import/
  Last Run, a checkbox, then a model-files block with Browse and Get for four paths. Add a **Ready check** card at the
  top ("Dataset: 16 images, 2 missing captions - Model files: 2 of 3 set - Memory: fits at 20 GB - Estimated time
  about 35 min") with green/amber/red ticks and "Fix" links, and the primary Start button in that card disabled until
  required items pass, with a reason on hover. Fizgig's flow of "pick a preset, press start" works because it asks for
  little; TagScribeR can make "what is still missing" explicit instead of discovered on error.
- **One accent button per card.** "LOAD SAVE... DELETE IMPORT... LAST RUN" are five identical outlined teal buttons in
  one row. Make Load/Save/Import/Last run secondary icon+text, Delete a red icon-only button inside an overflow menu.
- **Group headers.** "Output", "Training Parameters", "Dataset", "Samples" are heavy slate bars with small grey bold
  text and tiny arrows; they look like disabled controls. Use a lighter section header (title plus one-line summary of
  what is inside, such as "Dataset - 0.25 MP, batch 1, bucketed"), so closed groups still convey the current values.
  Power users get a read-only summary of the whole config without opening anything.
- **Setting rows.** The labels sit left of the control with no alignment grid ("Epochs: 30" is a bordered group box
  holding one spin box). A proper two-column form with consistent label width, units on the right ("MP", "steps"),
  a "reset to preset" arrow beside any value that differs from the preset (Figma-style "modified" dot) and a search
  box over "Show all settings" would make the long list navigable. Search in settings is the single biggest quality of
  life gain for a 100-setting panel and it already has a palette fuzzy matcher to reuse.
- **Diff from preset.** Show how many settings differ from the loaded preset ("3 changed") with a click to list them.
  Power users can see what they changed; beginners can undo a mistake.
- **Model files.** Browse and Get buttons per row are repeated four times; use a status icon (found, missing, wrong
  type) and one "Get" link that opens the download page. Paths are already remembered; show size and a green check.

Run (right):

- **Status text is small.** "Training - epoch 9/30 - step 171/600 - 3.02 s/it - ETA 21:27 - loss 0.1259" in tiny type
  above a flat progress bar. Make it a row of large tiles (Epoch 9/30, Step 171/600, ETA 21:27, Loss 0.126, s/it) and a
  thicker progress bar with percentage; the same data, readable from two metres away.
- **Chart.** Per-epoch loss and LR on one canvas with arrows is informative but cramped, with the right-axis label
  clipped ("4.0e-04"). Needs: a legend, tooltip on hover with exact values, an option for per-step loss with a smoothed
  line (Fizgig logs per-image loss and EMA; TensorBoard users expect smoothing), log-scale toggle, a "current" marker,
  and axis label padding. Feasible with `QPainter` or an embedded `pyqtgraph`.
- **Samples scrubber.** The samples tab is a thumbnail grid. Add the thing people actually do with previews: compare
  the same prompt across epochs. A slider along the epoch axis with a large preview and a crossfade (Fizgig's LoRA
  Royale does crossfade per epoch, and its run visualiser lets you scrub through epochs) lets users pick the best epoch
  visually. Add "Use this epoch" which reveals the matching checkpoint file.
- **Console.** Fine as a console; make it collapsible, with a "warnings and errors only" filter and copy-to-clipboard.
  Raw lines like "[adaptive_lr] epoch 6: ..." should also appear as human text in the Adaptive LR arrow tooltips.
- **Button states.** START/PAUSE/STOP are three equally weighted teal outlines. Start should be filled; Stop should be
  red-outlined and ask once if progress would be lost; Pause/Resume should toggle in place.
- **Run finished.** There is no description of a post-run state. Offer a summary card (final loss, time, peak VRAM,
  output folder, "Open in Gallery", "Use in Samples", "Queue another with same settings") and a desktop notification
  when the window is not focused. A run that takes hours must tell the user when it is done.
- **Queue tab.** Good that it exists. Show each queued run as a card with its preset, family and estimated time, with
  drag to reorder.

### 4.8 Help (`tabs/help.py`, `tabs/help_content.py`)

- The Help Center is searchable and context-aware. Its look (a plain `QTextBrowser`, inline CSS, teal h2) is
  document-like rather than app-like. Add: a left topic tree with icons and a "Getting started" pinned top, "Was this
  helpful?" is overkill, but "Related topics" and "Copy shortcut list" are cheap; render hotkeys as key-caps
  (`<kbd>`-style) so they look like keys; embed the existing annotated screenshots in-line (the README annotations are
  ideal Help Center content: the help can show the real screen with the callouts).
- **Contextual help `?`** buttons are 26 px circles, good. Add "Show me" for the top five tasks: an overlay tour that
  highlights the actual control (see section 7, onboarding).
- A **shortcut cheat sheet** overlay (press `?` or `Ctrl+/`) is a standard power-user expectation, and the app has
  enough shortcuts (Ctrl+K, Ctrl+1..7, Ctrl+Enter, Del, Esc ...) to justify one page.

### 4.9 Settings (`tabs/settings.py`)

- **Theme** is a list of `qt_material` names (dark_teal, dark_amber, light_blue ...) in a combo. Presenting them as
  palette swatches with a live preview would make this tab a pleasure rather than a dropdown. If tokens are adopted
  (section 6), "theme" becomes a handful of accent colours plus a surround option (neutral grey, near-black) and a
  density option (comfortable, compact).
- **Interface scale** requires a restart. Where possible apply live (rebuild the stylesheet with scaled `rem`-like
  tokens); otherwise say so next to the control (it does) and offer a "Restart now" button.
- **Settings is a vertical wall of GroupBoxes** (Appearance, AI defaults, model folders, quick tags, diagnostics).
  A left list of sections with search (the same pattern as the palette) will scale better as more settings are added.
- Diagnostics (GPU detection, versions, log folder, "copy report") is the first thing a support request needs; make
  it one click, and show GPU/driver/torch in a card with a status icon. Doubly important because only one platform has
  been tested (see CLAUDE.md rules 8 and 9).

## 5. Cross-cutting themes

### 5.1 Information density versus clarity

The app currently goes for density with uniform weight: every control the same size and colour. Premium tools keep
density but create hierarchy with *weight, not size*: one filled primary action, secondary outline or ghost buttons,
tertiary text buttons, icon-only for repeated actions with tooltips. Provide a "Compact / Comfortable" density
setting (Qt: a token for control height and spacing, 28 px and 24 px) so power users with 4K monitors can fit more.

### 5.2 Progressive disclosure for beginners and power users

Keep exactly two levels (simple, "Show all settings") per NN/g. Improve the wording and the affordance: replace the
checkbox with a segmented control "Simple | Advanced", show a count ("Advanced: 87 more settings"), keep the search box
across both so beginners can find a setting they heard about without switching, and make the preset decide the default
while indicating what a preset changes. Do not add a third "expert" level; add a search and a "changed from preset"
filter instead.

### 5.3 Comparison views

The app has one comparison (the caption diff in Review) and one small before/after (Editor). Recommended comparison
surfaces, in decreasing value:

1. **Gallery Compare/Survey** (C/N like Lightroom): two or four images side by side with synchronised zoom/pan, keys
   to keep or reject, used from Health groups. Straight dataset curation value.
2. **Train samples scrubber and A/B**: same seed across epochs and across runs (slider + crossfade + side by side).
3. **Editor and Review before/after slider**: a vertical divider the user can drag.
4. **Caption A/B**: two models' captions for the same image, pick one (a popular benchmark workflow for dataset
   builders choosing a captioner).

### 5.4 Long-running job UX

Required across Caption, Health, Metadata clean, Export and Train: a consistent **Job** component (title, percent,
step counter, ETA, current item, Stop). Place a compact version in the sidebar footer so jobs are visible from anywhere,
and fire a native notification (`QSystemTrayIcon.showMessage`) when a job finishes while the window is not active,
with the outcome ("Training finished - 30 epochs, final loss 0.112"). Provide a **Tray/minimised mode** for long runs.
Fizgig's phone view shows that glanceability matters to this audience; a local "web status page" is out of scope, but
a notification plus a sidebar chip captures most of the value.

### 5.5 Drag and drop

Support dropping a folder anywhere to open it (with a full-window overlay "Drop folder to open"), dropping images onto
a collection in Datasets, dropping `.safetensors` onto a model-file row in Train (with type validation), and dropping
a preset file onto the Train tab to import. Tag bubbles already reorder by drag. Each drop target should highlight
(a dashed accent border and label) before drop.

### 5.6 Undo

Undo/redo exist for caption changes. Extend the mental model: Editor operations and Metadata cleaning cannot be undone
as they write files; say so on the button ("Clean metadata (a backup copy is made)") or add a "Restore from backup"
history. Show an undo toast after batch operations.

### 5.7 Accessibility

- Contrast: the small grey captions and tab labels (grey on `#2b2b2b`) look below 4.5:1; hints, token counters and badge
  text should reach 4.5:1, and control borders 3:1
  ([WCAG 1.4.11](https://www.w3.org/WAI/WCAG22/Understanding/non-text-contrast.html)).
- Do not rely on colour alone: the audit severities and health badges should carry an icon or label (already partly
  true for badges).
- Focus visibility: provide a consistent, clearly visible focus ring in the stylesheet (Qt's default is easily lost with
  `qt_material`); keyboard order must be logical, and every action reachable by palette and menu.
- Scale: the 100 to 200% scale option is a strength; also test 125%/150% Windows display scaling with fonts in `pt`.
- Reduced motion: expose a "Reduce animations" switch and honour the OS setting where Qt exposes it.
- Screen readers: set `setAccessibleName` / `setAccessibleDescription` for icon-only buttons (which tooltips alone do not
  cover for assistive technology).

## 6. Design-system proposal (fits Qt and QSS)

Goal: one source of truth, generated into one stylesheet, replacing scattered inline `setStyleSheet` calls. Implement as
`tabs/theme.py` (tokens as constants, a function `build_qss(tokens)`, `apply_theme(name)`), keeping `qt_material` only
as a base or dropping it once the generated QSS covers the widgets in use. Dynamic state (modified, error, primary) via
Qt dynamic properties (`btn.setProperty("variant", "primary")`, then `style().unpolish/polish`), not per-widget strings.

### 6.1 Colour tokens (dark default)

Neutral stack with elevation by lightness, one accent, four semantic colours. The values below are proposals to tune on
a real display, not final; run each text/background pair through a contrast checker.

| Token | Value | Use |
|---|---|---|
| `bg-app` | `#16181b` | window background, behind panels |
| `bg-surface` | `#1d2024` | panels, sidebar |
| `bg-raised` | `#252a2f` | cards, inputs, popovers |
| `bg-hover` | `#2d333a` | hover rows, secondary buttons |
| `bg-image-surround` | `#1a1a1a` (neutral) or `#2a2a2a` | behind images and previews, no tint |
| `border-subtle` | `#2f353c` | dividers |
| `border-strong` | `#4a525b` | input borders (3:1 vs surface is the target) |
| `text-primary` | `#e6e8eb` | body |
| `text-secondary` | `#a7aeb6` | labels, hints (check 4.5:1) |
| `text-disabled` | `#6b727a` | disabled only |
| `accent` | `#19b394` (current teal family) | primary buttons, selection, focus |
| `accent-hover` / `accent-pressed` | lighter / darker steps of the accent | states |
| `accent-subtle` | accent at 14% on surface | selected rows and cards |
| `success` / `warning` / `danger` / `info` | `#3fb950` / `#e3b341` / `#f0605a` / `#58a6ff` | semantic only |
| `focus-ring` | accent or `#7cc4ff`, 2 px | keyboard focus |

Rules: accent only for the primary action, selection and focus (never for everything); semantic colours never as
decoration; remove the per-tab purples and blues (`#6c5ce7`, `#0984e3`). Light theme: same token names with inverted
elevation and the same accent at darker steps; mid-grey "image surround" option for tone-judging workflows (darktable's
reasoning). Offer a "neutral" accent (white/grey) for people who find saturated teal tiring over long sessions.

### 6.2 Typography

Use a system UI face (Segoe UI Variable on Windows 11, system on Linux) with a monospace for the console and tags
(Cascadia Mono / JetBrains Mono if bundled). Fixed scale in points (so DPI scaling stays correct):

| Role | Size | Weight | Notes |
|---|---|---|---|
| Caption/hint | 9 pt | regular | `text-secondary`, min contrast 4.5:1 |
| Body / controls | 10 pt | regular | default |
| Label (form) | 10 pt | medium | `text-secondary` |
| Section title | 11 pt | semibold | sentence case, no letter-spaced caps |
| Panel title | 13 pt | semibold | |
| Big number (run tiles) | 20 pt | semibold, tabular figures | |

Drop all-caps on buttons (currently `BROWSE`, `START TRAINING`): sentence case ("Start training") is calmer and
easier to read; keep caps only for tiny category labels.

### 6.3 Spacing, radius, elevation

4 px base unit: 4, 8, 12, 16, 24, 32. Panel padding 16; control height 28 (compact 24, comfortable 32); radius 4 for
inputs, 8 for cards, 999 for chips and badges. Elevation: lighter surface plus a 1 px `border-subtle`; avoid drop
shadows except for popovers and the palette (Qt supports `QGraphicsDropShadowEffect` but it is costly on large widgets;
use only on small floating ones). Use consistent margins from the token table instead of ad-hoc `padding: 6px / 8px / 15px`.

### 6.4 Iconography

Keep `qtawesome` (already used by `tabs/icons.py`), but standardise on **one** family (e.g. Phosphor or Material
Design Icons through qtawesome) at 16 px in buttons and 18 px in the sidebar, with `text-secondary` colour at rest and
`text-primary`/accent on hover/selected. Remove emoji from `main.py`. Icon-only buttons need tooltips with shortcuts
(`Undo (Ctrl+Z)`) and an accessible name. Icons for severity (info, warning, error) next to every coloured count.

### 6.5 Components to standardise (each a small class in `tabs/widgets_ui.py` or `core/widgets.py`)

- `Button(variant="primary|secondary|ghost|danger|icon")`
- `Card` (title, optional status icon, collapsible, one-line summary when collapsed)
- `Chip` / `Badge` (severity colours with icon)
- `Toast` and `Banner` (non-blocking messages, undo action)
- `EmptyState` (icon, headline, one sentence, primary button, secondary link)
- `JobCard` (title, percent, ETA, current item, Stop)
- `KeyCap` (for shortcuts in tooltips, Help, Review)
- `SegmentedControl` (Simple | Advanced, Selected | All shown)
- `SettingRow` (label, control, unit, "modified" dot, reset arrow, help `?`)

### 6.6 Motion and feedback

Short, purposeful, off by default only for reduced-motion. Under 120 ms for hover and press, 150 to 200 ms for panel
expand/collapse, toast fade, tab underline slide, card selection scale; none over 250 ms. Implement with
`QPropertyAnimation` on `maximumHeight`, `windowOpacity` or a custom property, and `QGraphicsOpacityEffect` for fades
(apply to small widgets only). Feedback rules: acknowledged within 100 ms (pressed state, disabled with spinner); work
longer than 1 s shows busy; longer than 10 s shows percent and a Stop (see NN/g limits above). Never animate the grid
while scrolling; keep scroll at native frame rate.

## 7. Onboarding, empty states and guided first runs

1. **First-launch welcome** (one dialog, skippable, not shown again): three cards "Browse and tag", "Caption with AI",
   "Train a LoRA", each with a button that opens the right tab, and a link "Open the demo dataset" (the generated demo
   data used for screenshots). Fizgig has a welcome tour and a video; this is the local equivalent.
2. **Checklist in the empty Gallery**: Open a folder - Check health - Caption missing - Fix tags - Train. Tick items as
   they complete; this doubles as a reminder when the user returns to a half-finished dataset.
3. **Coach marks** for the five key controls (filter box, Save, Caption run, Show all settings, Start training), as an
   opt-in "Show me" from Help. Qt implementation: a translucent top-level overlay widget that draws a hole around the
   target widget's `mapTo(window)` rect with a callout.
4. **Preflight on Train** (see 4.7) with actionable fixes. This is the most valuable onboarding surface because the
   first failed run costs the user time and trust.
5. **Empty states for every list**: Health ("No scan yet - Scan folder"), Queue ("Nothing queued - Queue this run"),
   Samples ("Previews appear after the first epoch"), Review ("Nothing to review"), Datasets ("No collections yet").
   Each states why it is empty and offers the next action.

## 8. Recommendations (tiered)

Effort: S = a day or less, M = a few days, L = a week or more. Files are the main ones touched. All items must keep
existing behaviour and follow the CLAUDE.md rules (help text, tooltips and docs in the same change; no narrowing of
features).

### Quick wins (S)

| # | Change | User benefit | Touches |
|---|---|---|---|
| Q1 | Button hierarchy: filled primary, outlined secondary, ghost, red danger; sentence case; Run no longer red | One obvious action per area, destructive actions stand apart; calmer screens | `tabs/settings.py` (APP_STYLE_TWEAKS), `tabs/caption.py`, `tabs/datasets.py`, `tabs/metadata.py`, `tabs/review.py`, `tabs/train.py` |
| Q2 | One accent colour: retire per-tab `#6c5ce7` and `#0984e3`; semantic colours only for status | Coherent identity; teal means "act here" | same files, `tabs/help.py` |
| Q3 | Selected nav row: accent bar plus tint instead of solid teal block; show Ctrl+N hints | Less visual shouting; teaches shortcuts | `main.py` |
| Q4 | Bigger Train run status: tiles for epoch, step, ETA, loss; thicker progress bar | Glanceable run state | `tabs/train.py` `_build_run_panel` |
| Q5 | Native notification when a run, caption batch or health scan finishes while unfocused | Users can leave for hours | `tabs/train.py`, `tabs/caption.py`, `main.py` (tray icon) |
| Q6 | Chart polish: legend, hover values, padding for right axis label | Readable and trustworthy loss chart | `tabs/train.py` (loss widget), `tabs/train_dialogs.py` |
| Q7 | Token counter as a meter (amber near the CLIP limit) | Prevents over-long captions | `tabs/gallery.py`, `tabs/workspace/panels.py` |
| Q8 | Key caps and a cheat-sheet overlay (`Ctrl+/`) | Power users find shortcuts | `main.py`, `tabs/help.py`, `tabs/help_content.py` |
| Q9 | Replace emoji sidebar labels with `qtawesome` icons; tooltips with shortcuts on icon-only buttons | Consistent rendering on all systems | `main.py`, `tabs/icons.py` |
| Q10 | Contrast pass on hints, counters and tab labels (4.5:1 text, 3:1 borders) | Readability, accessibility | `tabs/common.py` (`hint_label`), theme tweaks |
| Q11 | Empty states with a primary action (Gallery, Health, Queue, Samples) | Beginners know what to do first | `tabs/gallery.py`, `tabs/workspace/*`, `tabs/train.py` |
| Q12 | Visible Ctrl+K search field in the window top bar | Palette becomes discoverable | `main.py`, `tabs/palette.py` |

### Medium (M)

| # | Change | User benefit | Touches |
|---|---|---|---|
| M1 | `tabs/theme.py` token module and generated QSS; dynamic-property variants; remove inline styles | Consistent look, easy theming and density option | all `tabs/*.py`, `main.py` |
| M2 | Global job chip in the sidebar (and a persistent folder header strip) | Always know what is running and what is open | `main.py`, `tabs/workspace/jobs.py`, `tabs/workspace/context.py` |
| M3 | Train "Ready check" card with Fix links and a disabled Start with a reason | Fewer failed first runs | `tabs/train.py`, `training/pipeline.py` (read-only checks) |
| M4 | Settings search in the Train tab plus "changed from preset" markers and count | Power users find and review settings fast | `tabs/train.py`, `tabs/palette.py` (reuse `fuzzy_score`), `training/params.py` |
| M5 | Group headers with one-line value summaries; Simple/Advanced segmented control | Closed groups still informative | `tabs/common.py` (`CollapsibleSection`), `tabs/train.py` |
| M6 | Samples scrubber (epoch slider, large preview, crossfade, "Use this epoch") | Pick the best epoch visually | `tabs/train.py` (samples tab) |
| M7 | Selection bar for batch actions; hover quick actions on cards; stronger selected state | Batch editing discoverable | `tabs/workspace/delegate.py`, `tabs/workspace/browser.py` |
| M8 | Review window: key caps, before/after split, progress count | Faster, more enjoyable review | `tabs/review.py` |
| M9 | Health summary header, bucket bar chart, per-bucket warning, progress bar | Understand the dataset at a glance | `tabs/workspace/panels.py`, `core/health.py` (data only) |
| M10 | Editor: single large canvas, draggable before/after, operation stack, on-canvas crop | Feels like an editor | `tabs/editor.py` |
| M11 | Toast and banner layer with undo | Non-blocking feedback | `tabs/common.py`, `main.py` |
| M12 | Welcome dialog and demo dataset entry | First run success | `main.py`, `tabs/help.py`, `tools/make_screenshots.py` (demo data reuse) |
| M13 | Settings: swatch theme picker, section list with search, Diagnostics card | Delightful, supportable | `tabs/settings.py` |

### Large (L)

| # | Change | User benefit | Touches |
|---|---|---|---|
| L1 | Gallery Compare and Survey views (C/N), keep/reject keys, used by Health groups | Real dataset curation workflow | `tabs/workspace/*`, new `tabs/compare.py`, `core/health.py` |
| L2 | Culling flags and ratings (pick/reject, 1-5), auto-advance, filters `flag:` and `rating:` | Fast first pass; feeds export/training selection | `core/query.py`, `core/projects.py`, `tabs/workspace/*`, Datasets export |
| L3 | Guided "New LoRA" flow: a staged wizard that fills the Train tab (dataset check, family and preset, files, preflight, start) | Beginners succeed without learning the tab | new `tabs/train_wizard.py`, `tabs/train.py` |
| L4 | Run dashboard and history: runs list with notes, loss, samples, restart-with-same-settings; compare runs (overlay loss, A/B samples) | Learn from experiments; Fizgig parity on Runs and outputs | `tabs/train.py`, `training/pipeline.py` (run folder readers) |
| L5 | Interactive coach-mark tours (overlay) per tab | Learn in context | new `tabs/tour.py`, `tabs/help_content.py` |
| L6 | Layout customisation: dock panels (drag, dock, save workspace layouts), per-task layouts like Blender workspaces | Power users shape the app | would need `QDockWidget` refactor of tab layouts |
| L7 | Local status page or phone view for run progress | Watch a run from the sofa (Fizgig has this) | new small HTTP server in the app process; a design and security decision for the owner |

## 9. Feasibility in PySide6 and limits

Feasible with PySide6 as is:

- Token-driven QSS with dynamic properties, hover/pressed/focus states, rounded corners, custom scroll bars, segmented
  controls (checkable `QPushButton`s in a group), chips and cards (`QFrame` with object names).
- Custom painting with `QPainter` for chart, bucket bars, tiles, memory strip (already done), before/after slider,
  crop handles (`QGraphicsView` overlay), selection check marks in the item delegate (`tabs/workspace/delegate.py`).
- Motion with `QPropertyAnimation`, `QParallelAnimationGroup`, `QGraphicsOpacityEffect`. Keep it on small widgets.
- Native notifications with `QSystemTrayIcon.showMessage` (works on Windows and most Linux desktops; behaviour in a
  locked-down session or without a tray is not testable here).
- Drag and drop with `QMimeData` and drag-enter overlays; docking with `QDockWidget`.
- Optional charting with the `pyqtgraph` package (fast, hover and zoom out of the box) or QtCharts; adding a package
  needs an owner decision (rule 2 and the installer), and `QPainter` is enough for the current needs.
- Overlays for tours: a translucent top-level child widget above the central widget.

Limits and cautions:

- QSS is CSS-like but not CSS: no variables (so generate the string from tokens), no real transitions (animate with
  `QPropertyAnimation` instead), limited `box-shadow` (use `QGraphicsDropShadowEffect` sparingly), no CSS grid, and
  some properties differ per widget. A stylesheet change re-polishes every widget, which can be slow on huge trees;
  set it once at start and on theme change.
- True blur/acrylic backgrounds and smooth GPU-composited animation on huge lists are not practical with Widgets. QML
  (Qt Quick) would do this better but would mean rewriting the UI and mixing toolkits; not recommended for this app,
  whose strengths are dense forms and a virtualised grid.
- Emoji do not render in the offscreen test environment (existing rule), another reason to prefer icons.
- A web view (`QtWebEngine`) could host a Fizgig-style dashboard but adds a large dependency and a second styling system;
  not recommended. A tiny optional local HTTP status page (L7) is a lighter route.
- Accessibility APIs in Qt Widgets are decent on Windows (UI Automation) but need accessible names set by hand.
- Nothing proposed here can be verified without running the app on a real display; theme values need review by eye on
  more than one monitor, and Windows high-DPI scaling at 125/150% needs a manual pass.

## 10. What would make TagScribeR stand apart

Fizgig's web UI is a trainer with a dataset workflow bolted on the front; kohya_ss, OneTrainer and AI-Toolkit are
settings forms; taggui is a captioner. TagScribeR's difference is **the dataset is the product and the trainer is the
last step**, in a native, offline, keyboard-friendly app. Lean on that:

1. **One continuous pipeline you can see.** A persistent header strip (folder, counts, health score) and a visible
   workflow order from "open folder" to "trained LoRA", with every stage reading the same live dataset. Web trainers
   upload a folder; TagScribeR lives in it.
2. **Dataset quality as a living score.** Health, token counts, missing captions, bucket balance and duplicate groups
   unified into a pre-training report on the Train tab ("your dataset is ready: here is what would improve results").
   No other trainer connects data health to the Start button.
3. **Culling in the style of Lightroom for datasets**: Compare/Survey, flags, ratings, keep/reject keys, auto-advance,
   with caption review done the same way. This is the strongest "premium" signal for the audience.
4. **Honest, human-readable training**: Adaptive LR decisions explained in words on the chart, a "what changed from the
   preset" view, plain-language ready checks, and a per-epoch sample scrubber that picks the epoch for you to choose.
5. **Native feel**: instant filters over thousands of images, real drag and drop, tray notification, keyboard shortcuts
   everywhere, command palette across tools and settings, interface scale to 200%. Web UIs cannot match native
   responsiveness on a 5,000-image grid.
6. **Privacy as a feature**: the Metadata audit with a clear result card, offline operation, no account, a "nothing left
   your machine" indicator in the status strip when no API server is configured.
7. **Calm, neutral, image-first look**: neutral surround, one accent, no decorative colour; the images are the colour.

Where Fizgig's web UI is better and worth matching: remote and phone viewing of a run, a pods and multi-computer story,
a first-run tour and video, and a runs history with restart-with-same-settings. These are L-tier items and some (pods)
are out of scope for a local studio; surface them as questions for the owner.

## 11. Open questions for the owner

- Is a visual redesign wanted before the next training ports (Klein, Krea 2, H3), or after? The token module (M1) is
  independent of the training code and low risk.
- Keep `qt_material` as the base, or replace it with generated QSS from tokens? Replacing it is cleaner but touches
  every widget's look and needs a full visual regression pass; recommended, in stages.
- Is a new package (for example `pyqtgraph`) acceptable, or should charts stay `QPainter`? Recommendation: stay with
  `QPainter` for now.
- Should the sidebar order change (grouped by pipeline stage)? It changes the Ctrl+1..7 mapping that users may have
  learned.
- Is a local web status page for runs (L7) wanted? It has security implications (a listening port) and is a Fizgig
  parity feature rather than a TagScribeR necessity.
- Light theme: keep supporting all `qt_material` light themes, or reduce to one light and one dark with accent choice?
  (Rule 2: do not narrow without asking.)
- Screenshots in the README will change with any restyle; regenerate only with `tools/make_screenshots.py`.

## 12. Sources

Product and vendor documentation:

- Fizgig README and docs: https://github.com/shootthesound/Fizgig, https://github.com/shootthesound/Fizgig/blob/main/docs/WEB_UI.md
  (also read locally in the git-ignored checkout used for porting)
- taggui: https://github.com/jhc13/taggui
- OneTrainer: https://github.com/Nerogar/OneTrainer
- Blender manual (interface, workspaces, keymap, themes): https://docs.blender.org/manual/en/latest/interface/index.html

Guidance and standards:

- NN/g, Progressive Disclosure: https://www.nngroup.com/articles/progressive-disclosure/
- NN/g, Designing Empty States in Complex Applications: https://www.nngroup.com/articles/empty-state-interface-design/
- NN/g, Response Times: The 3 Important Limits: https://www.nngroup.com/articles/response-times-3-important-limits/
- W3C WCAG 2.2, Understanding SC 1.4.11 Non-text Contrast: https://www.w3.org/WAI/WCAG22/Understanding/non-text-contrast.html

Practitioner articles (secondary, treat as opinion): 

- Lightroom Compare/Survey and culling: https://lightroomkillertips.com/grid-loupe-compare-survey-view-one-use/,
  https://www.picturecorrect.com/lightroom-culling-mastery-tips-for-choosing-your-best-photos/
- DaVinci Resolve pages: https://www.tourboxtech.com/en/news/davinci-resolve-pages-explained.html
- Command palette patterns: https://ux.hubtel.com/design-system/components/command-palette/,
  https://www.techinterview.org/post/3233475212/build-command-palette-cmd-k/
- Dark mode design: https://sep.com/blog/6-tips-for-designing-a-dark-mode-version-of-your-app/,
  https://uxcel.com/blog/12-principles-of-dark-mode-design-627,
  https://www.accessibilitychecker.org/blog/dark-mode-accessibility/
- darktable neutral/mid-grey interface rationale and user feedback: https://discuss.pixls.us/t/neutral-theme-for-darktable-and-gimp/9504

Not read or not reachable during this research (claims about them are from general knowledge and flagged as such):
Adobe's Lightroom help pages (HTTP 403), Capture One, Eagle's own documentation, Figma, Linear and Raycast product
pages (only generic palette guidance was found), ComfyUI, SwarmUI, Forge, InvokeAI, kohya_ss and AI-Toolkit UIs, Material
Design 3 colour pages (fetch returned a title only), Qt class documentation pages (truncated). Their lessons appear
here only where a read source supports them.

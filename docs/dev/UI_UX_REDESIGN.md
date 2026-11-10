# TagScribeR UI/UX redesign: review, direction, speed budget and staged plan

Development notes, 2026-11-10. A design review and a buildable plan. Nothing here is implemented or decided; the
product decisions are listed as questions at the end. It follows `UI_UX_RESEARCH.md` (the earlier study) and does not
repeat it: section 1 says what that study got right and what its stage 1 delivered, the rest goes further.

**How this was made.** The real window was built offscreen with generated demo data and an isolated `user_data`
(the approach of `tools/make_screenshots.py`, driven from a scratch script that writes nothing into the repository)
and every tab was looked at under `tagscriber_dark`, `tagscriber_light` and the default `dark_teal`. Timings were
taken on CPU with generated pictures in a temporary profile, with the GPU hidden from the process. No real picture,
dataset, model weight or training run was used, and nothing was looked at on a real screen: every visual judgement
below is from offscreen renders at 100 % scale. A mockup (one HTML file, made-up data) was delivered with this review;
it is not in the repository (see question 12).

---

## 1. The earlier research and stage 1, judged

**What the research got right.** The diagnosis ("no visual system", one design-token module first), the button
hierarchy, one accent, the missing ambient job indicator, empty states, the ready check on Train, the selection bar,
and the insistence that the chrome stays neutral around pictures. All of that still stands and is built on here.

**Where it fell short.**

- It was written from screenshots and source, not from the running app, so it missed the two things that hurt most
  in use: the window **freezes** (measured in section 5: 20 s to open 20,000 pictures, 4 s every time the Gallery is
  shown again) and the window **cannot be made narrower than about 1,470 px** (1,713 px under the classic theme).
- It treats speed as a property of the toolkit ("web UIs cannot match native responsiveness on a 5,000-image grid").
  Today that claim is not true of TagScribeR: sorting 5,000 pictures takes 0.4 to 1.3 s on the UI thread.
- It proposes components but no identity: nothing says what TagScribeR should look like that another tool does not.
- Its token table makes every control outline 3:1 against the panel. Stage 1 implemented that faithfully, and the
  result is the main reason the new themes still read as a wireframe (see tell 3 below).

**What stage 1 (`tabs/theme.py`, commit "Themes: TagScribeR dark and light") delivered.** A sound foundation:
tokens, one generated stylesheet, variants and roles by dynamic property, painter colours by key, the classic look
preserved exactly through `LEGACY_QSS` / `LEGACY_COLORS`, sentence-case buttons, the calmer sidebar row, a focus ring
that is always visible. Applying a token theme costs 4 to 7 ms against 160 ms for a qt_material theme.

**What stage 1 did not reach** (seen in the renders, confirmed in the source):

| Area | State under the token themes |
|---|---|
| The image grid (`tabs/workspace/delegate.py`, `browser.py`) | Ten hard-coded colours and five inline styles. Under **TagScribeR light** the grid is a black slab with dark cards inside a light window. Under TagScribeR dark the grid is neutral grey (`#1f1f1f`, `#2b2b2b`) inside blue-grey chrome: two greys that do not belong together. |
| Toolbar | `Open Folder` is filled accent and `Save` is filled blue (`#0984e3`, inline) side by side: two primary buttons, two hues. |
| Tag bubbles (`core/widgets.py`) | Hard-coded dark; black pills on the light theme. |
| Inspector preview | Hard-coded near-black surround; a 1280 x 853 picture sits in a tall black box. |
| Memory strip (`tabs/memory_bar.py`) | A green-to-brown gradient bar and a blue bar; "peak 21.5" is clipped by the fill edge. The loudest element on every screen. |
| Health badges | White on purple (`rgb(155, 89, 182)`): a hue that means nothing in the token set. |
| Gallery batch panel | A purple `Auto Tag` button is still the only filled button in the panel. |
| Status line | `#9a9a9a` inline: 2.7:1 on the light surface. |
| Layout | Unchanged: 200 px sidebar, two toolbar rows, modal dialogs, the status line in each tab. |

So the honest summary of stage 1 is: **the forms are themed, the workspace is not.** The part of the app people look
at for hours (the grid, the card, the inspector) still has the old look in every theme.

---

## 2. Critique of the app as it renders today

Judged under TagScribeR dark unless said otherwise. "B" = costs a beginner, "P" = costs a power user.

**Shell.** Seven rows in a 200 px sidebar use 14 % of a 1440 px window to show seven words. No header says which
folder is open; the path is at the bottom of each tab's grid in 9 pt grey. Nothing outside the Auto Caption or Train
tab shows that a job is running (B, P). The memory strip spans the full width under every tab and is the most
saturated thing in the window. Help, Patreon and Ko-fi sit under a large empty column.

**Gallery.** Two toolbar rows hold fourteen controls of equal weight; `Undo`, `Redo`, `Save`, `Copy to Collection`
are a second row that exists on five tabs. Cards are a box (card) inside a box (thumbnail well) with a third box
drawn on selection; the file name is 7.5 pt at 4.1:1; the third caption line is cut through the middle of its
letters at most sizes ("landscape, l" with the bottom half missing); every card carries a resolution pill, so the
pills stop meaning anything. Selection is a 2.5 px outline only: at small sizes five selected cards among sixteen
are hard to count (P). With no folder open, the window is an empty dark rectangle with "Open a folder to start." in
small grey type at the bottom left, while every toolbar control is enabled and does nothing (B). The inspector's
previous / next buttons are bare `<` and `>` glyphs; the most useful number ("22 CLIP tokens") is the smallest text
on the screen. Health results are a tree of text; the bucket table has no bars.

**Auto Caption.** The primary action is below the fold: at 1600 x 950 the panel ends inside "Generation" and the Run
button is not on screen (B, P). The panel is one scroll of five group boxes. Icon-only buttons with no label (refresh, folder, save, more) sit beside the model and the preset. There is no place where results appear while a run goes: progress is per-card badges and a log box.

**Image Editor.** At 1600 px the right panel is wider than its space: the after-picture, the third rotate button and
the right edge of every row are cut off. Each operation has its own Apply button; the object being edited (the
picture) gets a 230 px preview while the grid, which is only a picker here, gets 60 % of the window.

**Datasets.** A plain list of names and counts; the export is a modal dialog over the workspace with nine controls
in three group boxes. Filled accent ("Add 16 to ...") and filled red ("Delete") sit one row apart.

**Metadata.** The audit result, the best content on the tab, is seven lines of 9 pt coloured text with a dot.
`Audit` and `Clean metadata...` run off the right edge at 1600 px, as in the Editor.

**Train.** Before a run: a form in which `Load`, `Save...`, `Delete`, `Import...`, `Last run` (and now `Export for
sharing...`) are six equal buttons beside the preset; four model rows each end in `Browse` and `Get`; "Epochs: 30"
is a full-width field for a two-digit number. During a run: the state of an hours-long job is one 10 pt line
("Training - epoch 9/30 - step 171/600 - 3.02 s/it - ETA 21:27 - loss 0.1259") over a thin bar; the left half of the
window still shows the form, most of it now irrelevant; the console takes the lower right third. Eight buttons sit
above the chart in three rows of unequal width. `Stop` is filled red and as large as `Start`.

**Settings.** One column of full-width group boxes; a combo box more than 1,200 px wide holds "100%". Theme is a text list.

**Help Center, palette, Hard images.** The Help Center is the most finished screen (clear tree, good type). The
palette is a list of "Group:  Title" strings with the shortcut appended as text; no icons, no key caps, no recent
commands. The Hard images table is clear, but flagged rows are coloured red text only.

**Consistency.** The grid toolbar is laid out 3 px lower on Auto Caption than on the Gallery; the right panel is a
tab widget on three tabs, a scroll of group boxes on two and a splitter on one; section headers are group-box titles
in some panels and collapsible toggles in others, at two different sizes; messages are about a hundred `QMessageBox` /
`confirm` call sites and one status-line flash.

### The ten things that most make it look like a hobby tool

1. **Every control has the same weight.** Two dozen outlined rectangles per screen, and where a button is filled
   there are two or three of them in different colours.
2. **The picture grid ignores the theme**, in every theme, and is a dark slab in the light one.
3. **Bright outlines around everything** (inputs, buttons, lists, group boxes): the window reads as a wireframe of
   itself. Professional tools separate surfaces by fill and keep outlines for focus and state.
4. **Forms stretched to the window.** A two-digit number in a 600 px field, a 1,200 px combo box, label columns of
   different widths in neighbouring boxes. Nothing has a measure.
5. **Panels that do not fit their space**: content cut off at the right edge of the Editor and Metadata panels, the
   Run button below the fold, a minimum window width of 1,470 px (1,713 px classic).
6. **Text glyphs as icons** (`<`, `>`, `↓`, `★`, `…`) beside real icons, and icon-only buttons with no label where
   the icon is not self-evident (the preset row of Auto Caption).
7. **The status bar that shouts**: gradient memory bars under every tab, while the things a user needs at a glance
   (which folder, what is running, how long is left) are in 9 pt grey or not shown at all.
8. **Empty, loading and finished states were never designed**: a blank rectangle before a folder is open, a frozen
   window while one opens, a modal box when a job ends.
9. **A long job has no face.** Hours of training are one line of small text; a captioning run has no results view.
10. **It stutters.** Opening, sorting, filtering and coming back to the Gallery each freeze the window for a
    noticeable time (section 5). Nothing reads as amateur faster than a window that stops answering.

---

## 3. Design direction: "Trace"

### 3.1 The idea

TagScribeR's own mark is a letter drawn as circuit traces: thin lines, round nodes, green at the top running through
teal to violet. The look is named after it and takes three things from it.

- **The line.** The interface is a thin drawn line around the pictures. Surfaces are flat graphite fills; structure
  comes from a 1 px hairline and from one surface being a step lighter than the one under it. No bevels, no
  gradients on controls, no glow.
- **The node.** State is a small round mark: the unsaved dot, the status dot in a chip, the nodes of a job's stages
  ("Checks, Latents cached, Captions cached, Training, Finished"), the tick in a selected card.
- **Two inks with meanings.** Teal-green is **yours**: the action to take, what is selected, what is saved and
  going well. Violet is **the machine's**: a caption a model proposed, a picture a model is working on, the
  captioning job. Nothing else is violet, and violet is never a button. The mark's full gradient appears in exactly
  two places: the logo and the 2 px progress line of a running job.

**The pictures are the colour.** Chrome is neutral (the token greys carry a slight cool cast, the picture wells
none), so a dataset of warm photos does not fight a tinted interface.

**What it deliberately is not.** Not Material (no caps, no ripples, no floating shadows on cards). Not a web
dashboard in a window (no hero gradients, no glass or blur, no card grid for its own sake). Not the look of any
other trainer or tagger: the names, the marks and the layout are TagScribeR's (`NAMING.md`, standing rule 10).
Not decorative: if an element has a colour, the colour says something.

### 3.2 Tokens (extends `tabs/theme.py`; names are proposals)

Stage 1's values stay unless marked. New tokens are in bold. Every text pair must pass the existing contrast test in
the theme tests; the values marked "tune" need a look on a real screen.

| Token | Dark | Light | Use |
|---|---|---|---|
| `bg_app` | `#16181b` | `#eef0f3` | behind panels, the dashboard ground |
| `bg_surface` | `#1d2024` | `#f8f9fb` | panels, rail, top bar |
| `bg_raised` | `#252a2f` | `#ffffff` | inputs, tiles, chips |
| `bg_hover` | `#2d333a` | `#e4e8ec` | hover rows and buttons |
| **`bg_sunken`** | `#111316` | `#dfe3e8` | the grid's ground, chart, console, command field (tune) |
| **`bg_overlay`** | `#2a3036` | `#ffffff` | menus, palette, toasts, the selection bar (with the one shadow) |
| `bg_image` | `#0d0e10` neutral | `#cfd4da` neutral | the well a picture sits in; a setting offers dark / mid grey / light |
| `border_subtle` | `#2f353c` | `#dde1e6` | hairlines between regions |
| **`border_control`** | `#454d57` | `#aab2bc` | default outline of inputs and secondary buttons (about 2:1; see 3.8) |
| `border_strong` | `#6b747e` | `#88919b` | hover outline; the default outline when "Stronger outlines" is on |
| `text_primary` / `text_secondary` | as stage 1 | as stage 1 | body / labels and hints |
| **`text_tertiary`** | about `#828a94` | about `#6b747e` | file names, axis labels, placeholders: 4.5:1 on the surface, asserted in tests |
| `accent` + hover, pressed, subtle | as stage 1 | as stage 1 | the one action, selection, "live" |
| **`ai`**, **`ai_subtle`** | `#a78bfa`, 16 % | `#6846d2`, 11 % | machine-written or machine-busy (6.0:1 and 5.9:1 on the surface) |
| `success` `warning` `danger` `info` | as stage 1 | as stage 1 | status only |
| **`*_subtle`** for the four | 14 % over surface | 9 to 11 % | banners, row tints, pill grounds |
| `focus_ring` | `#5eead4` | `#06483a` | 2 px, always |
| **`scrim`** | black 55 % | `#1e2837` 30 % | behind a modal sheet |

Accent rules: one filled accent button per region at most; the accent is otherwise a 2 px ring (selection), a 3 px
bar (current navigation), a dot or a thin line. A filled danger button exists only inside a confirmation; `Stop` at
rest is a danger outline.

Chart colours (keys already in `PAINT_KEYS`): every step = `text_tertiary` at 55 %; smoothed = `text_primary`, 2 px;
epoch average = `accent` dots; learning rate = `warning`, dashed, own axis; "now" = `accent` rule. Lines differ by
stroke (thin, bold, dots, dashes), so the chart reads without colour.

### 3.3 Type

Segoe UI (static family, as stage 1 chose), Cascadia Mono / Consolas for paths, filter terms, the console and key
caps. Sizes in points so display scaling holds. Segoe UI's digits are fixed-width, so counters and tiles do not
jitter; on other fonts request tabular figures where the Qt version allows it (to check on Linux).

| Role | Size / weight | Where |
|---|---|---|
| Micro label | 8 pt / 600, upper case, +6 % tracking | group labels in panels only ("TAGS", "THIS PICTURE") |
| Mark | 7.5 pt / 600 | the text inside a card mark |
| Small | 8.5 pt / 400 | card file names, axis labels, chip details |
| Hint | 9 pt / 400 | hints, secondary lines |
| Body | 10 pt / 400 | controls, lists, captions on cards at default size |
| Strong | 10 pt / 600 | names, the active tab, primary button |
| Section | 11 pt / 600 | panel section titles, box titles |
| Title | 13 pt / 600 | the run's name, dialog titles |
| Figure | 16 pt / 600 | stat tiles |
| Display | 18 pt / 600 | the empty state's headline only |

Sentence case everywhere. No text below 8.5 pt except inside a mark.

### 3.4 Space, radius, elevation

4 px scale: 2, 4, 8, 12, 16, 24, 32. Panel padding 12; gaps between boxes 12; between a label and its control 8.
Form rows have a **measure**: a label column of one width per panel and controls no wider than 360 px (numbers 96
px, a path 100 % of the column); the rest of a wide panel stays empty.

Radius: 4 marks and thumbnails, 6 controls, 8 tiles and cards, 10 boxes and floating sheets, full for chips.

Elevation is lightness plus a hairline. There is **one shadow** in the system, under floating things (menu, palette,
toast, selection bar). In Qt it is painted, not an effect: the floating widget is a translucent frameless window
whose own `paintEvent` draws a nine-slice shadow pixmap around its rounded body. `QGraphicsDropShadowEffect`
re-renders its widget into an off-screen pixmap on every repaint and must never sit on anything that repaints often
or is large; `QGraphicsOpacityEffect` has the same cost and is kept to toasts.

### 3.5 Icons and density

One family (Material Design outline through qtawesome, as now), 16 px in buttons, 18 px in toolbars, 20 px in the
rail, coloured `text_secondary` at rest and `text_primary` on hover or when active. Every glyph button (`<`, `>`,
`↓`, `★`, `…`) becomes an icon with a tooltip that names its shortcut and an accessible name. An icon alone is
allowed only for the universal ones (undo, redo, close, search, previous, next, more); everything else is icon plus
word.

Density is a setting with two values: **Comfortable** (control 32 px, list row 32, grid gap 12) and **Compact**
(28 / 26 / 8, the default, today's size). It scales the control-height and spacing tokens; the stylesheet is
rebuilt, which costs a few milliseconds under a token theme.

### 3.6 Components

| Component | Specification |
|---|---|
| Button | 28 px high, radius 6, 12 px side padding. **Primary** filled accent, 600. **Secondary** raised fill, `border_control`. **Quiet** no outline until hover. **Danger** red outline at rest; filled only as the confirm step. Disabled 40 % with no outline. A button that starts something long shows "Starting..." and is disabled until the job answers. |
| Confirm in place | `Stop`, `Delete`, `Clear captions` do not open a dialog: the button turns into a filled danger "Stop now?" for 5 s beside a quiet "Keep running" (replaces most of the message-box call sites over time; real data loss still asks in a dialog). |
| Input | 28 px, raised fill, `border_control`, focus = 2 px ring with no size change. Error = danger outline and one line of text under the field. Numbers right-aligned with the unit after them ("0.25 MP"). |
| Segmented control | Checkable buttons in one outline; the checked one has the `accent_subtle` fill and 600. For 2 to 4 exclusive choices (Simple / All settings; Selected / Shown / No caption / All; S / M / L). |
| Tabs | Text with an optional icon, 2 px accent underline on the current one; a count pill may follow the word (Health 9). |
| Chip | Full radius, raised fill, hairline; optional status dot; a bold key and a plain value ("**Rank** 8"). Clickable chips get the hover outline. Used for dataset counts, a run's recipe, a guided step's summary. |
| Box | Surface fill, hairline, radius 10, a 13 pt title row with quiet actions on the right. Replaces `QGroupBox` titles that sit on the border. |
| Stat tile | Raised, radius 8: 8.5 pt label, 16 pt figure, optional small unit. Fixed order per screen. |
| Table | 26 px rows, no vertical grid lines, sticky header in `text_secondary`, numbers right-aligned, the status column a pill plus word. |
| Image card | Section 3.7. |
| Toast | Bottom centre of the content area, overlay fill with the shadow, an icon, one line, one action ("Saved 12 captions. Undo"). 4 s, pauses under the pointer, at most three stacked. Errors do not fade. |
| Banner | A full-width strip at the top of a panel with a `*_subtle` fill: for a state that persists ("3 captions failed. Retry"). |
| Dialog vs sheet | A modal dialog only for what must stop everything (unsaved changes on quit, data loss). Everything else (export, save a preset, hard images, review) is a **side sheet**: a panel that slides over the right 480 px of the workspace with the scrim on the rest, so the grid stays visible. |
| Job chip | In the top bar: a status dot, the job's name, "132 / 480, 17:59 left", a 2 px gradient line along the bottom as progress. Click = go to the run. More than one job = one chip each, up to two, then "+1". |
| Progress | 8 px bar, radius 4. Determinate whenever a total is known; with it always a count, a rate and a time left. A row of stage nodes above it for a job with stages. |
| Empty state | A headline that says what belongs here, one sentence, one primary action, optionally one quiet alternative. Never an icon alone, never "No data". |
| Skeleton | A flat `bg_raised` block in the shape of what is coming (a tile, three text lines). No shimmer in the grid; elsewhere one shared 1.2 s opacity pulse driven by a single timer. |
| Key cap | Mono 8.5 pt in a hairline box with a 2 px bottom edge; used in tooltips, the palette, the review bar and Help. |

### 3.7 The image card

Painted by `CardDelegate`; no widgets. One tile, not a box in a box.

- **Layout.** A square well in `bg_image`, radius 4, the picture fitted inside (never cropped by default: people
  must see the whole frame they train on). Under it the file name (8.5 pt, tertiary, middle-elided) and, when
  "Captions" is on, two full lines of caption (never a cut line: the height is a whole number of lines). The card
  has no fill and no outline at rest; the 10 px gap is the separation.
- **Hover.** `bg_hover` fill behind the whole card.
- **Selected.** 2 px accent ring, `accent_subtle` fill, a filled accent circle with a tick at the top left.
- **Current** (the keyboard's place): a 2 px focus ring 2 px outside the card, independent of selection.
- **Marks.** Bottom left of the well, at most two and then "+2": a dark 82 % pill, a 6 px dot in the status colour
  and a word in sentence case ("no caption", "similar", "duplicate", "soft", "small", "proposal", "hard image").
  Below 120 px cards the word is dropped and the dot stays. Amber for "to do" and quality warnings, blue for
  duplicates and near-duplicates, violet for a proposal, red only for a failure. "No caption" is amber, not red: it
  is work to do, not an error.
- **Unsaved.** An amber dot at the top right with a dark halo.
- **Job state.** Queued: the picture at 55 %. Working: a 2 px violet ring inside the well (one shared timer may
  rotate an arc at 8 frames a second, and only for cards that are working and visible). Done as a proposal: a
  violet circle with a tick. Failed: a red circle with a cross.
- **Details view.** The resolution pill, megapixels and bucket appear on every card only when "Details" is on, and
  on the hovered card otherwise (question 8).
- **Loading.** The well alone with the file name; the picture appears when decoded, with no fade while the view is
  scrolling.
- **Unreadable.** The well with a small broken-picture icon and "Can't read this file"; the tooltip has the reason.

### 3.8 Motion

| Thing | Motion |
|---|---|
| Hover, press, focus | none: the state changes at once (QSS has no transitions, and instant is right here) |
| Side sheet, inspector show / hide | 140 ms, OutCubic, on `geometry` or `maximumWidth` |
| Toast in / out | 140 ms in (10 px rise and opacity), 200 ms out |
| Scrim | 120 ms opacity |
| Collapsible section | 120 ms on `maximumHeight`; none when the content is taller than 600 px |
| Progress value | 160 ms linear between two reports, so a bar never jumps |
| Tab and view switch | none |
| The previews' epoch scrubber | a 120 ms cross-fade between two epochs, painted by the scrubber itself |

Never animated: the grid while it scrolls or lays out, chart data arriving, form rows appearing or leaving on the
Train tab, text. Nothing lasts longer than 240 ms. **Reduce motion** is a setting that sets every duration to 0 and
follows the system preference where it can be read (on Windows the "animation effects" setting; how Qt exposes it in
the version in use was not checked).

### 3.9 Accessibility targets

- Text at least 4.5:1, including hints, file names and placeholder text; asserted by the theme tests for both
  themes.
- Focus ring, selection ring and the checked state of a control at least 3:1 against what is behind them.
- Control outlines at about 2:1 by default, because the fill already separates a field from its panel; a
  **Stronger outlines** switch restores stage 1's 3:1 outlines everywhere (question 5).
- Colour is never the only signal: every status has a word or a shape beside its colour.
- Everything reachable by keyboard: F6 cycles the regions (rail, top bar, grid, inspector), Tab moves inside one,
  the palette reaches every action, and every icon-only button has an accessible name.
- Scale 100 % to 200 % (the setting exists). At 100 % the window must be usable at **1280 x 720**.
- Targets at least 24 x 24 px; 28 px for the ones used all day.

---

## 4. Layout and flow, screen by screen

### 4.1 The shell

**Why.** The folder, the unsaved count and the running job are facts about the whole app, so they belong in a place
that is on every screen; today each tab repeats part of them and none shows all.

**Layout.** A 72 px **navigation rail** (icon over an 8 pt word, so a beginner still reads the names) and a 44 px
**top bar**. The rail keeps today's order and shortcuts (Ctrl+1 to 7); Help and Settings move to its foot, with the
two support links under a hairline as small quiet buttons. The rail gives 128 px back to the content.

The top bar, left to right: the **folder switcher** (name, parent path in grey; a menu with recent folders, Open
folder, the Subfolders option, Rescan); **count chips** (pictures, without caption, unsaved; each sets the matching
filter); the **command field** ("Search or run a command, Ctrl K"); Undo, Redo; **Save n** (a primary button that
exists only while something is unsaved); the **job chips**; two **memory meters** (4 px bars with the peak mark;
click opens the full strip of today).

**States.** No folder: the switcher reads "No folder open", counts, Save and meters are absent. A job running: its
chip, and a dot on the rail item it belongs to. A job finished while the window was not active: a notification and a
chip that stays until clicked ("Captioning finished: 478 written, 2 failed").

**What moves.** `Open Folder`, the recent menu and `Subfolders` leave every tab's toolbar for the switcher. `Undo`,
`Redo`, `Save` leave every tab for the top bar (one undo stack, one place). The status line's counts move to the
chips. The memory strip becomes the two meters by default; the strip stays available (question 3).

**Qt.** `QMainWindow` central widget = rail (`QWidget` with checkable `QToolButton`s, `ToolButtonTextUnderIcon`) +
a `QVBoxLayout` of top bar and `QStackedWidget`. A `JobCenter(QObject)` in `tabs/workspace/jobs.py` collects what
the batch worker, the file jobs and the Train tab already report and emits `changed`; the chips and the rail dots
listen to it. Toasts are child widgets of the central widget, positioned in `resizeEvent`.

### 4.2 Gallery

**Why.** This is where hours are spent; the picture area should be as large as possible and every tool one step
away, not always on screen.

**Always visible.** One filter row: the filter field with its terms shown as removable tokens (the text stays
editable as text; tokens are how it is drawn when not focused), saved filters, sort, thumbnail size, Captions /
Details, the inspector toggle. The grid. A one-line footer (shown count, sort, path).

**On demand.** The **selection bar**: a floating bar at the bottom centre of the grid whenever something is selected
("5 selected: Add tag, Caption, Copy to collection, Recycle Bin, Clear"). The **inspector** (344 px, F4 hides it):
Inspect, Batch, Tags, Health. A **loupe** on Space (the picture at window size over the grid, arrow keys move). A
side sheet for Health groups (compare, keep, remove).

**Inspector, Inspect.** The picture fitted in a 210 px well; one file line; the caption editor; a **token meter**
(a 4 px bar, "22 / 75 tokens", amber near the limit of the family chosen on Train); tags as chips with the trigger
word marked; "This picture": health, bucket and, when a run has tracked it, whether it was a hard image.

**States.** No folder: section 4.9. Opening: the grid shows skeleton tiles at once with "Reading 20,000 pictures",
names first, captions and pictures streaming in. Filter with no result: "Nothing matches `tag:xyz`" with "Clear the
filter". Folder unreadable: a banner with the reason and "Choose another folder".

**Qt.** `QListView` in icon mode with uniform sizes stays (it virtualises well); the delegate paints section 3.7.
The filter tokens are painted in the line edit's `paintEvent` when it has no focus. The selection bar is a child of
the view's viewport parent, raised, positioned in `resizeEvent`, with the painted shadow.

### 4.3 Auto Caption as a guided run

**Why.** A run has three decisions and one action; today they are five scrolling boxes and a button below the fold.

**Before a run.** Left: the grid with a scope control (Selected 12, Shown 480, No caption 37, All 1,284). Right:
three collapsed steps, each showing its current answer as one line and opening in place: **1 Model** (local or
server, which one, loaded or not, the memory it needs if known), **2 Instructions** (preset, the subject / trigger
word as the first field, tag hints, the full prompt under "Edit the instruction"), **3 Output** (proposals for
review, or write directly; what happens to existing captions). A **sticky footer** with the one primary button,
"Caption 480 pictures", and the estimate when there is a measured one ("about 3 s a picture on the last run").

**During a run.** The right panel becomes the run: title and state, the three answers as chips, the progress bar,
four tiles (done, speed, time left, failed), then a **live feed**, newest first: thumbnail, file name, the caption
as it is written (a violet caret while the model writes), Accept / Reject on each finished one, failures in place
with the reason and Retry. Cards in the grid show their state (section 3.7). Footer: `Stop` (Esc), "Review 130
proposals" and one sentence on what stopping keeps.

**After.** A summary banner in the feed (written, failed, time, pictures a minute) and the review button as the
primary action.

**Review** becomes a mode of this tab instead of a separate window: the grid narrows to a strip of the pictures with
proposals, the picture is large in the middle, the old caption and the word-level diff on the right, a bar of key
caps under it (A accept, R reject, E edit, Space next). The current dialog keeps working until this replaces it.

**Qt.** The steps are the existing `CollapsibleSection` with a summary label in its header. The feed is a
`QListView` with a small painted delegate fed by `BatchWorker`'s item signals (streaming text arrives where the
provider streams; otherwise the row appears when the caption is done, and the mockup's caret is then absent). The
sticky footer is a fixed widget under the scroll area.

### 4.4 Review, Editor, Datasets, Metadata

- **Editor.** The picture becomes the centre: one large canvas with a draggable before / after divider, zoom to fit
  / 100 %, crop handles on the picture. The grid shrinks to a strip of the selected pictures along the bottom.
  Operations sit on the right as a short stack ("Resize to 1024 px, Crop 3:2 centre") with **one** Apply for the
  stack, and the output rule as a chip at the top ("Saves copies, originals untouched"). Moves: the per-operation
  Apply buttons into the stack. Qt: `QGraphicsView` for the canvas and handles.
- **Datasets.** Collections as rows with a four-picture mosaic, counts and a caption-coverage bar; the export is a
  side sheet with a live summary ("16 pictures, 14 with captions, into `10_ohwx`"), not a modal dialog.
- **Metadata.** The audit as a list of finding rows: an icon, "GPS location", "1 of 16", a severity colour on the
  left edge, "Show these". Cleaning ends in a result banner, not a message box.
- All three keep every function they have; only the container changes.

### 4.5 Train: a calm form before, a dashboard during

**Before a run.** Two columns. Left, the recipe as four short boxes with one label width: **What to train**
(family; kind of LoRA as a segmented control; the preset as a list with one line of description each, built-in ones
marked with the diamond), **Dataset** (the open folder, picture and caption counts, trigger word, a link to Health),
**Model files** (one row each: a status dot, the file name, "Choose" and "Get" as quiet actions, the full path in
the tooltip), **Output** (name, folder). "All settings" is a segmented control at the top with a search field;
changed-from-preset values carry a dot and a reset arrow. Preset management (`Load`, `Save`, `Delete`, `Import`,
`Last run`, `Export for sharing`) becomes one "Preset" menu button; each action stays.

Right, a **Ready check** box that is the only place with the primary button: one line per check (captions, model
files, trigger word in captions, memory where it has been measured), each green, amber with "Start anyway" allowed,
or red with a "Fix" link; then `Start training` (or `Add to queue` while a run is going). Warnings never block
(audit 13, lesson 1). No time or memory estimate is shown unless it was measured on this card (standing rule 9).

**During a run.** The form folds into one box, "Recipe of this run" (chips, "frozen when the run started", "Show
all settings"), and the tab becomes a dashboard:

- a header: the run's name, a state pill, family and preset chips, then `Sample now`, `Save now`, `Pause after
  epoch`, `Stop`;
- the stage nodes and one progress bar;
- six tiles: Epoch, Step, Time left, Speed, Loss (smoothed), Learning rate;
- **Loss** (wide): the chart with a legend, a hover readout, a "now" rule, and the Adaptive LR decisions written as
  sentences under it ("epoch 8: halved after two flat epochs");
- **Hard images**: the five highest, each with its picture, verdict as a pill and a sentence, residual and LR
  factor, and "Fix captions" (opens the existing window as a side sheet);
- **Previews**: one group per sampled epoch, the same prompts side by side, the chosen epoch ringed, "Compare
  epochs" (the scrubber) and "Use epoch 8" (shows that checkpoint); the line under it says how the previews were
  rendered (model, steps, companion LoRA);
- **Queue**: one row per queued run with name, family, preset and epochs, reorder by drag;
- the console as one line at the bottom that expands to the full log.

**Paused / finished / failed.** Paused: the header says what resuming will do and which settings apply on resume.
Finished: a summary box (epochs, final loss, time, peak memory, where the files are; "Open in the Gallery", "Queue
again"). Failed: the reason in words and what to do, above the log line that says it.

**Qt.** The dashboard is a `QGridLayout` of box widgets that becomes one column under 1,100 px (re-laid out in
`resizeEvent`); the chart and the previews strip are painted widgets (the chart already is); tiles are labels fed
by the existing progress tracker. Nothing here needs a second process or a web view.

### 4.6 First run and the empty state

The Gallery with no folder is the first screen a new user sees, so it is the onboarding: a headline ("Open a folder
of pictures to begin"), one sentence on how the app treats files, a drop zone with **Open folder** (primary) and
**Try the demo pictures** (question 7), the three steps of the pipeline in a row, the recent folders with their
counts, and a row of honest system chips (which GPU was found or "no GPU: captioning through a server still works",
whether the AI runtime is ready, "nothing leaves this PC" while no server is set up) plus "Take the 2 minute tour".
Dropping a folder anywhere in the window opens it.

### 4.7 The command palette as a first-class tool

One list, typed prefixes for scope: nothing = commands, `/` = filters (built-in and saved; Enter applies), `#` =
tags of this folder (Enter filters, Ctrl+Enter adds to the selection), `@` = a training setting (Enter goes to it,
opening "All settings" if needed), `?` = help topics. Rows have an icon, the title with the matched letters bold,
the group in grey and the shortcut as key caps on the right. Recent commands first when the field is empty. Commands
that act on the selection say so ("Add tag to 5 selected pictures..."). Running jobs appear as rows ("Training: go
to the run", "Stop captioning"). Opened by Ctrl+K, Ctrl+Shift+P or a click in the top bar's field. Qt: the existing
dialog with a `QListView` and a painted row delegate; `fuzzy_score` stays.

---

## 5. "Lightning quick" as a budget

### 5.1 Measured

Machine: a 12-core / 24-thread desktop CPU (AMD Ryzen 9 3900X), Windows 11, Python 3.12, PySide6 6.10, offscreen
Qt (raster painting, scale 1). Other test suites were running on the machine during the measurements, so the
figures are on the slow side and vary by about 10 %. Files were in the operating system's cache unless said. "5k" /
"20k" are folders of 5,000 / 20,000 generated JPEGs of about 0.1 MP with tag captions.

**Start.** From process start to the first painted window, without the AI runtime preload:

| Step | TagScribeR dark | dark_teal |
|---|---|---|
| Imports (core, logging, PySide6, all tabs) | 1.0 s | 1.0 s |
| Apply the theme | 0.005 s | 0.16 s |
| `MainWindow()` (2,832 widgets) | 1.33 to 1.35 s | 1.87 to 1.97 s |
| Show and first paint | 0.37 to 0.41 s | 0.42 to 0.44 s |
| **Total** | **about 2.8 s** | **about 3.5 s** |

- `import torch` alone, with the GPU hidden: **2.4 s**. `main.py` does this (and probes the devices, not measured
  here because it touches the GPU) behind a splash before the window is built. A cold start is therefore more than
  5 s before the window appears; that total is a sum of measured parts, not a measurement.
- Of `MainWindow()`, the seven tab constructors account for 0.56 to 0.88 s. A profile shows the rest is Qt
  re-resolving the stylesheet each time a finished sub-tree is put into a layout, splitter or stack: with **no**
  stylesheet the same construction takes **0.38 to 0.44 s**. Building the window first and applying the token theme
  after takes 0.38 + 0.24 s, **0.6 s less** than today. (The same trick is a loss under a qt_material theme: applying
  it to a built window takes 2.1 to 2.25 s, which is also what switching to a classic theme at run time costs.)
- Imports that the first screen does not need: `cv2` and `numpy` 0.13 s (pulled in by `core/image_utils.py` for one
  helper), `importlib.metadata` 0.17 s, `qt_material` 0.06 s under a token theme, `tabs.train` 0.11 s;
  `huggingface_hub` and `httpx` are imported on the UI thread while the Auto Caption tab is built.
- Memory after the first paint: 175 MB; 485 MB once torch is imported.
- Switching tabs with no folder open: 10 to 55 ms the first time, 10 to 30 ms after.
- Minimum window width: 1,471 px under the token themes, 1,713 px under dark_teal (the Train tab needs 1,271 /
  1,513 px, plus the 200 px sidebar).

**Opening a folder** (the UI thread is blocked for all of it):

| | 5k | 20k |
|---|---|---|
| `open_folder` returns after | 4.5 s | 20.0 s |
| of which reading names, sizes and captions | 1.6 s | 6.2 s |
| of which sorting (five browsers, one sort each) | 3.0 s | 14.2 s |
| First thumbnail after it returns | 0.18 s | 0.69 s (0.6 s of it the first paint) |
| Picture sizes known for the whole folder (background) | 2.1 s later | 8.4 s later |

One first read of 5,000 freshly written files took 19 s (cold files; probably the virus scanner): a single
observation, but it is what a UI-thread read can cost on a cold disk.

**Using the grid.**

| | 5k | 20k |
|---|---|---|
| Coming back to the Gallery tab | 1.0 s | 4.0 to 4.6 s |
| Other tab switches | 13 to 61 ms | 12 to 60 ms |
| Filter, text or tag (apply + relayout) | 96 to 176 ms | 232 to 922 ms |
| Clearing the filter | 0.46 s | 2.0 s |
| Sort by name / megapixels / caption length / tag count | 0.4 / 0.7 / 0.8 / 1.3 s | 1.7 / 2.8 / 3.5 / 5.1 s |
| Select one picture | 47 ms | 58 ms |
| Select all | 105 ms | 374 ms |

- Coming back to the Gallery re-reads every caption file on the UI thread (`pick_up_external_changes`).
- Scrolling, thumbnails in memory: at 220 px cards with captions 19 cards a frame, **0.65 to 0.69 ms a card**, frame
  median 14 to 15 ms (95th percentile 17 to 20). At 100 px cards without captions 86 cards a frame, 0.25 to 0.29 ms
  a card, frame median **24 to 27 ms** (95th percentile 25 to 39): under 60 frames a second, on a fast desktop CPU
  at scale 1.
- Where a card's paint time goes at 220 px (profiler): 32 % in `QFontMetrics.horizontalAdvance` (the caption is
  re-elided by binary search on every paint, about nine calls a card), 9 % drawing the pixmap (a 384 px thumbnail
  smooth-scaled to about 210 px on every paint), 9 % text, 5 % rounded paths.

**Thumbnails** (1.5 MP JPEGs and 1 MP PNGs, generated):

- Decode to the 384 px tier with the current path (Pillow, draft mode, Lanczos): JPEG 15 ms, PNG 29 ms. With
  `QImageReader.setScaledSize`: JPEG **3.9 ms**, PNG 19 ms.
- 140 cold decodes: 2.9 s on one thread, 0.92 s on four, 0.66 s on twelve.
- A cache file (JPEG, quality 88): 1.9 ms to write, 1.0 ms to read, 5.5 KB. WebP is a third of the size and 3.5
  times slower to write.
- One 384 px thumbnail is 384 KB in memory. The pixmap cache is capped at 400 MB, so it holds about 1,000; process
  memory went from 255 MB to **684 MB** after browsing 1,000 pictures and stayed there to 4,000.

**The same work done another way** (measured, to size the fixes):

| | 5k | 20k |
|---|---|---|
| Sort as one keyed list sort: name / caption length / tag count | 25 / 2 / 13 ms | 24 / 5 / 54 ms |
| Filter as one pass over the list | 6 to 42 ms | 24 to 161 ms |
| Build a tag index, then look a tag up | 28 ms, 0.8 ms | 118 ms, 3.8 ms |
| One directory pass for names, sizes, dates and which captions exist | 30 ms | 129 ms |
| Read the captions that exist: one thread / eight | 0.58 / 0.45 s | 2.9 / 2.0 s |
| Find changed captions by date instead of re-reading all | 23 ms | 92 ms |

### 5.2 Budgets

| What | Budget | Today |
|---|---|---|
| Window visible and painted | under 1.0 s from process start (files cached) | about 2.8 s plus the AI runtime preload |
| Gallery usable | under 1.5 s | the same |
| Folder of 10,000: names and skeleton tiles on screen | under 0.3 s | about 10 s (interpolated) |
| Folder of 10,000: browsable, pictures streaming in | under 1.0 s | about 10 s |
| Folder of 10,000: captions, counts and filters complete | under 2.5 s, with a count going up | about 10 s, frozen |
| Scroll, 220 px cards | 8 ms a frame at the median, 12 ms at the 95th percentile | 14 / 20 ms |
| Scroll, 100 px cards | 12 ms / 16 ms | 24 / 39 ms |
| Filter after the last key | 50 ms for text, 16 ms for a tag, at 20,000 | 230 to 920 ms |
| Sort | 100 ms at 20,000 | 1.7 to 5.1 s |
| Any tab switch | 50 ms | up to 4.6 s |
| Any click acknowledged (a pressed state, a spinner, a disabled button) | 100 ms | not held |
| Work on the UI thread without a visible state | 16 ms | not held |
| Longer than 1 s | a busy state; longer than 10 s: a count, a rate, time left and Stop | partly |
| Memory after browsing 20,000 pictures, torch not loaded | under 500 MB | 684 MB at the cap |
| Usable window | 1280 x 720 at 100 % | 1,471 px wide at least |

The scroll budgets are for scale 1 on this class of CPU. At 150 % a frame has 2.25 times the pixels; the budgets
must be re-measured on a real screen at 125 % and 150 %.

### 5.3 The changes that get there

Measured gains are marked M, estimates E.

1. **Coming back to the Gallery.** Compare caption dates in one directory pass on a worker thread (92 ms off the UI
   thread at 20k, M) and re-read only the files that changed, instead of all of them (4.5 s, M).
2. **Sort and filter without the proxy's Python callbacks.** Each browser keeps an ordered list of rows computed by
   one keyed `sorted` and one filter pass (24 to 161 ms at 20k against 1.7 to 5.1 s, M) and serves it through a thin
   list model. Sort keys are cached per mode and dropped when a caption changes; a tag index makes tag terms a
   lookup. Above a few thousand rows the pass runs on a worker with a generation number, so typing never waits.
3. **Hidden tabs do nothing.** Only the visible browser sorts and filters when a folder opens; the others mark
   themselves stale and compute when shown (removes four of the five sorts: 11 s of the 14 s at 20k, M by
   division).
4. **Open a folder on a worker.** Step one: the whole session is read on a thread (a directory pass, then captions
   on a small pool) while the grid shows skeleton tiles and a count; the finished session is handed to the model in
   one reset. Step two, later: names arrive first and the grid is browsable at once, captions follow in the order of
   what is visible, and the features that need every caption (tag filters, tag statistics, save) show a count until
   the read is complete.
5. **Thumbnail pipeline.**
   - Sizes: add a 192 px tier for cards up to 180 px (a quarter of the memory and decode of 384 px), keep 384 and
     768.
   - Decode with `QImageReader` and a scaled size (JPEG 3.9 ms against 15 ms, M), EXIF rotation through
     `setAutoTransform`; keep the Pillow path for what Qt cannot read or reads wrongly (16-bit files, unusual
     TIFFs) and for the pixel-count guard.
   - On disk: JPEG quality 85 in the existing hashed layout (1 ms to read back, M). No format change.
   - Workers: the pool of today (half the cores), newest request first as now, plus **cancellation**: when the
     visible range moves, queued jobs outside it and one screen around it are taken back out of the pool
     (`QThreadPool.tryTake`), and one screen ahead in the scroll direction is requested at low priority.
   - In memory, two levels: the compressed bytes of every thumbnail seen (about 5 KB each, so 20,000 pictures are
     about 110 MB, E) and decoded pixmaps only for the visible range and a few screens around it. That replaces the
     400 MB pixmap cache that holds 1,000 pictures.
6. **A delegate that paints from caches.** Keep the elided caption per card and width (removes the 32 %, M as a
   share); keep the font metrics; pre-render each mark as a small pixmap; hold a pixmap scaled to the card's exact
   size for the visible range so a paint is an unscaled blit. Target 0.2 ms a card (E). If that is not enough for
   the dense view, cache the whole painted tile for the visible range (about 15 MB) and blit it while scrolling (E).
7. **Start.** Build only the shell and the Gallery before the first paint and build the other tabs one per idle
   tick afterwards (first paint drops by most of the 0.56 to 0.88 s of constructors and their share of the
   re-styling, E); under a token theme apply the stylesheet after the first tab is built (0.6 s on the whole window,
   M); make the `cv2`, `qt_material`, keyring and `huggingface_hub` imports lazy (about 0.3 s, M as import time);
   show the window before the AI runtime is loaded (question 9: it interacts with the reason the preload exists).
8. **Selection and status.** Keep the counts (pictures, without caption, unsaved) as running totals instead of
   three passes over the folder on every selection change; coalesce selection signals with a zero-length timer; do
   not build a list of 20,000 keys for "select all" until an action needs it.
9. **Skeletons and states** wherever something takes longer than 100 ms (sections 3.6 and 4).
10. **Width.** Let the Train preset actions collapse into a menu, elide model paths, and lower the panels' minimum
    widths until the window is usable at 1,280 px; a test asserts the minimum width of every tab.

A small benchmark script that prints the numbers of 5.1 from generated data belongs in `tools/` so every stage can
be checked against this table.

---

## 6. Staged plan

Each stage ships on its own. The classic `dark_teal` look keeps working throughout: every painted colour goes
through `theme.color()` with the old literal in `LEGACY_COLORS`, every restyled widget keeps its `legacy_style`,
and structural changes that would alter the classic look sit behind the layout setting of question 1. Every stage
carries its tooltips, Help Center text and hotkeys (standing rule 7). Sizes: S a day or less, M a few days, L a
week or more.

| # | Stage | Size | Risk | Files | What the user sees | Tested offscreen by | Look at on a real screen |
|---|---|---|---|---|---|---|---|
| 0 | **Stop the freezes.** Caption change check by date on a worker; hidden browsers do not sort; elided-caption and metrics cache in the delegate; running counts; lazy imports; token stylesheet applied after construction; the benchmark script | S to M | low | `tabs/gallery.py`, `tabs/workspace/context.py`, `browser.py`, `delegate.py`, `core/image_utils.py`, `tabs/settings.py`, `tabs/caption.py`, `main.py`, `tools/` | Nothing new to look at; the Gallery comes back at once, a folder opens several times faster, the app starts about a second sooner | unit tests with counters (no full re-read on show, no sort in a hidden browser), the paint cache hit rate, the benchmark numbers before and after | scrolling feel at 100 % and 150 % |
| 1 | **The workspace joins the theme.** New tokens (3.2); the card of 3.7; the grid, toolbar, tag bubbles, inspector well, status line, health marks and memory strip read tokens; one filled button per toolbar | M | medium (the card is the most-seen thing) | `tabs/theme.py`, `tabs/workspace/delegate.py`, `browser.py`, `panels.py`, `core/widgets.py`, `tabs/memory_bar.py` | TagScribeR light is light everywhere; cards are tiles with clear selection; marks have meanings | contrast tests for new tokens; a pixel test that the classic theme's card is unchanged; renders of all three themes compared | both themes on two monitors; selection visibility at 100 px cards |
| 2 | **Sort, filter and open off the UI thread.** The ordered-list model (5.3.2), tag index, worker pass, folder open on a worker with skeleton tiles and a count (5.3.4 step one) | M | medium (every tab reads the proxy) | `tabs/workspace/model.py`, `browser.py`, `context.py`, `core/dataset_session.py`, tabs that call `proxy` | A 20,000-picture folder shows its grid in about a second and never freezes; filters answer as you type | the existing filter and sort tests against the new model, a property test (same order as the proxy for every sort mode), cancel-while-opening | a cold disk; a network drive |
| 3 | **The shell.** Rail, top bar (folder switcher, counts, command field, Undo / Redo / Save, job chips, memory meters), `JobCenter`, toasts, notification when a job ends unfocused; other tabs built after the first paint | M | medium | `main.py`, new `tabs/shell.py`, `tabs/workspace/jobs.py`, `tabs/memory_bar.py`, `tabs/common.py`, `tabs/theme.py`, `tabs/help_content.py` | The folder, the unsaved count and the running job are visible from every tab; 128 px more room | widget tests for the chips from fake jobs, shortcuts unchanged, the window's minimum width | the rail's legibility, the notification on Windows |
| 4 | **Thumbnail pipeline and painted caches** (5.3.5, 5.3.6) | M | medium | `tabs/workspace/thumbs.py`, `delegate.py`, `core/image_utils.py` | Pictures appear faster, dense grids scroll smoothly, memory stays flat | decode equivalence on generated JPEG / PNG / WebP / 16-bit / rotated files, cancellation counts, memory ceiling test | sharpness at 125 % and 150 %, colour-managed files |
| 5 | **Gallery workspace.** One filter row with tokens, selection bar, inspector (token meter, "This picture"), loupe, empty and first-run state, drop a folder | M | low | `tabs/gallery.py`, `tabs/workspace/browser.py`, `panels.py`, `tabs/help_content.py` | Batch actions are where the selection is; a new user is led to the first step | widget tests (selection bar counts, filter tokens round-trip to text), render review | drag and drop, the loupe |
| 6 | **Auto Caption as a guided run** with the live feed; review as a mode | M | medium | `tabs/caption.py`, `tabs/review.py`, `inference/worker.py` (signals only) | The Run button is always on screen; results appear as they are written | the caption simulator already in `tests/`, feed rows from fake signals | a real run with a local model and with a server (the owner) |
| 7 | **Train: ready check, calm form, dashboard** | L | medium | `tabs/train.py`, `tabs/train_dialogs.py`, `training/pipeline.py` (read-only checks) | A run you can read from across the room; warnings before Start instead of errors after | the tiny family's smoke run feeds the dashboard; widget tests per state (idle, caching, training, paused, finished, failed) | a real run (the owner); the window at 1,280 px |
| 8 | **The rest.** Palette with scopes and key caps; Settings as sections with theme swatches and density; Editor canvas; Datasets and Metadata lists; side sheets and confirm-in-place replacing message boxes; the tour | L, in pieces | low each | `tabs/palette.py`, `tabs/settings.py`, `tabs/editor.py`, `tabs/datasets.py`, `tabs/metadata.py`, `tabs/common.py` | The remaining tabs match the first four | per piece | per piece |

---

## 7. The mockup, and how each view maps to Qt

One self-contained HTML file with a dark / light switch and four views, delivered beside this document. It uses
only flat fills, hairlines, rounded rectangles, text, painted placeholder pictures and one shadow; there is no blur,
no CSS transition and nothing a widget cannot do. All names, paths and figures in it are made up.

| View | Widgets |
|---|---|
| **Shell** | Rail: checkable `QToolButton`s with text under the icon in a `QButtonGroup`; the accent bar and tint are QSS on `:checked`. Top bar: a `QHBoxLayout` of a menu button, chip buttons (QSS, full radius), a read-only line edit that opens the palette, tool buttons, the job chip (a small painted widget: text plus a 2 px gradient line) and two painted meters. |
| **(a) Gallery** | `QListView` in icon mode with the painted delegate of 3.7 (well, fitted pixmap, name, two caption lines, marks, tick, focus ring: all `QPainter`). Filter tokens painted in the line edit when unfocused. Segmented controls are checkable buttons in a group. The selection bar is a raised child widget with a painted shadow. The inspector is a `QTabWidget` with icons; the token meter and tag chips are small painted or flow-layout widgets (the flow layout exists in `core/widgets.py`). |
| **(b) Auto Caption, mid-run** | The queue is the same list view with a smaller card. Scope is a segmented control. The run panel is a `QVBoxLayout`: labels, chips, a `QProgressBar` styled by QSS, four tile frames, a `QListView` with a painted row delegate for the feed (the caret is a filled rectangle; Accept / Reject are hit-tested areas of the row or an index widget on the hovered row only), and a fixed footer. |
| **(c) Train dashboard** | A `QGridLayout` of box frames inside a scroll area, re-laid to one column when narrow. Stage nodes and tiles are labels and tiny painted dots. The chart is the existing `LossChart` with a legend row, a "now" rule and a boxed hover readout added to its `paintEvent`. The previews strip and the hard-image rows are painted list delegates. The recipe chips are labels in a flow layout. |
| **(d) First run** | A centred column in the Gallery's stacked widget, shown when no folder is open: labels, two buttons, a frame with a dashed QSS border that accepts drops, three label blocks, a `QListWidget` of recent folders, chip labels. |

What the mockup simplifies: the browser's text rendering is not Qt's (Qt's is a little heavier at small sizes, so
8.5 pt must be checked on a real screen); the placeholder pictures are gradients; the streaming caret appears only
for providers that stream.

---

## 8. Not rendered, not measured, not verified

- Nothing was seen on a real screen: no judgement here covers display scaling above 100 %, ClearType, a second
  monitor, real hover and focus behaviour or animation smoothness.
- The loss chart with data and the previews grid did not render in the offscreen run (the screenshot tool's fake
  run feeds console lines; the chart now reads the step log file, and the Train tab was being edited by another
  change at the time). The chart was judged from its code and from the tracked README screenshot.
- Device probing at start (`detect_devices`) and anything that needs a GPU context were not measured.
- Scroll timings are synchronous repaints of the viewport in a script, not frames on a display.
- The cold-disk open (19 s) is one observation.
- The batched layout of the list view after a filter change (50 to 415 ms of "relayout" in 5.1) was not broken down,
  and the list's scroll range did not grow as expected in one scripted run; both need a look before stage 2 fixes
  its design.
- Whether Qt reports the system's reduced-motion preference, and tabular figures on Linux fonts, were not checked.
- Gains marked E in 5.3 are estimates until the benchmark script confirms them.
- Linux and NVIDIA: nothing in this review was run there.

---

## Questions for the owner

Product decisions only. Each has a recommendation.

1. **New shell for everyone, or opt-in?** The rail and top bar change where things are in every theme, including
   the classic one. *Recommendation:* ship it as "Layout: Studio / Classic" in Settings, Classic as the default at
   first (as with the themes), and switch the default after you have used Studio for a while. The classic layout
   stays until you retire it.
2. **Navigation order.** Keep today's order and Ctrl+1 to 7, with Help and Settings at the foot of the rail?
   *Recommendation:* yes; learned shortcuts are worth more than a pipeline-ordered list.
3. **Memory strip.** In the Studio layout, show two small meters in the top bar by default and keep the full strip
   one click away, or keep the strip always? *Recommendation:* meters by default, strip on click and as a setting.
4. **Default theme.** Make TagScribeR dark the default for new installs once stage 1 here is verified on your
   screen (existing users keep what they chose)? *Recommendation:* yes, after you have looked at it.
5. **Control outlines.** Softer outlines by default with a "Stronger outlines" switch for accessibility, or 3:1
   outlines on everything as stage 1 has now? *Recommendation:* softer by default, the switch in Settings.
6. **Violet for the machine.** Use the logo's violet only for model-written and model-busy things (proposals, a
   picture being captioned, the captioning job), and remove it from the Auto Tag button and the health marks?
   *Recommendation:* yes.
7. **Demo pictures.** Add "Try the demo pictures" to the first screen, generating the demo set of the screenshot
   tool on the user's PC (no download)? *Recommendation:* yes.
8. **Resolution on every card.** Show the size pill only on the hovered card and in the "Details" view, instead of
   on every card always? *Recommendation:* yes, with "Details" remembered per tab.
9. **Window before the AI runtime.** Show the window first (about a second) and load PyTorch afterwards with
   "Loading the AI runtime" in the top bar, instead of a splash before the window? The preload exists because a
   first import on a worker thread once failed halfway, so this needs a careful design and your test on the real
   card. *Recommendation:* try it behind a setting, off by default, and decide from your runs.
10. **Train while running.** When a run starts, fold the form into "Recipe of this run" and show the dashboard,
    with a button to bring the form back? *Recommendation:* yes.
11. **Notifications.** A system notification when a job ends while the window is not active? *Recommendation:*
    yes, on by default, with a switch.
12. **The mockup.** Keep the HTML mockup in the repository (for example under `docs/dev/mockups/`) as the reference
    for the stages, or leave it out? *Recommendation:* keep it; it has no private data and no external requests.
13. **Message boxes.** Replace confirmations of undoable actions with confirm-in-place and a toast with Undo, and
    keep dialogs only where data would be lost? *Recommendation:* yes, tab by tab in stage 8.

## Order of work

1. Stage 0, stop the freezes (S to M): it needs no decision and every later stage is judged against its benchmark.
2. Stage 1, the workspace joins the theme (M): finishes what the themes started; unblocks question 4.
3. Stage 2, sort, filter and open off the UI thread (M).
4. Stage 3, the shell (M): after question 1 to 3.
5. Stage 4, thumbnails and painted caches (M).
6. Stage 5, the Gallery workspace and first run (M): after questions 7 and 8.
7. Stage 6, Auto Caption as a guided run (M).
8. Stage 7, Train (L): after question 10; verified only by a real run.
9. Stage 8, the rest, in pieces, as the port work allows.

Stages 0, 1 and 2 touch files no training port touches, so they can run beside the port plan.

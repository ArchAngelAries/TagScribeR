# TagScribeR: vision and evolution paths

Development notes, 2026-10-19. A read-only design report: nothing here is decided, scheduled or promised. The
tracked roadmap stays `PORT_PLAN.md`; where an idea below overlaps something PORT_PLAN or the README roadmap already
schedules, this file says so and only adds what is new. The standing rules of `CLAUDE.md` apply to every idea: keep
what works, Fizgig (https://github.com/shootthesound/Fizgig) stays the reference for training numerics, every
feature ships with help, nothing leaves the machine without the user's explicit action, and nothing is "verified"
until it has run on real hardware.

How to read an idea: **problem** (whose pain, in their words) / **feature** (concrete, what the user sees) / **why
it is new or better** / **builds on** (what in the codebase makes it cheap) / **needs** (packages, hardware runs,
decisions) / **risk**. Tiers: **Achievable** (days), **Substantial** (weeks), **Ambitious** (months). Effort
estimates are for one developer plus subagents and assume the port work in PORT_PLAN continues in parallel.

Where this report compares TagScribeR with other trainers (kohya_ss, OneTrainer, AI-Toolkit, SimpleTuner,
diffusion-pipe, musubi-tuner, taggui, the Civitai on-site trainer) it relies on general knowledge of those projects,
not on reading their current source. Those claims are marked *(unverified)* and should be checked before they are
repeated in user-facing text. Everything said about Fizgig comes from its v8.7.1 checkout (README, `docs/`,
`src/fizgig/web/`, the release notes).

---

## 1. Who the users are, and what nobody solves for them yet

TagScribeR and Fizgig attract the same people, with one difference: TagScribeR's users arrive with a *dataset
problem* first and a training problem second, and a disproportionate share of them are on AMD. Six personas, drawn
from the README's framing, Fizgig's release notes and the issue traffic they describe:

1. **The character / likeness trainer** (the largest group). Thirty to two hundred photos of one person or OC,
   wants a LoRA that looks like them and does not leak onto other people. Pain: does not know which of their photos
   are hurting the run; cannot tell epoch 14 from epoch 19 except by squinting; captions are inconsistent because
   three tools wrote them; the face is 60 px in half the photos; the trigger word turns out to mean something to the
   model already.
2. **The style trainer.** Hundreds to thousands of images, often scraped, often duplicated at several sizes with
   watermarks and text overlays. Pain: dedup at scale; "caption the content, never the style" discipline; balance
   across sub-styles; finding the epoch where the style is learned but the compositions have not collapsed.
3. **The booru / anime trainer** (SDXL, Illustrious, NoobAI, Pony, Anima). Tag-based captions, tag ontologies,
   quality tags, thousands of images; tools like taggui and BooruDatasetTagManager are their home *(unverified)*.
   Pain: tag consistency (synonyms, implied tags), per-tag balance, tag-to-trigger interactions, v-pred vs eps,
   knowing which tags the base already knows.
4. **The concept / edit / slider builder** (Fizgig's newest audience): before/after pairs, prompt sliders, edit
   LoRAs. Pain: pairing and aligning images, judging a slider's range, writing the pair captions.
5. **The AMD user on Windows** (TagScribeR's home ground): one card at its limit, no ComfyUI running alongside,
   wants honest VRAM numbers for *their* card, not NVIDIA tables. Pain: every other trainer treats them as an
   afterthought; "works on CUDA" is not a promise.
6. **The person with no usable GPU** (the Reddit request: "a web UI to drive training on a remote GPU"). A laptop,
   an iGPU, or a 6 GB card: can browse, caption through an API, curate, but not train. Pain: the whole pipeline
   assumes the dataset, the trainer and the eyes are on one machine. Also in this group: the owner of *two*
   machines (desktop with the card, laptop on the sofa) and the renter of pods by the hour who pays for every idle
   minute.

Across all six, the unmet needs that no current tool (TagScribeR, Fizgig, kohya_ss, OneTrainer, AI-Toolkit) solves
well, as far as this review can tell:

- **"Is my dataset ready?"** answered with evidence, not a checklist. Today the first honest feedback a user gets
  is Fizgig's loss watch at epoch 3, ten minutes into a paid pod hour. Faces too small at the chosen megapixels,
  captions that contradict the image, a trigger the model already knows, three near-duplicate framings that will
  dominate the run: all knowable before Start.
- **"Which epoch, and why?"** Fizgig's LoRA Royale and likeness score are the state of the art and still leave the
  user eyeballing a crossfade. Nobody measures *prompt adherence* (the first casualty of overtraining, as Fizgig's
  own README tip says), *memorisation* (a preview that is a training image), or lets two LoRAs be compared blind.
- **"What did this run actually learn from?"** The run and the dataset are not linked. Add ten photos, retrain,
  and nothing tells you what changed between the two LoRAs except your memory.
- **Captions as a system, not a text box.** Every tool writes `image.txt`. None keeps the layers apart (trigger,
  tags, prose, per-family template) so one curated dataset can caption itself for Illustrious and for Krea 2.
- **The GPU is somewhere else.** Fizgig solved "the app is somewhere else" (its web UI runs on the pod). The
  dataset work, though, still wants a desktop app with a fast grid, a real file system and no browser upload
  dialog. The desirable shape is: the *studio* at home, the *runner* wherever the card is.
- **Honest numbers for this card.** Memory tables are measured on one or two NVIDIA cards. The user needs "on your
  card, this run will take about N hours at M s/it and peak at P GB", measured on their card in 30 seconds.

---

## 2. Where TagScribeR stands

### Against Fizgig v8.7.1

**TagScribeR is ahead on** (verified by reading both code bases):

- *The dataset workspace.* One folder open in every tab, a virtualised grid for thousands of images, the
  search-style filter language (`tag:`, `missing:`, `res:`, `ar:`, `flag:`), saved filters per folder, batch tag
  editing with undo/redo, tag statistics with rename and merge. Fizgig's Captions tab is a paged list with a search
  box and Find & Replace.
- *Captioning.* Transformers 5 VLMs (Qwen3-VL, Qwen3.5, Gemma 4, JoyCaption), any OpenAI-compatible server (so
  GGUF quants and cloud APIs), WD v3 taggers with thresholds and blacklists, instruction presets, the review queue
  with a word-level diff. Fizgig captions with Qwen3-VL-4B or Florence-2 and overwrites in place.
- *Data safety.* Atomic writes with daily backups, Recycle Bin deletes, copies that never overwrite, keys in the OS
  keychain. Fizgig keeps its web password in plain text in `prefs.json` (documented as such).
- *Dataset health and metadata.* Exact and near duplicates, blur, low resolution, bucket preview; the privacy
  audit (GPS, serials, embedded prompts), lossless cleanup, authorship and camera presets, embedded A1111 / ComfyUI /
  NovelAI prompts to captions. Fizgig has nothing comparable.
- *The image editor and export.* Crop to bucket ratios with a before/after, metadata-preserving edits, kohya-style
  export. Fizgig's Image Prep resizes, converts to PNG and face-crops.
- *Training extras Fizgig dropped or never had:* fp8 as a selectable base, batch size above 1, NoobAI v-pred with
  zero-terminal-SNR, LoCon, Klein's extra timestep modes, the ROCm wide-attention VAE fix (measured: encode error
  0.04-0.06 down to 0.005 on an RX 7900 XT), the discrete-GPU pick on APU machines.
- *AMD on Windows as the primary tested platform,* and a native Qt desktop app (no browser, no WebView2 dependency,
  no port, no password).
- *Help.* Ctrl+K palette, F1 context help, a searchable Help Center with 26 topics, tooltips on every control,
  simple view versus "Show all settings".
- *Tests.* About 495 test functions that never touch the GPU.

**TagScribeR trails Fizgig on** (verified in Fizgig's docs and web layer; PORT_PLAN already schedules the ones
marked †):

- The web app: own window or browser, phone view with a QR login, LAN discovery of other machines running it
  (`web/lan.py`, UDP broadcast on 47810), the Pods tab (RunPod API, start/stop, "cheapest per run"), resumable
  chunked dataset upload (8 MB pieces), a pod file manager, auto-stop when a run finishes, systemd and
  reverse-proxy recipes, `--web` headless.
- The workbench: Studio (per-block sliders, donor blending, "I'm feeling lucky"), LoRA Royale (every epoch on one
  seed, crossfade, likeness, MP4/GIF exports), Profiler, Extract, the safetensors Metadata editor †.
- Training methods: full fine-tune †, DOP †, Multi Concept †, LoRA Isolation † (fused-token class decoupling,
  stranger composites, strided DOP), the trigger-word checker, Context LoRA (ported), training adapters (ported),
  resume at a new megapixel size, "Sample now / Save now" (ported in wave 2).
- Families: MOSS-Image †, Flux 3 Image Klein †, Z-Image Turbo †, ACE-Step (skipped by the owner), H3 clips, audio,
  voice, RefMods †.
- Judging: live ArcFace likeness in the sample gallery against three dataset photos, the per-epoch trend chart,
  the run visualiser, per-image residual charts in Runs & outputs; Runs & outputs itself (every run with notes,
  "Start again with these settings").
- Prep: InsightFace face crops with gender targeting †, the Look Consistency Filter † (ArcFace outliers that also
  feed the loss watch's warm-up), bilingual captions.
- Onboarding: "Download models for me" per family, a welcome tour, numbered tabs.

### Against the wider field *(unverified; general knowledge, not read for this report)*

- **kohya_ss / sd-scripts.** The widest family list (SD 1.5, SDXL, SD3, FLUX.1, video via musubi-tuner), textual
  inversion, DreamBooth, multi-GPU through `accelerate`, the largest optimizer catalog (Prodigy, DAdaptation, CAME,
  Adafactor), masked loss, and the kohya folder convention that half the community's guides assume. No dataset
  curation, no loss watch, a Gradio UI. TagScribeR trails on family breadth below the 2026 models, multi-GPU and
  the self-tuning optimizers (Fizgig removed them on purpose; keep that decision).
- **OneTrainer.** A strong desktop UI, per-concept settings and balancing, masked training, prior preservation,
  built-in captioning (BLIP / WD), multi-resolution, sampling during training. No per-image intelligence. Its
  concept-level configuration is the one thing TagScribeR should learn from (see theme B).
- **AI-Toolkit.** Ostris's trainer: FLUX, Krea, Qwen, Wan, DOP's origin, YAML configs, a web UI, a RunPod
  template. Research-grade recipes first, UX second; no dataset tools.
- **SimpleTuner, diffusion-pipe, musubi-tuner.** Headless, multi-GPU, DeepSpeed, video. The right comparison for
  the remote theme: they prove that a trainer can be entirely file-and-config driven. TagScribeR's child-process
  design is already in that family.
- **taggui, BooruDatasetTagManager, Hydrus.** Captioning and tag management, many VLMs (taggui), booru workflows.
  taggui's multi-model captioning and its tag autocompletion from booru lists are the bar for theme C.
- **Civitai's on-site trainer.** The zero-install path the Reddit user wanted; opaque settings, pays per run, the
  dataset is uploaded to a third party. The remote theme is the answer to "like that, but mine".

The combined picture: TagScribeR is the only one of these with a real dataset studio *and* a 2026-model trainer in
one app, and the only one built AMD-first. Its gaps are remote use, the post-training workbench, and the
pre-training evidence that the dataset is ready. The ideas below go after those three.

---

## 3. Evolution paths

Nine themes. Each starts with the problem it answers, then ideas by tier. Themes G (remote) and F (judging) are the
two the owner asked for specifically.

### Theme A. The dataset as a versioned, run-aware object

**Problem.** A dataset changes between runs and nothing records it. The README roadmap lists "dataset versions tied
to the run that used them"; the new part here is making the version a *manifest with content hashes* that the run
folder, the export, the publish card and the comparison tools all read, so "what changed" becomes a diff, not a
memory.

- **Achievable: dataset manifest and run stamp.** `core/dataset_session` can write `dataset.manifest.json` (per
  image: size, mtime, blake2b of the file, sha1 of the caption, the caption layers of theme C when they exist) into
  the run folder next to `dataset.json`, and a `tagscriber.dataset_hash` into the LoRA's metadata via
  `training/metadata.py`. Two runs then diff by manifest: "12 images added, 3 captions changed, 1 image replaced".
  *Builds on:* the run folder (`training/pipeline.build_run`), the SAI metadata writer, `core/health` already
  hashing files. *Needs:* nothing new. *Risk:* none to data; hashes of large PNGs cost seconds (the health scan's
  fingerprint trick in `IMPROVEMENT_IDEAS.md` A1 applies).
- **Achievable: "Open the dataset this LoRA was trained on."** Read the manifest from a `.safetensors` header (the
  reader exists for the Context LoRA path) and open the folder, showing which images have changed since. Nobody
  does this; it is the dataset half of Fizgig's "Start again with these settings".
- **Substantial: snapshots in Collections.** A collection (already a named set of images and captions in
  `tabs/datasets.py`) gains "Snapshot": a frozen copy of the captions and a manifest, not the images (copies never
  overwrite; images are referenced by hash, so a replaced file is detected). The Train tab trains a snapshot, and a
  snapshot can be exported, published or diffed against the live folder. *Builds on:* Collections, `core/export`.
  *Risk:* users expect snapshots to copy images; the UI must say what is frozen.
- **Ambitious: lineage graph.** Runs, snapshots, caption-fix events (the loss watch's `caption_updates_applied.json`
  already records them) and LoRAs as nodes; "this LoRA came from snapshot 3 after 2 caption fixes at epochs 4 and
  7". A view in the Train tab's run list. *Needs:* a small SQLite index in `user_data/`. *Risk:* scope creep; keep
  it to what the files already record.

### Theme B. Curation intelligence: "is this dataset ready?"

**Problem.** The first real feedback today is the loss watch at epoch 3. Most of its verdicts (viewpoint
mismatch, tiny faces, near-duplicates dominating a bucket) are knowable in seconds before Start, on the CPU. PORT_PLAN
schedules "dataset balance and caption consistency reports" and the "bucket preview and fit check"; the ideas here
add the evidence those reports need.

- **Achievable: training-aware Health.** The Health tab takes a family and target megapixels (defaults from the
  Train tab) and adds rows the trainer cares about: images that will be *upscaled* (`bucket_no_upscale` off) or
  shrunk by more than 2x; captions over the family's token limit (Krea 2's 512, SDXL's 225, read from
  `FamilyDescription`) counted with the family's own tokenizer (already cached offline for the text stage); caption
  files that are empty or trigger-only; the bucket histogram with "3 images alone in a 1536x640 bucket". *Builds
  on:* `core/health.make_buckets`, `training/dataset.py` bucket rules, the HF tokenizer cache. *Needs:* a torch-free
  tokenizer load (`tokenizers` is already a dependency of Transformers). *Risk:* none.
- **Achievable: subject size check.** A face / subject box per image, with "faces under 64 px at this megapixel
  target: 23 images". Fizgig's CLI guide explains why (a face at 80 px reaches the model as 10x10 latent pixels).
  PORT_PLAN stage 7 brings insightface for face tools; until then OpenCV's YuNet detector (a 230 KB ONNX, runs on
  the CPU through the existing `onnxruntime`) is enough for a box. *Why new:* no trainer warns before the run;
  Fizgig's answer is "add face crops", found in a doc. *Needs:* the YuNet model file (Apache-2.0) in the models
  folder, a note in THIRD_PARTY_NOTICES. *Risk:* anime faces need a different detector; say so in Help and let
  the check be skipped.
- **Achievable: "what the model sees."** A per-image preview of the *training input*: the image cropped to its
  bucket, resized, then round-tripped through the family's VAE (the latent is already cached; decode it). Users
  discover that their 4000 px photo trains as a 512 px soft crop with the hands cut off. *Builds on:* the latent
  cache, the VAE decoder every driver has, the Inspect panel. *Needs:* a small child-process tool
  (`python -m training.decode <cache file>`) so no torch enters the app process; the wide-attention fix applies on
  ROCm. *Why new:* no trainer shows the VAE round-trip; it is the most honest preview of what a run learns from.
- **Substantial: embedding index per folder.** A SigLIP or CLIP image embedding per image, computed once in the
  background on the GPU (or CPU for small sets), stored beside the thumbnail cache. It powers: near-duplicate
  detection far better than dHash (same subject, different crop), "find images like this", a text search box
  ("profile view, outdoors"), a *coverage map* (clusters with counts: "80 close-ups, 4 full-body, 0 from behind")
  and a diversity score the Health report quotes. The WD v3 taggers already compute a feature vector internally;
  whether SmilingWolf's ONNX exports expose it as an output is *unverified* and worth checking first, because it
  would make the index free for booru users. *Builds on:* `inference/` providers and `manager`, the thumbnail
  loader's priority pool, `core/query` (a `like:<image>` term). *Needs:* a model download (SigLIP 2 base is about
  400 MB), a sidecar store (`user_data/projects/<folder>/embeddings.npy`), `numpy` only for search. *Risk:* one
  more model to download; keep it optional and the health scan independent of it.
- **Substantial: balance and curriculum planner.** With clusters and tags in hand, a "Balance" panel proposes
  repeats per cluster (OneTrainer's per-concept repeats, but derived from the data), shows the effective share of
  each cluster per epoch, and writes it as `num_repeats` per sub-dataset into `dataset.json` (the format already
  supports several folders). Also "drop N near-duplicates from the largest cluster" as a reviewable change set.
  *Builds on:* `training/dataset.py`'s multi-folder config, `DatasetSession.transform` change sets. *Risk:* repeats
  change what a run learns; the panel proposes, the user applies, the run stamp (theme A) records it.
- **Substantial: regularisation set builder.** For SDXL-family users (prior preservation is their norm) and as a
  home for DOP-style class sets: generate N class images with the base model from a class prompt list, caption them
  with the same preset, and keep them as a separate folder marked `is_reg` (Fizgig's TOML has the flag; TagScribeR's
  `dataset.json` can carry it). *Builds on:* the preview renderer (`training/train.py:_render_previews`) lifted into
  a `python -m training.render` tool (also needed by theme F). *Risk:* it is a real GPU job; the memory bar and the
  queue must treat it as one.
- **Ambitious: the readiness score.** One page before Start: subject size, caption contradictions (theme C),
  duplicates, coverage, token limits, the fit check, with a traffic light each and one sentence of advice. The
  guided first run in the README roadmap would open with it. *Risk:* a score invites gaming; keep it a checklist
  with evidence, not a number to maximise.

### Theme C. Captions as a system

**Problem.** Captions are the lever that moves a run the most and the part of the pipeline with the least
tooling. PORT_PLAN schedules "caption consistency reports" and "find and replace across captions". The ideas below
go further: caption layers, verification against the image, and the loss log joined to the words.

- **Achievable: caption lint.** Rules over the whole folder, each with a fix: duplicate tags; tags that differ only
  by spelling or synonym (`blonde hair` / `blond hair`, a user-editable synonym list per project with a built-in
  seed list); contradictions (`1girl` with `2girls`, `solo` with `2boys`); the trigger missing, doubled or in the
  wrong position; a quality tag inside a prose caption; a caption that names the *style* in a style dataset (the
  Fizgig rule: for style, caption the content and never the look). *Builds on:* `core/captions` transforms, tag
  statistics, the project's subject word, the batch change-set machinery with undo. *Risk:* none to data (every
  fix is a reviewable change set).
- **Achievable: trigger-word check for every family.** Fizgig's checker exists for LoRA Isolation. TagScribeR can
  offer it for any run: tokenise the trigger with the family's tokenizer (does it split into meaningful pieces? does
  it share a first piece with another trigger in the project?), and show what the *base model already thinks the
  word means* by framing the existing `sample_at_first` epoch-0 preview as "before training" in the samples panel,
  with the trigger alone as one of the prompts. *Builds on:* the tokenizer cache, `sample_at_first`. *Risk:* none.
- **Substantial: caption layers and profiles.** Keep the layers apart on disk (`image.txt` stays the compiled,
  trainer-readable caption; `image.tags.txt`, `image.prose.txt` and the project's trigger are the sources) and
  compile `image.txt` from a *profile*: "trigger, tags" for Illustrious; "trigger + prose" for Krea 2; "prose,
  then tags as hints" for Qwen. Changing the profile recompiles; the sources are never lost; export and the run
  stamp record which profile was used. *Why new:* every tool stores one string; TagScribeR already generates both
  kinds (WD tags and VLM prose) and already passes tags as hints to the VLM. *Builds on:* `core/caption_io` (atomic,
  backups), `inference/prompts` (tag hints), `core/projects` (per-folder settings). *Needs:* a decision on sidecar
  names and on whether `image.txt` is always derived (recommendation: yes, with "edit the compiled caption"
  turning that image's layers off until re-enabled). *Risk:* two sources of truth if the rule is unclear; the Help
  topic must be written first.
- **Substantial: caption verification by VLM.** For each caption, ask the loaded VLM a structured question ("Does
  this caption match the image? List claims that are wrong: viewpoint, count of people, colours, setting.") and
  show the disagreements as a review queue, with one-click fixes. Fizgig's loss watch finds that
  "caption-viewpoint mismatches are the single most common thing" it convicts on Krea 2; this finds them before the
  run, with the VLM the user already has loaded. *Builds on:* `inference/worker` (batch, cancellable, failure
  tolerant), the review window's diff. *Needs:* a new instruction preset and a parser for the answer (JSON-mode
  where the provider supports it). *Risk:* VLMs are wrong sometimes; the queue is advisory and nothing is written
  without acceptance, as with every AI caption today.
- **Substantial: loss by tag.** Join the per-image loss log (`loss_log/per_image_loss.jsonl`, residual by epoch)
  with each image's tags: "images tagged `profile view` have a mean residual of +0.12 and 4 of 6 are stuck".
  Shown in the Problem Images window and as Gallery badges (theme E). *Why new:* Fizgig charts per image; nobody
  aggregates by concept, which is where the actionable insight is. *Builds on:* `training/loss_logger`'s residuals,
  tag statistics. *Risk:* small tag counts give noisy means; show n.
- **Ambitious: multi-model consensus captioning.** Run two or three captioners (a VLM, the WD tagger, an API
  model) and merge: tags both saw are kept, disagreements go to review. taggui lets you run several models; nobody
  merges with provenance. *Builds on:* providers with explicit capabilities, the provenance index idea in
  `IMPROVEMENT_IDEAS.md` I1. *Risk:* slow; make it a batch job with the usual cancel and save-as-you-go.

### Theme D. Fit and plan before you press Start

**Problem.** "Will it fit, and how long?" The README roadmap lists a fit check from measured tables. The new part:
measure on *this* card, in seconds, and turn the number into time and money.

- **Achievable: the 30-second probe.** `training.train --probe`: load the planned base precision, run two or three
  steps at the *largest* bucket with the real batch size and previews off, report peak allocated and reserved,
  s/it, and exit 0 (or the OOM). The Train tab shows "On this GPU: about 4.9 s/it, peak 17.3 GB of 20; 60 images x
  30 epochs = 2 h 27 m" and, when the run is remote (theme G), "about 2.9 USD at this pod's rate". *Builds on:* the
  child-process launcher, `training.progress` parsing, the memory bar's peak marker. *Needs:* the probe skips
  caching if caches exist and otherwise caches only the largest bucket's images first. *Why better than tables:*
  it is the user's own card, with their own desktop and driver. *Risk:* a probe is a real GPU job; it must obey the
  owner's "never run training in tests" rule (it is a user action, never automatic).
- **Achievable: time-boxed runs.** "Train for 2 hours" sets epochs from the probe's s/it, rounding down to whole
  epochs. For pod users this is the control that matters. *Builds on:* the probe. *Risk:* none.
- **Substantial: preset recommender from dataset facts.** Given counts, clusters, face sizes, caption style and the
  family, suggest a preset and explain each deviation ("25 images: Fizgig's Fast preset; faces are small, so 0.5 MP
  instead of 0.25; 300 steps per epoch is low, so 40 epochs"). Never changes a Fizgig preset; only picks and
  annotates. *Builds on:* presets per family, theme B facts. *Risk:* rules of thumb must be labelled as such and
  cite where they come from (Fizgig's docs give several).
- **Substantial: sweeps.** "Queue a sweep": vary one or two parameters (rank 8/16/32, LR min/max, megapixels) from
  the current settings, auto-named `<name>-r16-mp05`, into the existing queue, with a comparison sheet at the end
  (theme F). *Builds on:* the queue (`presets.load_queue`), the run folder per run. *Why new:* OneTrainer and
  kohya have no sweep; AI-Toolkit needs a YAML per run. *Risk:* disk (caches are shared per folder and family, so
  only checkpoints multiply).

### Theme E. The run as a living object

**Problem.** TagScribeR already lets the run be steered while it goes (pause, preview override, caption fixes at
the boundary, Sample now / Save now). The next step is to make the *dataset tabs* aware of the run, since the
workspace is shared.

- **Achievable: loss badges in the Gallery.** While a run goes (or afterwards, from the run folder), each image's
  verdict (`easy`, `suspect`, `stuck`, `exhausted`, `excluded`) and residual trend show as a badge on its card, and
  the filter language gains `loss:stuck` and `loss:>0.1`. Fixing the caption in the Gallery queues it through
  `pipeline.queue_caption_updates` exactly as the Problem Images window does. *Why new:* Fizgig's per-image charts
  live in a separate Runs page; here the dataset itself lights up. *Builds on:* `read_problem_images`, the
  delegate, `core/query`. *Risk:* none.
- **Substantial: add or remove images mid-run.** The loss watch already re-encodes a fixed caption at the boundary.
  Extend the boundary protocol with `loss_log/dataset_updates.json`: images added (cache them at the boundary, join
  the next epoch), images removed (drop from the sampler), repeats changed. The epoch count per image is recorded
  so the run stamp stays honest. *Builds on:* `Watch.boundary`, `cache.py`'s incremental re-cache, the shuffle at
  epoch start. *Needs:* care with resume states (the sampler's RNG order changes when the set changes; record the
  change in the state). *Risk:* this changes what a run learns; opt-in, logged, and a question for the owner before
  it exists (see section 5).
- **Substantial: notifications.** On epoch end, plateau banner, problem image, failure, finish: a desktop
  notification, and optionally a user-entered webhook URL (ntfy, Discord, Slack, Home Assistant) with a short
  message and the latest preview thumbnail. Off by default; the URL is the user's explicit action; nothing else is
  sent. *Builds on:* `training.progress` events in the Train tab. *Why new:* Fizgig's phone view requires opening
  the app; a push is what a pod renter actually wants (it is also how they know to stop the pod). *Risk:* the
  thumbnail leaves the machine; make the thumbnail a separate tick.
- **Ambitious: the run timeline.** One scrollable view per run: loss and LR, epochs, previews, caption fixes,
  adaptive-LR rollbacks, pauses, dataset updates, notes; click an event to see what changed. The data is all in
  the run folder today (`.tagscriber_loss_*.jsonl`, `sample/.prompts_*.json`, `caption_updates_applied.json`,
  `run.log`). *Risk:* UI effort; the value is in making the existing files legible.

### Theme F. Judging and comparing results

**Problem.** Fizgig's LoRA Royale and likeness scoring are the bar. PORT_PLAN stage 7 schedules face likeness and
LoRA comparison; the ideas here add the measures and workflows that go beyond "does it look like them".

- **Achievable: the render tool.** `python -m training.render --family krea2 --lora <file> --prompts <json> --seed N`
  rendering with any LoRA (or a folder of epoch checkpoints) through the family's own preview recipe, in a child
  process, writing PNGs with the prompt and settings in their metadata. Everything else in this theme is a UI over
  this one tool. *Builds on:* `_render_previews`, the LoRA reader (kohya, OneTrainer, AI-Toolkit, LoHa, LoKR), the
  Turbo / Distilled preview paths. *Needs:* a Compare tab or a section of the Train tab. *Risk:* it is a GPU job
  while a run may be going; the queue owns the GPU.
- **Achievable: canary prompts, measured.** Fizgig's README tip (give one sample prompt something the dataset does
  not have; the last epoch that still follows it is a good one) becomes a feature: mark one or more preview prompts
  as *canaries*, and after each epoch ask the loaded VLM a yes/no checklist derived from the prompt ("Is the subject
  seen from a low angle? Is she looking over her shoulder?"). Plot adherence per epoch beside the loss; the first
  epoch where adherence drops is the *overtraining estimate*. *Why new:* no trainer measures prompt following; it
  is the earliest and most reliable overtraining signal in practice, and TagScribeR already has the VLM. *Builds
  on:* `inference/` providers, the per-epoch preview files and `read_preview_prompts`. *Needs:* the VLM loaded in
  the app process while training runs in the child (VRAM: use the API provider or a small model; Help must say so).
  *Risk:* VLM errors; show the questions and answers, not only the score.
- **Achievable: memorisation check.** For each preview, the nearest training image by embedding (theme B's index)
  or by dHash as a fallback; a similarity that jumps toward 1.0 means the LoRA is reproducing a training image.
  Shown as a per-epoch line and a side-by-side. *Why new:* nobody does it; it catches the "the LoRA only draws my
  photo 12" failure that likeness scoring rewards. *Risk:* near-duplicate datasets trigger it for the right
  reason; say so.
- **Substantial: epoch sheet and keeper workflow.** A grid of epochs x prompts on one seed (the Royale idea, as a
  sheet rather than a crossfade), the likeness (stage 7), canary adherence and memorisation lines above it, a
  suggested keeper with the reasons, and "Promote epoch N": rename it to the final name, embed the preview as its
  thumbnail (`training/metadata` already does this), write the keeper decision into the run stamp, and offer to
  move the other epochs to the Recycle Bin or a `discarded/` folder. *Builds on:* `pipeline.checkpoints`,
  `samples`, metadata thumbnails. *Risk:* none to data (Recycle Bin).
- **Substantial: blind A/B.** Two LoRAs (two epochs, two runs, a TagScribeR run against a kohya file), same seeds
  and prompts, shown shuffled in pairs; the user picks; an Elo-style score accumulates; the reveal shows which was
  which. It is the honest way to answer "did rank 32 help?" and "is my new dataset better?". *Why new:* no trainer
  has a blind test. *Builds on:* the render tool. *Risk:* none.
- **Substantial: run diff.** Two run folders side by side: settings diff (`train_config.json`), dataset diff (theme
  A manifests), loss curves overlaid, epoch sheets aligned by epoch. The sweep (theme D) ends here. *Builds on:*
  everything already in the run folder. *Risk:* none.
- **Ambitious: "what you see is what you get" export.** For each family, write a ComfyUI workflow JSON (or an
  A1111-style settings block) that reproduces the preview recipe exactly (sampler, scheduler, steps, CFG, Turbo LoRA
  strength, seed, size) with the chosen LoRA, so the user's first render outside TagScribeR matches the epoch they
  picked. Fizgig documents the ComfyUI settings in prose for MOSS-Image; shipping the workflow is the next step.
  *Needs:* per-family node graphs, kept current with ComfyUI's loaders; a Help note that it is a starting point.
  *Risk:* maintenance as ComfyUI changes; mark versions.

### Theme G. Remote, headless, cloud and multi-machine training

**Problem.** The Reddit request, the two-machine owner, and the pod renter all need the same split: the *studio*
(grid, captioning, curation, judging) where the user sits, the *runner* (cache, train, render) where the GPU is.
TagScribeR's process model is already that split in miniature: the app never trains; it freezes a run folder and
starts `training.cache` and `training.train` as children controlled by files. Making the children run on another
machine is a transport problem, not a trainer problem. This is the strongest architectural advantage TagScribeR
has over Fizgig here: Fizgig's web app runs *all* of itself on the remote machine, so the dataset work happens in a
browser; TagScribeR can keep the desktop studio and move only the GPU work.

Privacy and honesty constraints for the whole theme: every transfer is the user's explicit action; no discovery
packets, uploads or callbacks happen by default; the runner never listens beyond localhost until a token is set;
plain HTTP is for LAN and SSH or Tailscale tunnels only (say so in Help, as Fizgig does); nothing here can be
verified on a pod by the owner's usual workflow, so each stage ships as "unit-tested; a LAN run between two
machines confirms it".

- **Achievable: the headless runner CLI.** `python -m training.pipeline build <values.json> <folder>` writes a run
  folder exactly as the Train tab does (it already calls `build_run`); `python -m training.run <run_dir>` runs the
  three stages in order, honours the same control files, and streams the same progress lines. With `--probe` from
  theme D. Document it for Linux servers and `systemd` the way Fizgig's `WEB_UI.md` does. *Builds on:*
  `pipeline.build_run`, `train_kwargs`, `progress`. *Needs:* a page in the Help Center and `docs/`. *Why first:* it
  is the contract every remote mode speaks, it is nearly written, and a power user with SSH can use it on a pod
  tomorrow (copy the run folder and the dataset, run it, copy the results back).
- **Achievable: portable run bundle.** "Export run bundle" zips the run folder's `dataset.json`, `train_config.json`,
  prompts and the dataset (images and captions; or caches when they are already computed and the user prefers to
  skip the remote VAE), rewriting the folder paths to relative ones; `training.run --bundle` unpacks it and
  rewrites them back. The results (checkpoints, samples, loss logs) come back the same way. This is the
  Reddit user's path today with any pod and no server code at all. *Builds on:* `cache_dir_for`, relative path
  support in `dataset.json` (needs a check). *Risk:* text caches are large today (31 MB per Krea 2 caption until
  `IMPROVEMENT_IDEAS.md` T3 lands); images are usually smaller than caches, so the default is to ship images.
- **Achievable: captioning on a remote GPU, documented.** It already works: run LM Studio, llama-server or vLLM on
  the machine with the card and point the OpenAI-compatible provider at it over LAN or a tunnel. A Help topic and a
  profile preset ("Remote llama-server") make it a feature. *Risk:* none.
- **Substantial: the runner service.** A torch-free HTTP service (`training.runner --serve`, standard library or
  `aiohttp`, one file) on the GPU machine: list runs, receive a run folder and dataset (chunked, resumable,
  manifest-checked: Fizgig's 8 MB pieces are the right design), start / pause / stop, stream progress (server-sent
  events), serve samples and checkpoints, accept the control files (`.sample_override.json`,
  `caption_updates.json`), report the GPU and free VRAM through `core/hardware` and `core/vram_monitor`. Token
  auth, localhost-only until a token is set, no TLS of its own (tunnels and reverse proxies, as Fizgig documents).
  *Builds on:* the run folder protocol is the API; the service only moves files and relays stdout. *Needs:* a
  security review before it ships (the owner's audits 08-10 set the bar), rate limits on the token, path
  validation on every file operation (the `run_dir_for` fix is the precedent). *Risk:* the biggest attack surface
  the app would have; keep it tiny and off by default.
- **Substantial: "Train on" in the Train tab.** A selector: *This computer* or a saved runner (name, address,
  token). Start builds the run folder locally, pushes it and the dataset (or bundle) with a progress bar, starts it
  remotely, and the whole Train tab (console, loss chart, samples, Problem Images, preview override, caption fixes
  at the boundary, pause and resume) works unchanged because it reads the same files, fetched by the client as they
  change. Results pull back to the local output folder on finish or on demand. The queue gains a column for the
  runner, so a sweep can spread over two machines. *Builds on:* everything in `tabs/train.py` that already reads
  the run folder (`read_step_log`, `samples`, `read_problem_images`). *Why better than Fizgig:* the dataset stays in
  the desktop studio with its grid, filters, health and editor; only the GPU work moves. *Risk:* two copies of the
  run folder; the client must treat the runner's as the truth while the run goes.
- **Substantial: the pod image and template.** A Docker image with the `training/` package, the runner, the
  installer's CUDA path and the model fetcher (PORT_PLAN's "Get" buttons already know the repos), published for
  RunPod and Vast with a template that sets the token from an environment variable and prints the connect address.
  The Train tab's runner dialog accepts a RunPod pod id and reads its proxy address (Fizgig's `pods.py` shows the
  GraphQL calls; TagScribeR must write its own). A "stop the pod when the run finishes" tick with a countdown, and
  the theme-D cost estimate. *Needs:* a CUDA machine to test the image (the owner cannot; say so), a volunteer
  tester, a decision on referral links (recommendation: none, or disclosed). *Risk:* the AMD owner cannot verify
  the CUDA image; NVIDIA testers are a prerequisite, as the README already asks for.
- **Substantial: the phone page.** The runner serves one static page: progress, ETA, loss sparkline, the latest
  previews, Pause and Stop, behind the token. No framework, one HTML file. *Builds on:* the runner's SSE. *Risk:*
  none beyond the runner's.
- **Substantial: resume elsewhere.** A run paused on the laptop's small card resumes on the desktop or a pod: the
  state dir, caches and run folder travel in a bundle; `--resume` continues the LR curve, as upstream does. *Builds
  on:* `_save_state` / `_load_state`, the bundle. *Risk:* the base precision may differ between cards (INT8 on one,
  NF4 on the other); the trainer must refuse or warn (an open question in section 5).
- **Ambitious: TagScribeR Web, the thin studio.** For the user with *no* desktop GPU and no wish to run a Qt app:
  a browser front end to the workspace served by the runner, with the grid, captioning (through the runner's GPU),
  tag editing, health and the Train tab. The `core/` layer is Qt-free by design, so the model, filters and change
  sets are reusable; the UI is a rewrite. This is the one path on which TagScribeR would resemble Fizgig's shape,
  and it is the last to build: the runner plus the desktop studio already serve the two-machine and pod cases.
  *Risk:* a second UI to keep in parity with the Qt one; only worth it if the no-GPU audience proves large.
- **Ambitious: LAN discovery.** Fizgig's UDP broadcast (`web/lan.py`) is small and standard-library; TagScribeR can
  implement its own (not copy GPL-incompatible code; Fizgig is Apache-2.0, so copying with attribution is allowed,
  but an original implementation is tiny). Off by default, opt-in on both sides, never across the internet.

### Theme H. Publishing and provenance

**Problem.** The README roadmap lists a "publish helper". The new part: the model card is *generated from the run*,
the privacy audit is reused on the samples, and publishing is a single explicit action.

- **Achievable: the model card from the run.** From the run folder and stamp: trigger word(s), family, base model
  file names and the Turbo/preview recipe, rank/alpha/LR/epochs/MP, the keeper epoch and why (theme F), dataset
  *summary* (counts, clusters, caption style; never file names or images), licence, sample images with their prompts
  and seeds, and the SAI metadata already embedded. Markdown for Civitai and Hugging Face. *Builds on:*
  `training/metadata.py`, the run stamp. *Risk:* none; nothing is uploaded.
- **Achievable: pre-publish audit.** Run `core/metadata.privacy_findings` on every sample image and on the
  thumbnail, strip EXIF from samples, check that no caption or prompt in the card contains a path, and show the
  result before the user copies anything. *Builds on:* the Metadata tab's audit and lossless strip. *Risk:* none.
- **Substantial: Hugging Face upload.** `huggingface_hub` upload of the LoRA, card and samples to the user's repo,
  token in the keychain (as API keys are today), with a dry run that lists every file first. Civitai's upload API
  is less documented *(unverified)*; a "prepare a Civitai post" (card, samples, trigger words and recommended
  strength in Civitai's fields) that opens the browser is the honest first step. *Risk:* API changes; keep the
  dry run and never upload without the listing.
- **Substantial: dataset cards and dataset publishing.** For users who share datasets: the same card for a
  snapshot (theme A) with its manifest, licence and the privacy audit over every image. *Risk:* the audit cannot
  catch every face or location; Help must say what it checks.

### Theme I. The image editor and prep (beyond the roadmap)

The README roadmap already lists the cleanup brush, background removal, smart crop to the subject at bucket ratios,
upscaling and batch recipes; PORT_PLAN stage 7 brings face crops. Additions:

- **Achievable: crop variants as augmentation.** From one photo, generate the tight face crop, the half-body and
  the full frame as separate training images with derived captions ("close-up portrait" prefixed), as a reviewable
  batch; the Fizgig CLI guide's "40x the face area" advice as a button. *Builds on:* the editor's crop and
  caption-preserving copies, the subject box from theme B. *Risk:* near-duplicates by design; the health scan must
  know they are intentional (a `variant_of` note in the manifest).
- **Substantial: pair tools for edit and slider LoRAs.** Side-by-side alignment check (same size, same framing),
  auto-pairing by name or by embedding similarity, pair captions written once and mirrored, a "difference" overlay
  so the edit the LoRA will learn is visible. *Builds on:* `pipeline.edit_pairs`, `pair_problems`. *Risk:* none.
- **Substantial: the training-time augmentation preview.** Where a family supports flips or caption dropout, show
  the user what fraction of steps see each, with examples. *Risk:* none.

### Theme J. Onboarding, platforms and the plug-in surface

- **Achievable: "Explain this setting, for this dataset."** The Help Center's `train_*` topics made
  context-aware: hovering a parameter shows the general help *and* what it means for the open folder ("60 images x
  1 repeat = 60 steps per epoch at batch 1; 30 epochs = 1800 steps"). *Builds on:* `help_content`, the dataset
  session. *Risk:* none.
- **Achievable: an AMD diagnostics page.** The owner's machine is the one where every ROCm quirk is known: a
  Settings page that checks the driver version, the pip ROCm SDK path, `HIP_VISIBLE_DEVICES`, the allocator config
  in force, the wide-attention fix status, and the T0 probe numbers from `IMPROVEMENT_IDEAS.md`, with "Copy report".
  *Risk:* none.
- **Substantial: family plug-ins from a folder.** `training/families/` is already one package per family plus one
  line in `registry.py`; let `user_data/families/` be scanned too, with a signed-off "load third-party family code"
  tick (the `trust_remote_code` precedent), so community families (Fizgig's driver docs invite the same) can be
  tried without a release. *Risk:* arbitrary code; the tick must be explicit and per family.
- **Substantial: localisation.** Fizgig has a community Korean translation as an add-on; Qt's `tr()` and `.ts`
  files make it native here. The help content is the bulk of the strings. *Risk:* maintenance.
- **Ambitious: project templates and the guided first run** (README roadmap) built on themes B and D: "Character
  LoRA on Krea 2" walks from folder to keeper epoch with the readiness page, the probe, a preset and the epoch
  sheet, each step explaining what it checks.

---

## 4. What to do first, and why

Ordered by value divided by effort, and by how much each unlocks later work. None of these touches training
numerics; all of them are unit-testable on CPU with the tiny family or generated images.

1. **The headless runner CLI and the portable run bundle (G, achievable).** Days of work; it is nearly all there in
   `training/pipeline.py`. It gives the Reddit user and every pod renter a working path at once ("build at home,
   run on the pod, bring results back"), it is the contract every later remote mode speaks, and it makes the
   trainer usable from `systemd` on a Linux box, which NVIDIA and Linux testers (whom the README asks for) will use.
2. **The render tool (F, achievable).** One child-process command that renders any LoRA or epoch folder with the
   family's own preview recipe. Every judging feature (epoch sheet, blind A/B, canary scoring, run diff,
   regularisation images, the WYSIWYG export) and PORT_PLAN's stage-7 LoRA comparison sit on it.
3. **The 30-second probe and time-boxed runs (D, achievable).** The README's fit check, made honest for the user's
   own card, with ETA and (for remote runs) cost. It also produces the numbers `IMPROVEMENT_IDEAS.md` T0 asks for.
4. **Loss badges in the Gallery and loss by tag (E and C, achievable).** The cheapest "nobody else has this"
   feature: the shared workspace plus the per-image loss log already on disk. It makes the loss watch visible where
   the user fixes things.
5. **Training-aware Health and the subject-size check (B, achievable).** Token limits per family, upscale warnings,
   lonely buckets, small faces: the pre-run evidence that saves pod hours and first-run disappointment.
6. **Caption lint and the trigger-word check (C, achievable).** Reviewable change sets over existing transforms;
   the trigger check reuses the tokenizer cache and the epoch-0 preview.
7. **Canary prompts, measured (F, achievable).** The earliest overtraining signal, using the VLM the app already
   has; it turns Fizgig's best tip into an instrument.
8. **The dataset manifest and run stamp (A, achievable).** Small, and the foundation of run diff, snapshots, the
   model card and "open the dataset this LoRA came from".
9. **The runner service and "Train on" (G, substantial).** The real answer to the two-machine owner and the pod
   renter, after 1 has proven the protocol and a security review has set the rules.
10. **Caption layers and profiles (C, substantial).** The one idea that makes a single curated dataset serve the
    booru families and the prose families without rewriting captions; it needs the owner's decision on the
    sidecar rule first.

Deliberately *not* first: TagScribeR Web (only after the runner shows demand), the embedding index (valuable but a
new model download and a new store; let the health work land first), LAN discovery (a nicety once the runner
exists), and anything in PORT_PLAN's port stages, which keep their order.

---

## 5. Questions for the owner

1. **Remote scope.** Is the target (a) power users with SSH and a pod (the CLI and bundle are enough), (b) the
   two-machine home user and pod renter inside the Train tab (the runner service), or (c) the no-GPU user who never
   installs the desktop app (TagScribeR Web)? The recommendation is (a) now, (b) next, (c) only on demand.
2. **Pod providers and referral links.** Fizgig's Deploy links carry a referral code. Should TagScribeR's pod
   template (if any) carry one, disclosed in the README, or none? Recommendation: none, or disclosed.
3. **Network code and the public repo.** A runner service is new attack surface. Is a security review (the audit
   08-10 style) a precondition for merging any listening code to `main`? Recommendation: yes, and localhost-only
   until a token is set.
4. **Caption layers.** May `image.txt` become a *compiled* caption (from trigger, tags and prose layers) with the
   sources kept in sidecars, or must `image.txt` stay the only caption file a user ever edits? This decides theme
   C's largest idea.
5. **Mid-run dataset changes.** Adding and removing images at an epoch boundary changes what a run learns and how
   it resumes. Fizgig does captions only. Should TagScribeR go further (opt-in, logged in the run stamp), or hold
   the line at captions?
6. **Resume across cards.** When a state saved on INT8 is resumed on a card that plans NF4 (or the reverse), should
   the trainer refuse, warn and continue, or force the saved precision (and block swap) even if it is slow?
7. **The VLM during training.** Canary scoring and caption verification want a VLM while the child trains. On a
   20 GB card that means the API provider or a small model. Is "use the API source or a small model while a run
   goes" an acceptable rule for Help, or should these features queue until the run ends?
8. **Face detection before stage 7.** Is a lightweight CPU face detector (YuNet via the existing `onnxruntime`)
   acceptable now for the subject-size check, with insightface replacing it when the face tools land, or should
   the check wait for stage 7?
9. **The embedding index.** SigLIP / CLIP means another model download (about 400 MB) and a per-folder store. Worth
   it for dedup, search and coverage, or should the health scan stay hash-based?
10. **Notifications.** A user-entered webhook (ntfy, Discord) sends a message and optionally a thumbnail off the
    machine. Acceptable as an explicit, off-by-default setting with the thumbnail as a separate tick?
11. **What TagScribeR is not.** The owner skipped ACE-Step (music). Should video (H3 clips, stage 6) stay in scope
    for the dataset tools too (clip grid, scene cuts, like Fizgig's Gizmo), or is TagScribeR an *image* studio that
    trains H3 stills only? The answer shapes themes B, C and I.

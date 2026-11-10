# Naming and credit: TagScribeR is its own program

The owner's decision (2026-11-10). TagScribeR is an independent project. It is not affiliated with, endorsed by or a
clone of Fizgig (https://github.com/shootthesound/Fizgig). Parts of its training engine are adapted from Fizgig's
code under the Apache License 2.0, and Fizgig stays the reference for which models and training features are worth
supporting. What a user sees and reads is TagScribeR's own.

## Rules

1. **Names a user sees are TagScribeR's.** Feature labels, built-in preset names, window titles, button texts and
   help headings that Fizgig coined get TagScribeR's own names (the table below). Terms the whole field uses stay:
   LoRA, LoKR, rank, alpha, EMA, adaptive learning rate, slider LoRA, edit LoRA, fine-tune, training adapter, Turbo
   LoRA, block swap, INT8, NF4, timestep, CFG.
2. **Fizgig is named in one place.** The Help Center's *Credits and licences* topic, the README's acknowledgements,
   `THIRD_PARTY_NOTICES.md`, and the source-file headers of adapted code. Tooltips, help prose, console and log
   messages, dialogs and error messages do not mention Fizgig.
3. **The licence notices are not optional.** Every adapted file keeps its header (source file, copyright, licence,
   changes made) and its row in `THIRD_PARTY_NOTICES.md`. Code comments and docstrings may cite the upstream file and
   line a piece of logic came from: they are provenance for developers, not user text. Do not remove them.
4. **Whose measurement is it.** A speed, memory or quality figure that TagScribeR did not measure says so without the
   name: "measured upstream on NVIDIA cards, not by TagScribeR" (or "an upstream figure"). Never present such a figure
   as TagScribeR's own, and never call a preset "measured" or "tested" unless it was (standing rule 9).
5. **Hidden setting keys stay** (`FAMILY_FT`, `KREA2_LOSS_WATCH`, `H3_*` ...) and so do the stored option values of
   choice settings. Users never see the keys; keeping them means a preset file written by Fizgig still imports. In the
   UI that is "Import a preset file" / "preset files from other trainers that use the same setting names"; the Credits
   topic says which.
6. **Files TagScribeR writes carry TagScribeR's names.** Metadata keys in saved models are `tagscriber_*` (the older
   `fizgig_*` keys are still read). Sidecar and scratch files are `.tagscriber_*`.
7. **Third-party file names are facts.** A model file published under a name that contains another project's name
   (for example a training adapter's file name on Hugging Face) keeps that name: it is what the user downloads.

## Rewording guide (tooltips, help, messages)

| Instead of | Write |
|---|---|
| "as Fizgig does" / "as in Fizgig" | nothing (state what TagScribeR does), or "as the reference implementation does" only where the reader needs to know it is not arbitrary |
| "Fizgig's presets" / "Fizgig's recipe" | "the built-in presets" / "the built-in recipe" |
| "Fizgig measured X" | "X (measured upstream on NVIDIA, not by TagScribeR)" |
| "ported from Fizgig" | "adapted from open-source training code (see Credits and licences)" |
| "Fizgig's default" | "the default" |
| "Fizgig preset files work as they are" | "preset files from other trainers that use the same setting names import as they are (see Credits and licences)" |
| "TagScribeR keeps this; Fizgig dropped it" | describe the feature only |
| "the one Fizgig uses" | describe the file or version only |

## Built-in presets

Built-in presets start with `◆ ` (not `✨ `) and do not repeat the family's name (the list is per family). Presets
that are TagScribeR's own starting points, not measured by anyone, say "Community".

| Family | Old name | New name |
|---|---|---|
| Krea 2 | ✨ Krea 2 Ultra Fast (rank 8, adaptive LR) | ◆ Quick start (rank 8, adaptive LR) |
| Krea 2 | ✨ Krea 2 Standard (rank 32, full model) | ◆ Balanced (rank 32, full model) |
| Krea 2 | ✨ Krea 2 Style (rank 16, gentle LR) | ◆ Style (rank 16, gentle LR) |
| Krea 2 | ✨ Krea 2 Slider (rank 8, 2e-4) | ◆ Slider (rank 8, 2e-4) |
| Qwen Image 2.1 | ✨ Qwen 2.1 Fast (rank 8, adaptive LR) | ◆ Quick start (rank 8, adaptive LR) |
| Qwen Image 2.1 | ✨ Qwen 2.1 Fast Identity Mode (rank 8) - very close to full-model likeness, ~1.5x faster | ◆ Identity blocks only (rank 8, faster steps) |
| Qwen Image 2.1 | ✨ Qwen 2.1 Standard (rank 16, adaptive LR) | ◆ Balanced (rank 16, adaptive LR) |
| Qwen Image 2.1 | ✨ Qwen 2.1 Style (rank 16, 1.5e-4) | ◆ Style (rank 16, 1.5e-4) |
| Qwen Image 2.1 | ✨ Qwen 2.1 Edit (rank 8, adaptive LR) | ◆ Edit pairs (rank 8, adaptive LR) |
| Qwen Image 2.1 | ✨ Qwen 2.1 Edit Strong (rank 16, adaptive LR) - trickier edits | ◆ Edit pairs, harder edits (rank 16, adaptive LR) |
| Qwen Image 2.1 | ✨ Qwen 2.1 Slider (rank 8, 2e-4) | ◆ Slider (rank 8, 2e-4) |
| MiniMax H3 | ✨ MiniMax H3 Fast (LoRA 8, 50 epochs) | ◆ Quick start (rank 8, 50 epochs) |
| MiniMax H3 | ✨ MiniMax H3 (rank 16, 60 epochs) | ◆ Balanced (rank 16, 60 epochs) |
| MiniMax H3 | ✨ MiniMax H3 Style (LoRA 8) | ◆ Style (rank 8) |
| MiniMax H3 | ✨ MiniMax H3 Slider (rank 8, 2e-4) | ◆ Slider (rank 8, 2e-4) |
| FLUX.2 Klein | ✨ Old Reliable (rank 16, full model, single subject) | ◆ Single subject (rank 16, full model) |
| FLUX.2 Klein | ✨ Old Reliable - Flavour 8 (rank 8, full model, single subject) | ◆ Single subject, lighter (rank 8, full model) |
| FLUX.2 Klein | ✨ Identity (rank 8, single subject) | ◆ Identity blocks (rank 8, single subject) |
| FLUX.2 Klein | ✨ Identity (rank 8, harder dataset) | ◆ Identity blocks (rank 8, harder dataset) |
| FLUX.2 Klein | ✨ Multi-Character (rank 16, multi character or concept) | ◆ Several subjects or concepts (rank 16) |
| FLUX.2 Klein | ✨ Style (late timesteps) | ◆ Style (late timesteps) |
| FLUX.2 Klein | ✨ Style+Composition (all timesteps) | ◆ Style and composition (all timesteps) |
| MOSS-Image | ✨ MOSS-Image Standard (rank 32, 1e-4) | ◆ Balanced (rank 32, 1e-4) |
| MOSS-Image | ✨ MOSS-Image Fast (rank 16, 0.25 MP) | ◆ Quick start (rank 16, 0.25 MP) |
| MOSS-Image | ✨ MOSS-Image Style (rank 16, 5e-5) | ◆ Style (rank 16, 5e-5) |
| MOSS-Image | ✨ MOSS-Image Slider (rank 8, 2e-4) | ◆ Slider (rank 8, 2e-4) |
| MOSS-Image | ✨ MOSS-Image Fine-tune (1e-5) | ◆ Full fine-tune (1e-5) |
| SDXL family (each of the five variants; `{name}` was the variant's name) | ✨ {name} Strong (rank 32, alpha 16, 5e-5) | ◆ High capacity (rank 32, alpha 16, 5e-5) |
| SDXL family | ✨ {name} Standard (rank 16, alpha 8, 5e-5) | ◆ Balanced (rank 16, alpha 8, 5e-5) |
| SDXL family | ✨ {name} Slider (rank 32, alpha 16, 5e-5) | ◆ Slider (rank 32, alpha 16, 5e-5) |
| SDXL family | ✨ {name} Fine-tune (1e-5) | ◆ Full fine-tune (1e-5) |
| SDXL family | ✨ {name} Fine-tune Official (3e-6) | ◆ Full fine-tune, conservative (3e-6) |
| SDXL family | ✨ {name} Fast (rank 16, adaptive LR) | ◆ Community: fast (rank 16, adaptive LR) |
| SDXL family | ✨ {name} Cosine (rank 32, community) | ◆ Community: cosine (rank 32) |
| SDXL family | ✨ {name} Style (rank 16, gentle LR) | ◆ Community: style (rank 16, gentle LR) |
| Anima | ✨ Anima Character (rank 16, 1e-4) | ◆ Character (rank 16, 1e-4) |
| Anima | ✨ Anima Style (rank 16, 5e-5) | ◆ Style (rank 16, 5e-5) |
| Anima | ✨ Anima Official (rank 32, 2e-5) | ◆ Model card recipe (rank 32, 2e-5) |
| Anima | ✨ Anima Slider (rank 8, 2e-4) | ◆ Slider (rank 8, 2e-4) |
| Anima | ✨ Anima Fine-tune (1e-5) | ◆ Full fine-tune (1e-5) |
| Anima | ✨ Anima Fine-tune Official (1e-6) | ◆ Full fine-tune, conservative (1e-6) |
| Anima | ✨ Anima Fast (rank 8, adaptive LR) | ◆ Community: fast (rank 8, adaptive LR) |
| Anima | ✨ Anima Standard (rank 32, LR 2e-5) | ◆ Community: standard (rank 32, 2e-5) |
| Anima | ✨ Anima Style (rank 16, gentle LR) | ◆ Community: style (rank 16, gentle LR) |

## Setting labels and groups

The keys do not change; only what the user reads.

| Key | Old label | New label |
|---|---|---|
| (group) | Loss watch | Hard images |
| `KREA2_LOSS_WATCH` | Detect problem images | Track hard images |
| `KREA2_PER_IMAGE_LR` | Per-image adaptive LR | Per-image learning rate |
| `KREA2_AUTO_RECAPTION` | Auto-recaption stuck images | Recaption images that stay hard |
| `KREA2_WARMUP_LOOK` | Warm up look outliers | Ease in visual outliers |
| `CONTEXT_LORA_PATH` | Context LoRA | Companion LoRA (frozen) |
| `CONTEXT_LORA_STRENGTH` | Context LoRA strength | Companion LoRA strength |
| `FAMILY_FAST_ID` | Fast Identity Mode | Train identity blocks only |
| `FAMILY_SLIDER_ULTRA` | Slider Ultra mode | Train composition blocks only |
| `FAMILY_SLIDER_GUIDANCE` | Slider push strength | Slider guidance |
| `FAMILY_EDIT_PICTURES` | What the edit uses | Edit inputs |
| `FAMILY_FT_FUSED` | Free each gradient as soon as it's used (less memory - recommended) | Step each weight as its gradient arrives (less memory, recommended) |
| `H3_STRUCTURE` | Training structure | Noise emphasis |
| `H3_LOWNOISE_PCT` | Clean-end share (%) | Low-noise share (%) |
| `H3_HIGHNOISE_LR_PCT` | Medium to High Noise LR (%) | High-noise learning rate (%) |
| `H3_LIKENESS_MODE` | Training mode | Blocks preset |
| `H3_ADAPTER_RAMP` | Adapter-relative LR | LoRA-size-relative learning rate |
| `MOSS_LOWNOISE_PCT` | Clean-end share (% of steps) | Low-noise share (% of steps) |
| `SAMPLE_USE_DISTILLED` (Klein's default tick text) | Use Distilled model for samples (4-step, matches ComfyUI) | Preview with the Distilled model (4 steps, as ComfyUI renders) |

Other user-facing names: the *Problem Images* window, button and command become **Hard images**; the help topic
"Problem images (loss watch)" becomes "Hard images (per-image loss)". In running text: "loss watch" becomes "hard-image
tracking", "Context LoRA" becomes "companion LoRA", "clean-end share" becomes "low-noise share", "Fast Identity Mode"
becomes "identity blocks only", "Slider Ultra mode" becomes "composition blocks only".

## Not renamed yet (recorded so nobody assumes it was forgotten)

- **Stored option values** of choice settings are still the strings a preset file carries (for example the three
  *Noise emphasis* choices, the fine-tune *Window size* choices, *Edit inputs*' two choices, Klein's model areas).
  Renaming what the user reads there needs a display-text layer over the stored value in the Train tab.
- **Console tags** the progress tracker parses (`[adaptive_lr]`, `[cache]` ...) and descriptive engineering words in
  the trainer's log lines (rotation, window, block swap).
- **Optimizer identifiers** (`automagic3` ...): algorithm names, stored as values.

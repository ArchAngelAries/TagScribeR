"""docs/dev/NAMING.md, pinned: nothing a user can read names another trainer, built-in presets carry TagScribeR's
names, and the credit sits where it belongs. Source headers, comments and docstrings are licence notices and
developer provenance; they are not user text and are not checked here."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORD = "fizgig"
# description fields that hold source citations for developers and are never shown in the app
PROVENANCE_KWARGS = {"notes", "source", "note", "block_note"}
# strings that must keep the word: published third-party file / repository names (NAMING.md rule 7) and the older
# names TagScribeR still reads for compatibility (rule 6)
ALLOWED = ("fizgig_qwen_image_2.1_training_adapter", "fizgig-qwen-image-2.1-training-adapter",
           "fizgig_look_scores.json", "fizgig_finetune", "fizgig_")


def _user_strings(path: Path):
    """(line, text) of every string constant in a file that is not a docstring and not a provenance field."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    skip = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant):
                skip.add(id(body[0].value))
        if isinstance(node, ast.keyword) and node.arg in PROVENANCE_KWARGS:
            skip.update(id(n) for n in ast.walk(node.value))
    for node in tree.body:                      # a description's private citation constants (_FETCH = "<source>")
        if (path.name == "description.py" and isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id.startswith("_")):
            skip.update(id(n) for n in ast.walk(node.value))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in skip:
            yield node.lineno, node.value


def test_no_runtime_string_names_another_trainer():
    files = [ROOT / "main.py"]
    for folder in ("tabs", "training", "core", "inference"):
        files += sorted((ROOT / folder).rglob("*.py"))
    hits = []
    for f in files:
        if f.name == "help_content.py":                # its Credits topic is the one place; tests/test_help_content.py
            continue
        for line, text in _user_strings(f):
            low = text.lower()
            if WORD in low and not any(a in low for a in ALLOWED):
                hits.append(f"{f.relative_to(ROOT).as_posix()}:{line}: {text[:90]!r}")
    assert not hits, "user-facing text names another trainer (docs/dev/NAMING.md):\n" + "\n".join(hits)


def test_built_in_presets_carry_tagscribers_names():
    from training import registry
    old_words = ("Ultra Fast", "Old Reliable", "Flavour", "Fast Identity", "Multi-Character", "✨")
    for d in registry.FAMILIES.values():
        names = [n for n, _ in d.presets]
        assert len(names) == len(set(names)), (d.key, names)
        for n in names:
            assert n.startswith("◆ "), (d.key, n)
            assert not any(w in n for w in old_words), (d.key, n)
            assert d.display_name not in n, (d.key, n)          # the list is per family: no need to repeat it


def test_labels_and_groups_are_tagscribers():
    from training import params as P
    assert "Hard images" in P.GROUPS and "Loss watch" not in P.GROUPS
    labels = {p.key: p.label for p in P.PARAMS}
    assert labels["KREA2_LOSS_WATCH"] == "Track hard images"
    assert labels["CONTEXT_LORA_PATH"] == "Companion LoRA (frozen)"
    assert labels["FAMILY_FAST_ID"] == "Train identity blocks only"
    assert labels["FAMILY_SLIDER_ULTRA"] == "Train composition blocks only"
    assert labels["MOSS_LOWNOISE_PCT"] == "Low-noise share (% of steps)"
    for p in P.PARAMS:                                   # no label or tooltip names another trainer
        assert WORD not in (p.label + " " + p.tip).lower(), p.key
    # the keys themselves did not move: preset files written elsewhere still import
    for key in ("FAMILY_FT", "KREA2_LOSS_WATCH", "FAMILY_FAST_ID", "CONTEXT_LORA_PATH", "H3_STRUCTURE"):
        assert key in P.BY_KEY


def test_credit_is_given_once_where_it_belongs():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    credits_at = readme.index("Credits and licence\n")
    start = readme.rindex("\n## ", 0, readme.index("not affiliated"))
    assert WORD in readme[start:].lower() and "apache" in readme[start:].lower()
    body = readme[:start]
    assert credits_at and WORD not in body.replace("credits-and-licence", "").lower(), "the README names it outside its credits"
    notices = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    assert "https://github.com/shootthesound/Fizgig" in notices and "Apache License" in notices


def test_support_links():
    from core import support
    urls = {i: u for i, _label, u, _tip in support.SUPPORT_LINKS}
    assert urls == {"patreon": "https://www.patreon.com/c/ArchAngelAries", "kofi": "https://ko-fi.com/archangelaries"}
    assert all(u.startswith("https://") for u in urls.values())

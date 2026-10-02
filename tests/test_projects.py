import json

from core.projects import BUILTIN_FILTERS, Project


def test_project_roundtrip_per_folder(tmp_path):
    a, b = tmp_path / "set_a", tmp_path / "set_b"
    a.mkdir()
    b.mkdir()
    pa = Project(a)
    pa.set("subject", "ohwx woman")
    pa.set("subject_first", True)
    pa.save_filter("No smile", "-tag:smile")
    assert Project(a).get("subject") == "ohwx woman" and Project(a).get("subject_first") is True
    assert Project(a).filters() == {"No smile": "-tag:smile"}
    assert Project(b).get("subject") == ""          # other folders unaffected
    assert not any(a.iterdir())                     # nothing written into the dataset folder


def test_project_tolerates_bad_values(tmp_path):
    p = Project(tmp_path)
    p.path.parent.mkdir(parents=True, exist_ok=True)
    p.path.write_text(json.dumps({"subject": 5, "subject_first": "yes", "filters": [1], "health_resolution": "x"}))
    q = Project(tmp_path)
    assert q.get("subject") == "" and q.get("subject_first") is False
    assert q.filters() == {} and q.get("health_resolution") == 0


def test_delete_filter_and_builtins(tmp_path):
    p = Project(tmp_path)
    p.save_filter("x", "tag:x")
    p.delete_filter("x")
    assert Project(tmp_path).filters() == {}
    assert "Missing captions" in BUILTIN_FILTERS

from tabs.palette import Command, fuzzy_score, rank


def cmds(*titles):
    return [Command(t, lambda: None) for t in titles]


def test_substring_and_subsequence():
    assert fuzzy_score("save", "Save captions") is not None
    assert fuzzy_score("mis cap", "Select images missing captions") is not None
    assert fuzzy_score("xyz", "Save captions") is None
    assert fuzzy_score("", "anything") == 0


def test_rank_prefers_word_starts():
    ordered = rank("cap", cmds("Escape hatch", "Auto Caption", "Recapture"))
    assert ordered[0].title == "Auto Caption"


def test_rank_filters_non_matches():
    assert [c.title for c in rank("health", cmds("Scan dataset health", "Save"))] == ["Scan dataset health"]

from core import captions as c


def test_split_and_join_roundtrip():
    assert c.split_tags(" a, b ,, c\n d ") == ["a", "b", "c", "d"]
    assert c.join_tags(["a", "b"]) == "a, b"
    assert c.split_tags("") == []


def test_dedupe_case_insensitive_keeps_first():
    assert c.dedupe_tags(["Red Hair", "red  hair", "solo"]) == ["Red Hair", "solo"]
    assert c.dedupe_tags(["A", "a"], case_sensitive=True) == ["A", "a"]


def test_add_tags_skips_existing_and_respects_position():
    assert c.add_tags("a, b", ["b", "c"]) == "a, b, c"
    assert c.add_tags("a, b", ["c"], position="prepend") == "c, a, b"
    assert c.add_tags("", ["x"]) == "x"
    assert c.add_tags("a, b", ["A"]) == "a, b"  # unchanged text returned as-is


def test_remove_and_replace_whole_tags_only():
    assert c.remove_tags("red hair, hair, solo", ["hair"]) == "red hair, solo"
    assert c.replace_tag("1girl, solo, smile", "solo", "") == "1girl, smile"
    assert c.replace_tag("a, b, c", "b", "c") == "a, c"  # dedupes after replacement


def test_find_replace_literal_and_regex():
    assert c.find_replace("a.b a.b", ".", "-") == "a-b a-b"
    assert c.find_replace("cat dog", r"\bdog\b", "fox", regex=True) == "cat fox"
    assert c.find_replace("Cat", "cat", "dog", case_sensitive=False) == "dog"


def test_normalize_tag_options():
    assert c.normalize_tag("long_hair", underscores_to_spaces=True) == "long hair"
    assert c.normalize_tag("^_^", underscores_to_spaces=True) == "^_^"
    assert c.normalize_tag("ganyu (genshin impact)", escape_parentheses=True) == r"ganyu \(genshin impact\)"
    assert c.normalize_tag(r"already \(escaped\)", escape_parentheses=True) == r"already \(escaped\)"


def test_merge_generated_tags_modes():
    assert c.merge_generated_tags("a, b", ["b", "c"], "append") == "a, b, c"
    assert c.merge_generated_tags("a, b", ["c"], "overwrite") == "c"
    assert c.merge_generated_tags("keep me", ["c"], "ignore") == "keep me"
    assert c.merge_generated_tags("", ["c"], "ignore") == "c"
    out = c.merge_generated_tags("x", ["trigger", "y"], "append", prepend=["trigger"], append=["end"])
    assert out == "trigger, x, y, end"


def test_tag_frequencies_counts_once_per_caption():
    freq = c.tag_frequencies(["a, b, a", "A, c"])
    assert freq["a"] == 2 and freq["b"] == 1 and freq["c"] == 1


def test_to_ascii():
    assert c.to_ascii("café naïve") == "cafe naive"


def test_clean_model_output():
    assert c.clean_model_output("<think>hmm</think>\nA red car.") == "A red car."
    assert c.clean_model_output("A cat.<think>unfinished") == "A cat."
    assert c.clean_model_output("reasoning...</think>Final.") == "Final."
    assert c.clean_model_output("```\ntag1, tag2\n```") == "tag1, tag2"
    assert c.clean_model_output("Sure! Here is the description: A dog.") == "A dog."


def test_looks_like_tags():
    assert c.looks_like_tags("1girl, solo, long hair")
    assert not c.looks_like_tags("A woman stands in a field of flowers at sunset.")


def test_estimate_clip_tokens():
    assert c.estimate_clip_tokens("") == 0
    assert c.estimate_clip_tokens("1girl, solo, long hair, smile") == 9
    long_caption = ", ".join(["detailed background"] * 30)
    assert c.estimate_clip_tokens(long_caption) > 75

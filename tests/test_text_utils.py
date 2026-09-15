from text_utils import clean_text


def test_clean_text_normalizes_case_and_spacing():
    assert clean_text("  HELLO   World  ") == "hello world"


def test_clean_text_removes_urls_numbers_and_punctuation():
    text = "WIN $1000 now!!! Visit https://example.com today."
    assert clean_text(text) == "win now visit today"


def test_clean_text_handles_empty_like_input():
    assert clean_text("123 !!! https://example.com") == ""

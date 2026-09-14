import pytest

from text_utils import clean_text


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Hello WORLD!", "hello world"),
        ("Visit https://example.com now", "visit now"),
        ("Win £1,000 TODAY!!!", "win £ today"),
        ("  extra\t whitespace\n", "extra whitespace"),
        ("12345 !!!", ""),
    ],
)
def test_clean_text(raw, expected):
    assert clean_text(raw) == expected


def test_clean_text_rejects_non_string():
    with pytest.raises(TypeError):
        clean_text(None)

from tools.message_management import force_plaintext, requires_force_plaintext


def test_force_plaintext_removes_markdown():
    src = """Here's what I know about you:

*   **Name:** Wouter
*   **Date of Birth:** 14 November 1996
# Header
"""
    out = force_plaintext(src)
    assert "*" not in out
    assert "#" not in out
    assert "Name:" in out
    assert "Date of Birth:" in out


def test_requires_force_plaintext_detects_drop_chars():
    assert requires_force_plaintext("**Bold** text with # headings")


def test_requires_force_plaintext_skips_clean_text():
    clean = "Your name is Wouter."
    assert not requires_force_plaintext(clean)


def test_requires_force_plaintext_detects_bullets():
    bullet = "- item one\nsecond line"
    assert requires_force_plaintext(bullet)

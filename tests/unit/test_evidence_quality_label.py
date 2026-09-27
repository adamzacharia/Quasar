"""The 'Likely scholarly' label needs a publication signal, not the bare words
"journal" or "accepted" (a Merriam-Webster definition page carried the label,
2026-09-25)."""
from services.evidence_quality import assess_web_source_quality


def test_dictionary_page_with_the_word_journal_is_general_web():
    q = assess_web_source_quality(
        "https://www.merriam-webster.com/dictionary/current",
        "CURRENT Definition & Meaning - Merriam-Webster",
        "The meaning of CURRENT is ... a journal accepted as current; recent accepted usage.",
    )
    assert q["tier"] == "general" and q["label"] == "General web"


def test_real_publication_signals_still_count():
    assert assess_web_source_quality("https://example.org/p", "A paper", "Bibcode: 2025ApJ...990L..28L")["label"] == "Likely scholarly"
    assert assess_web_source_quality("https://example.org/p", "A paper", "doi: 10.3847/2041-8213/ade1e2")["label"] == "Likely scholarly"
    assert assess_web_source_quality("https://example.org/p", "GRB 250702B", "published in ApJ Letters, arXiv:2509.22792")["label"] == "Likely scholarly"
    assert assess_web_source_quality("https://example.org/p", "Results", "MNRAS 545, 1234 (2026)")["label"] == "Likely scholarly"



def test_g1630_cx19_bare_word_arxiv_is_not_a_scholarly_signal():
    q = assess_web_source_quality("https://example.org/help/search", "How to search arXiv",
                                  "Tips for searching arXiv listings and setting up arXiv email alerts.")
    assert q["label"] == "General web"
    for text in ("see arXiv:2509.22792v2", "arXiv 2509.22792", "https://arxiv.org/abs/2509.22792",
                 "astro-ph/0601001"):
        assert assess_web_source_quality("https://example.org/p", "A paper", text)["label"] == "Likely scholarly", text

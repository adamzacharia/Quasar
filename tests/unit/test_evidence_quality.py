from services.evidence_quality import (
    annotate_web_source_evidence,
    choose_better_evidence_quality,
    rank_web_sources,
)


def test_primary_archive_scores_above_general_web():
    archive = annotate_web_source_evidence({
        "title": "ALMA Science Archive",
        "url": "https://almascience.eso.org/aq/",
        "snippet": "",
    })
    general = annotate_web_source_evidence({
        "title": "Blog post",
        "url": "https://example.com/alma-overview",
        "snippet": "",
    })

    assert archive["evidenceQuality"]["tier"] == "primary"
    assert archive["evidenceQuality"]["score"] > general["evidenceQuality"]["score"]


def test_peer_reviewed_and_preprint_are_distinguished():
    peer_reviewed = annotate_web_source_evidence({
        "title": "Paper record",
        "url": "https://ui.adsabs.harvard.edu/abs/2023ApJ...123..456A/abstract",
        "snippet": "",
    })
    preprint = annotate_web_source_evidence({
        "title": "Preprint",
        "url": "https://arxiv.org/abs/2401.12345",
        "snippet": "",
    })

    assert peer_reviewed["evidenceQuality"]["label"] == "Peer reviewed"
    assert preprint["evidenceQuality"]["label"] == "Preprint"
    assert peer_reviewed["evidenceQuality"]["score"] > preprint["evidenceQuality"]["score"]


def test_rank_web_sources_orders_by_authority_not_provider_order():
    sources = [
        {"title": "General result", "url": "https://example.com/post", "snippet": ""},
        {"title": "ADS result", "url": "https://ui.adsabs.harvard.edu/abs/2024A%26A...1..2B/abstract", "snippet": ""},
        {"title": "Archive result", "url": "https://mast.stsci.edu/search/ui/", "snippet": ""},
    ]

    ranked = rank_web_sources(sources)

    assert ranked[0]["url"] == "https://mast.stsci.edu/search/ui/"
    assert ranked[1]["url"].startswith("https://ui.adsabs.harvard.edu/")
    assert ranked[2]["url"] == "https://example.com/post"


def test_duplicate_merge_keeps_stronger_existing_quality():
    chosen = choose_better_evidence_quality(
        {"score": 92, "tier": "primary", "label": "Observatory docs"},
        {"score": 42, "tier": "general", "label": "General web"},
    )

    assert chosen["score"] == 92
    assert chosen["label"] == "Observatory docs"

from services.content_safety import (
    FILTER_NOTICE,
    is_blocked_url,
    is_explicit_query,
    is_safe_web_source,
    sanitize_web_payload,
    web_images_enabled,
)


def test_blocks_known_adult_hosts_on_dns_boundaries(monkeypatch):
    monkeypatch.setenv("QUASAR_STRICT_WEB_FILTER", "true")

    assert is_blocked_url("https://www.xvideos.com/video123") is True
    assert is_blocked_url("https://cdn.xvideos.com/image.jpg") is True
    assert is_blocked_url("https://xvideos.com.example.org/article") is False


def test_blocks_explicit_source_metadata_without_blocking_ordinary_adult_word(monkeypatch):
    monkeypatch.setenv("QUASAR_STRICT_WEB_FILTER", "true")

    assert is_safe_web_source(
        {
            "url": "https://example.com/profile",
            "title": "Biography",
            "snippet": "A pornographic actress and director.",
        }
    ) is False
    assert is_safe_web_source(
        {
            "url": "https://example.edu/education",
            "title": "Adult education programs",
            "snippet": "Continuing education options for working learners.",
        }
    ) is True
    assert is_explicit_query("find pornographic videos") is True
    assert is_explicit_query("adult education programs") is False


def test_explicit_context_withholds_all_sources_images_and_answer(monkeypatch):
    monkeypatch.setenv("QUASAR_STRICT_WEB_FILTER", "true")
    monkeypatch.setenv("QUASAR_WEB_IMAGES_ENABLED", "true")

    result = sanitize_web_payload(
        {
            "success": True,
            "answer": "The subject is a pornographic actress.",
            "results": [
                {
                    "title": "Explicit profile",
                    "url": "https://example.com/profile",
                    "snippet": "Adult film performer.",
                },
                {
                    "title": "Generic profile",
                    "url": "https://example.org/profile",
                    "snippet": "Biography.",
                },
            ],
            "images": [
                {
                    "url": "https://images.example.org/photo.jpg",
                    "description": "Portrait",
                }
            ],
        }
    )

    assert result["results"] == []
    assert result["images"] == []
    assert result["answer"] == FILTER_NOTICE
    assert result["content_filter"]["filtered"] is True
    assert result["content_filter"]["blocked_results"] == 2
    assert result["content_filter"]["blocked_images"] == 1


def test_general_web_images_are_enabled_by_default(monkeypatch):
    monkeypatch.delenv("QUASAR_WEB_IMAGES_ENABLED", raising=False)

    assert web_images_enabled() is True
    result = sanitize_web_payload(
        {
            "success": True,
            "results": [{"title": "Safe", "url": "https://example.com", "snippet": "Safe"}],
            "images": [{"url": "https://example.com/image.jpg", "description": "Safe image"}],
        }
    )

    assert len(result["results"]) == 1
    assert len(result["images"]) == 1
    assert result["content_filter"]["blocked_images"] == 0
    assert result["content_filter"]["web_images_enabled"] is True


def test_env_override_disables_general_web_images(monkeypatch):
    monkeypatch.setenv("QUASAR_WEB_IMAGES_ENABLED", "false")

    assert web_images_enabled() is False
    result = sanitize_web_payload(
        {
            "success": True,
            "results": [{"title": "Safe", "url": "https://example.com", "snippet": "Safe"}],
            "images": [{"url": "https://example.com/image.jpg", "description": "Safe image"}],
        }
    )

    assert len(result["results"]) == 1
    assert result["images"] == []
    assert result["content_filter"]["blocked_images"] == 1
    assert result["content_filter"]["web_images_enabled"] is False


def test_sanitizer_is_idempotent(monkeypatch):
    monkeypatch.setenv("QUASAR_STRICT_WEB_FILTER", "true")

    once = sanitize_web_payload(
        {
            "success": True,
            "results": [
                {
                    "title": "Blocked",
                    "url": "https://xvideos.com/example",
                    "snippet": "",
                }
            ],
        }
    )
    twice = sanitize_web_payload(once)

    assert twice["answer"] == FILTER_NOTICE
    assert twice["content_filter"]["filtered"] is True
    assert twice["content_filter"]["blocked_results"] == 1

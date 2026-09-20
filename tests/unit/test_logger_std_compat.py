"""core.logger must render stdlib-style records.

Callers use the loguru logger with stdlib idioms —
``logger.error("failed for %s: %s", conv_id, err, exc_info=True)``. Loguru
formats with str.format, so the ``%s`` placeholders were printed literally and
``exc_info`` was swallowed as an unused format kwarg: the provider-failure log
at core/runner.py read ``Responses API request failed for conversation %s: %s``
with no traceback, which hid the exception class behind the 2026-09-17
provider-killed benchmark trials.
"""

import io

import pytest

from core.logger import logger


@pytest.fixture
def sink():
    buf = io.StringIO()
    handle = logger.add(buf, level="DEBUG", format="{level}|{name}:{function}|{message}\n{exception}")
    try:
        yield buf
    finally:
        logger.remove(handle)


def test_percent_style_args_are_rendered(sink):
    logger.error("Responses API request failed for conversation %s: %s", "conv-1", "boom")
    text = sink.getvalue()
    assert "conversation conv-1: boom" in text
    assert "%s" not in text


def test_exc_info_true_renders_the_traceback_and_names_the_caller(sink):
    try:
        raise RuntimeError("mid-stream death")
    except RuntimeError:
        logger.error("request failed for %s: %s", "conv-2", "mid-stream death", exc_info=True)
    text = sink.getvalue()
    assert "request failed for conv-2: mid-stream death" in text
    assert "RuntimeError: mid-stream death" in text and "Traceback" in text
    # depth is corrected so the record names THIS test, not the wrapper method
    assert "test_exc_info_true_renders_the_traceback_and_names_the_caller" in text


def test_exc_info_exception_instance_is_accepted(sink):
    err = ValueError("explicit instance")
    logger.warning("caught %s", "something", exc_info=err)
    text = sink.getvalue()
    assert "caught something" in text and "ValueError: explicit instance" in text


def test_logger_exception_renders_traceback_without_exc_info(sink):
    try:
        raise KeyError("k")
    except KeyError:
        logger.exception("tool %s raised", "x")
    text = sink.getvalue()
    assert "tool x raised" in text and "KeyError" in text


def test_loguru_brace_style_and_plain_messages_are_unchanged(sink):
    logger.info("brace style {} and {name}", 1, name="n")
    logger.info("literal braces {not formatted} 100%")
    text = sink.getvalue()
    assert "brace style 1 and n" in text
    assert "literal braces {not formatted} 100%" in text


def test_loguru_api_surface_still_passes_through():
    assert callable(logger.opt) and callable(logger.bind) and callable(logger.add)
    bound = logger.bind(request_id="r-1")
    assert callable(bound.info)

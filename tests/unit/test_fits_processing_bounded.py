from astropy.io import fits

from services.fits_processing import FITSProcessingService


class _StreamingResponse:
    def __init__(self, payload: bytes, status_code: int = 200):
        self.payload = payload
        self.status_code = status_code
        self.closed = False
        self.bytes_yielded = 0

    @property
    def content(self):
        raise AssertionError("bounded header reads must not access response.content")

    def iter_content(self, chunk_size=16384):
        for offset in range(0, len(self.payload), chunk_size):
            chunk = self.payload[offset:offset + chunk_size]
            self.bytes_yielded += len(chunk)
            yield chunk

    def close(self):
        self.closed = True


def test_range_reader_stays_bounded_when_server_ignores_range(monkeypatch):
    header = fits.Header()
    header["SIMPLE"] = True
    header["BITPIX"] = 8
    header["NAXIS"] = 0
    header["OBJECT"] = "TW Hya"
    header_bytes = header.tostring(endcard=True, padding=True).encode("ascii")
    response = _StreamingResponse(header_bytes + (b"x" * 500_000), status_code=200)

    monkeypatch.setattr(
        "services.fits_processing.requests.get",
        lambda *args, **kwargs: response,
    )

    parsed = FITSProcessingService._fetch_header_via_range(
        "https://example.test/large-cube.fits",
        max_bytes=131072,
    )

    assert parsed["OBJECT"] == "TW Hya"
    assert response.closed is True
    assert response.bytes_yielded <= 131072


def test_missing_end_card_returns_none_without_full_download(monkeypatch):
    response = _StreamingResponse(b" " * 200_000, status_code=206)
    monkeypatch.setattr(
        "services.fits_processing.requests.get",
        lambda *args, **kwargs: response,
    )

    parsed = FITSProcessingService._fetch_header_via_range(
        "https://example.test/not-fits.bin",
        max_bytes=131072,
    )

    assert parsed is None
    assert response.closed is True


def test_range_416_retries_without_range_but_remains_bounded(monkeypatch):
    header = fits.Header()
    header["SIMPLE"] = True
    header["BITPIX"] = 8
    header["NAXIS"] = 0
    header["OBJECT"] = "TW Hya"
    header_bytes = header.tostring(endcard=True, padding=True).encode("ascii")
    range_response = _StreamingResponse(b"", status_code=416)
    plain_response = _StreamingResponse(
        header_bytes + (b"x" * 500_000),
        status_code=200,
    )
    responses = iter([range_response, plain_response])
    calls = []

    def fake_get(*args, **kwargs):
        calls.append(kwargs)
        return next(responses)

    monkeypatch.setattr("services.fits_processing.requests.get", fake_get)

    parsed = FITSProcessingService._fetch_header_via_range(
        "https://example.test/range-rejected.fits",
        max_bytes=131072,
    )

    assert parsed["OBJECT"] == "TW Hya"
    assert range_response.closed is True
    assert plain_response.closed is True
    assert plain_response.bytes_yielded <= 131072
    assert "Range" in calls[0]["headers"]
    assert "headers" not in calls[1]

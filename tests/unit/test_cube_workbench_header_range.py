import io

import numpy as np
from astropy.io import fits

from services.cube_workbench import CubeWorkbenchService


def _extension_cube_fits_bytes() -> bytes:
    primary = fits.PrimaryHDU()
    extension = fits.ImageHDU(data=np.zeros((4, 8, 9), dtype="float32"))
    extension.header["BUNIT"] = "Jy/beam"
    extension.header["RESTFRQ"] = 230.538e9
    extension.header["CTYPE1"] = "RA---TAN"
    extension.header["CTYPE2"] = "DEC--TAN"
    extension.header["CTYPE3"] = "FREQ"
    extension.header["CUNIT3"] = "Hz"
    extension.header["CRVAL3"] = 230.0e9
    extension.header["CDELT3"] = 1.0e6
    extension.header["CRPIX3"] = 1
    buffer = io.BytesIO()
    fits.HDUList([primary, extension]).writeto(buffer)
    return buffer.getvalue()


def test_header_range_parser_finds_first_image_extension_hdu():
    raw = _extension_cube_fits_bytes()

    parsed = CubeWorkbenchService._first_image_header_from_fits_bytes(raw[:5760])

    assert parsed is not None
    assert parsed["hdu_index"] == 1
    assert parsed["header_offset_bytes"] == 2880
    assert parsed["data_offset_bytes"] == 5760
    assert parsed["data_bytes_estimated"] == 4 * 8 * 9 * 4

    metadata = CubeWorkbenchService._metadata_from_header(parsed["header"], "https://example.test/cube.fits", "cube.fits")
    assert metadata["shape"] == [4, 8, 9]
    assert metadata["channel_count"] == 4
    assert metadata["rest_freq_ghz"] == 230.538
    assert metadata["unit"] == "Jy/beam"


def test_remote_header_only_metadata_uses_bounded_stream(monkeypatch, tmp_path):
    raw = _extension_cube_fits_bytes()[:5760]

    class FakeResponse:
        status_code = 200
        headers = {
            "Content-Length": str(50 * 1024 * 1024),
        }

        def iter_content(self, chunk_size=65536):
            for start in range(0, len(raw), chunk_size):
                yield raw[start:start + chunk_size]

        def close(self):
            pass

    def fake_get(url, headers, timeout, stream):
        assert headers["Range"].startswith("bytes=0-")
        assert stream is True
        return FakeResponse()

    monkeypatch.setattr("services.cube_workbench.requests.get", fake_get)
    service = CubeWorkbenchService(session_dir=tmp_path / "sessions", cache_dir=tmp_path / "cache")
    try:
        metadata = service._metadata_for_remote_header_only("https://example.test/cube.fits")
    finally:
        service.shutdown()

    assert metadata["success"] is True
    assert metadata["header_strategy"] == "bounded_remote_range"
    assert metadata["range_header_read"] is True
    assert metadata["range_truncated"] is True
    assert metadata["header_bytes_fetched"] == len(raw)
    assert metadata["data_hdu_index"] == 1
    assert metadata["shape"] == [4, 8, 9]

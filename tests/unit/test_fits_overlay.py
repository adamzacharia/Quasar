from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from PIL import Image

from services.fits_service import overlay_fits_images


def _write_test_fits(path: Path, data: np.ndarray) -> None:
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [data.shape[1] / 2, data.shape[0] / 2]
    wcs.wcs.cdelt = np.array([-0.0001, 0.0001])
    wcs.wcs.crval = [53.1625, -27.7914]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    header = wcs.to_header()
    fits.PrimaryHDU(data=data.astype("float32"), header=header).writeto(path)


def test_overlay_fits_images_renders_nonblank_local_png(tmp_path):
    y, x = np.mgrid[0:64, 0:64]
    base = np.exp(-((x - 32) ** 2 + (y - 32) ** 2) / 300)
    contour = np.exp(-((x - 35) ** 2 + (y - 30) ** 2) / 60)
    base_path = tmp_path / "base.fits"
    contour_path = tmp_path / "contour.fits"
    _write_test_fits(base_path, base)
    _write_test_fits(contour_path, contour)

    result = overlay_fits_images(str(base_path), str(contour_path), base_label="JWST", contour_label="ALMA")

    assert result["success"] is True
    image_path = Path(__file__).resolve().parents[2] / "data" / "rendered_images" / Path(result["image_path"]).name
    assert image_path.exists()
    with Image.open(image_path) as rendered:
        arr = np.asarray(rendered)
    assert arr.std() > 0

"""ALMA FITS product preview endpoint.

The route owns the URL allowlisting + bounded HTTP download; the (heavy)
matplotlib rendering lives in ``api.serializers.fits``.
"""

from fastapi import APIRouter, Depends, HTTPException

from api.deps import get_current_user
from api.models import FitsPreviewRequest
from api.serializers.fits import render_fits_preview

router = APIRouter()


@router.post("/api/fits/preview")
async def preview_fits(req: FitsPreviewRequest, current_user: dict = Depends(get_current_user)):
    """Download a small ALMA FITS product and return workbench-ready previews."""
    from urllib.parse import urlparse

    import requests

    parsed = urlparse(str(req.url or "").strip())
    allowed_hosts = {"almascience.nrao.edu", "almascience.eso.org", "almascience.nao.ac.jp"}
    if parsed.scheme != "https" or parsed.hostname not in allowed_hosts:
        raise HTTPException(status_code=400, detail="Only HTTPS ALMA data product URLs can be previewed.")

    max_bytes = 50 * 1024 * 1024
    try:
        with requests.get(req.url, stream=True, timeout=90) as response:
            response.raise_for_status()
            length = response.headers.get("content-length")
            if length and int(length) > max_bytes:
                raise HTTPException(status_code=413, detail="This FITS file is too large for in-browser preview. Open or download it from ALMA instead.")

            chunks = []
            total = 0
            for chunk in response.iter_content(chunk_size=1024 * 256):
                if not chunk:
                    continue
                total += len(chunk)
                if total > max_bytes:
                    raise HTTPException(status_code=413, detail="This FITS file is too large for in-browser preview. Open or download it from ALMA instead.")
                chunks.append(chunk)
        fits_bytes = b"".join(chunks)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Could not fetch FITS product: {exc}")

    return render_fits_preview(
        fits_bytes,
        download_url=req.url,
        filename=req.filename,
        parsed_path=parsed.path,
        hostname=parsed.hostname,
    )

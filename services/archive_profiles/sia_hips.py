"""SIA / HiPS imagery profile (non-tabular: tables=None).

Live survey imagery through CDS hips2fits (hips_cutout / hips_multiband_panel /
vlass_cutout). Grounds the two chronic traps: fov is DEGREES, and coverage is
survey-specific (VLASS Dec > -40; Legacy Surveys g/r/z only).
"""

from __future__ import annotations

from services.archive_profiles.schema import ArchiveProfile

PROFILE = ArchiveProfile.model_validate(
    {
        "archive": "sia_hips",
        "aliases": ("hips", "hips2fits", "sia", "imagery", "cutouts"),
        "description": (
            "Live sky imagery via CDS hips2fits (~1000 HiPS surveys, alias shortcuts for common "
            "ones) plus the VLASS radio quicklook; Data Lab SIA cutouts cover DECam coadds."
        ),
        "endpoints": [
            {
                "id": "hips2fits",
                "description": "CDS hips2fits cutout service.",
                "url": "https://alasky.cds.unistra.fr/hips-image-services/hips2fits",
                "protocol": "hips2fits",
            },
        ],
        "query_surfaces": [
            {
                "id": "cutout",
                "purpose": "Single survey cutout for 'show me X' / appearance questions.",
                "tool": "hips_cutout",
                "request_kind": "http_get",
                "parameters": [
                    {"name": "survey", "json_type": "string",
                     "description": "Alias (optical/dss2, dss2_red, sdss, 2mass/nir, wise/mir, galex/uv, "
                                    "rosat/xray, fermi/gamma, vlass/radio) or a raw HiPS id like CDS/P/DSS2/color."},
                    {"name": "fov_deg", "json_type": "number", "unit": "deg",
                     "description": "Field of view in DEGREES (default 0.25) — size it from the target's apparent extent."},
                    {"name": "width", "json_type": "integer", "unit": "pix",
                     "description": "Image width in pixels (default 512)."},
                ],
                "endpoint_ids": ["hips2fits"],
            },
            {
                "id": "multiband_panel",
                "purpose": "Side-by-side multiwavelength panels (defaults optical/2MASS/WISE).",
                "tool": "hips_multiband_panel",
                "request_kind": "http_get",
                "parameters": [
                    {"name": "fov_deg", "json_type": "number", "unit": "deg",
                     "description": "Field of view in DEGREES shared by all panels."},
                ],
                "endpoint_ids": ["hips2fits"],
            },
            {
                "id": "vlass",
                "purpose": "VLASS 3 GHz radio-continuum cutout.",
                "tool": "vlass_cutout",
                "request_kind": "http_get",
                "parameters": [
                    {"name": "fov_deg", "json_type": "number", "unit": "deg",
                     "description": "Field of view in DEGREES (default 0.1)."},
                ],
                "endpoint_ids": ["hips2fits"],
            },
        ],
        "tables": None,
        "pitfalls": [
            {
                "id": "fov_degrees",
                "summary": "fov_deg is DEGREES — pick it from the target's apparent size (M31's D25 is ~3 deg; a 'center' view is ~0.1-0.2 deg), never a fixed constant",
                "applies_to": [{"kind": "archive", "ref": "sia_hips"}],
                "prompt_rank": 1,
            },
            {
                "id": "coverage_limits",
                "summary": "VLASS covers Dec > -40 only; Legacy Surveys imaging is g,r,z (NO i band); if a DECam color image lacks 3 usable bands, fall back to hips_cutout and say so",
                "applies_to": [{"kind": "archive", "ref": "sia_hips"}],
                "prompt_rank": 2,
            },
            {
                "id": "fresh_image_per_turn",
                "summary": "whenever the user asks to SEE something, call an imaging tool in THIS turn — earlier images are not re-displayed",
                "applies_to": [{"kind": "archive", "ref": "sia_hips"}],
            },
        ],
        "golden_examples": [
            {
                "id": "optical_cutout",
                "intent": "What does M87 look like in the optical?",
                "invocation": {
                    "tool": "hips_cutout",
                    "arguments": {"target_name": "M87", "survey": "optical", "fov_deg": 0.25},
                },
                "request": {
                    "kind": "url",
                    "url": (
                        "https://alasky.cds.unistra.fr/hips-image-services/hips2fits"
                        "?hips=CDS%2FP%2FDSS2%2Fcolor&ra=187.70593&dec=12.39112"
                        "&fov=0.25&width=512&height=512&format=png"
                    ),
                },
                "note": "The underlying GET: hips id + ra/dec/fov in degrees.",
            },
            {
                "id": "multiwavelength_panels",
                "intent": "Show NGC 253 across optical, near-IR and mid-IR.",
                "invocation": {
                    "tool": "hips_multiband_panel",
                    "arguments": {"target_name": "NGC 253", "surveys": ["optical", "2mass", "wise"], "fov_deg": 0.4},
                },
            },
            {
                "id": "radio_quicklook",
                "intent": "Radio appearance of a source at known coordinates.",
                "invocation": {
                    "tool": "vlass_cutout",
                    "arguments": {"ra": 187.70593, "dec": 12.39112, "fov_deg": 0.1},
                },
                "note": "Check Dec > -40 before promising VLASS coverage.",
            },
        ],
        "citations": [
            {"id": "cite_hips2fits", "text": "CDS hips2fits service (Boch et al.)",
             "url": "https://alasky.cds.unistra.fr/hips-image-services/hips2fits"},
        ],
    }
)

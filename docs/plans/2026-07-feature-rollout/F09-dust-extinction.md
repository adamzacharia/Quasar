# F09 — Galactic dust / extinction (IRSA SFD v1)

Read `docs/plans/2026-07-feature-rollout/CONVENTIONS.md` and the exemplars it
names BEFORE coding.

## Objective

One tool call for E(B−V) at any sky position (SFD98 + the Schlafly &
Finkbeiner 2011 recalibration) and per-band extinctions A_λ. v1 uses IRSA's
DUST service over REST — NO `dustmaps` package, NO local map downloads.

## Files

- CREATE `services/dust_extinction.py`
- CREATE `tests/unit/test_dust_extinction.py`
- CREATE `scripts/smoke/smoke_f09_dust.py`
- EDIT `core/agent.py` (4 anchors)

## Endpoint

`https://irsa.ipac.caltech.edu/cgi-bin/DUST/nph-dust` (env
`IRSA_DUST_BASE_URL`, `IRSA_DUST_TIMEOUT`, default 30 s).

GET params: `locstr=<ra> <dec> equ j2000` (URL-encoded; use
`params={"locstr": f"{ra} {dec} equ j2000", "regSize": "2.0"}` and let
requests encode), response is XML.

Parsing (be defensive — tag names have varied): walk the XML tree
(`xml.etree.ElementTree`); within the `<statistics>` section of the E(B−V)
result, collect leaf tags whose names contain (case-insensitive):
- `refPixelValueSandF` / `meanValueSandF` → SF11 values
- `refPixelValueSFD` / `meanValueSFD` → SFD98 values
Values look like `"0.0210 (mag)"` — strip non-numeric suffix with a regex
(`float(re.search(r"[-+0-9.eE]+", text).group())`). If SandF tags are absent,
compute SF11 = 0.86 × SFD and add a warning. Prefer `refPixelValue` as the
headline number; report `meanValue` alongside.

## Extinction coefficients

Embed `EXTINCTION_COEFF: dict[str, float]` = A_λ/E(B−V) for R_V=3.1 from
Schlafly & Finkbeiner (2011), Table 6 — cite in a comment:

Landolt U 4.334, B 3.626, V 2.742, R 2.169, I 1.505;
SDSS u 4.239, g 3.303, r 2.285, i 1.698, z 1.263;
PS1 g 3.172, r 2.271, i 1.682, z 1.322, y 1.087;
2MASS J 0.723, H 0.460, Ks 0.310;
WISE W1 0.189, W2 0.146.

Key format: `"U","B","V","R","I","sdss_u",...,"ps1_g",...,"J","H","Ks",
"W1","W2"`.

## Class and methods

```python
class DustExtinctionService:
    def __init__(self, *, base_url=None, timeout=None): ...

    def ebv(self, ra, dec) -> dict:
        """{"success": True, "ebv_sfd": float, "ebv_sf11": float,
        "ebv_sfd_mean": float|None, "ebv_sf11_mean": float|None,
        "warnings": [...], "provenance": {...}}
        Validate ra/dec ranges. |b| < 5 deg (compute Galactic b via
        astropy.coordinates.SkyCoord — this import is fine lazily) →
        append warning 'low Galactic latitude: SFD values unreliable in the
        plane'."""

    def extinction_table(self, ra, dec, bands=None) -> dict:
        """Calls self.ebv(); rows = [{"band", "A_lambda", "coeff"}] using
        ebv_sf11 (state that in provenance). bands=None → all known bands;
        unknown requested band → warning, skipped. Returns rows/count/
        warnings/provenance + the two E(B-V) values."""
```

## agent.py wiring

Tools (category `"archive"`):

1. `galactic_extinction` — wrapper `_galactic_extinction(target_name=None,
   ra=None, dec=None, bands=None)`; resolves position; calls
   `extinction_table`; table card via `_external_catalog_table_result`,
   columns `["band","A_lambda","coeff"]`, source `"IRSA DUST (SFD98/SF11)"`,
   filter_label `f"E(B-V) at {label}: SFD={...:.4f}, SF11={...:.4f}"`.
   Description: "Galactic dust reddening E(B-V) (SFD98 + Schlafly-Finkbeiner
   2011) and per-band extinction A_lambda at a sky position — use before any
   photometric correction, color, or distance-modulus work."

Status label: `"galactic_extinction": "Querying IRSA dust maps"`.

Prompt bullet: `Use \`galactic_extinction\` for E(B-V)/A_lambda whenever
photometry, colors, or distance moduli need dereddening.`

## Unit tests

1. Correct params (locstr formatting) + timeout; canned XML with both SFD and
   SandF stats → both values parsed, `(mag)` suffix stripped.
2. Canned XML missing SandF → sf11 = 0.86×sfd, warning present.
3. Low-latitude position (e.g. ra=266.4, dec=-29.0 → b≈0) → warning.
4. HTTP 500 → success False.
5. `extinction_table`: known bands math (A_V = 2.742×E(B−V)); unknown band
   'foo' → warning + skipped; bands=None returns all coefficients.

## Smoke expectations

- `ebv(187.2779, 2.0524)` (3C 273, high latitude): SFD ≈ 0.02 (accept
  0.005–0.05), SF11 < SFD.
- `extinction_table(...)['rows']` contains V with A_V ≈ 2.742×E(B−V)_SF11.
Exit 0 only if both hold.

## Chat acceptance question

"What's the Galactic extinction toward M87 in the SDSS r band?"

## Acceptance checklist

- [ ] CONVENTIONS definition-of-done ticked
- [ ] Unit tests green; smoke passes; wiring complete (1 tool)

# First-Class Cube/Product Workbench Architecture

This document defines the system needed to turn Quasar's current FITS preview
modal into a first-class ALMA cube/product workbench.

## Product Goal

The workbench should let an ALMA user move from an archive product row to a
science-ready inspection environment:

- inspect FITS metadata with WCS and spectral-axis context
- browse channels and channel-map grids
- compute moment 0/1/2 maps
- extract spectra from a click, aperture, or region
- draw or parameterize PV slices
- tune contours, stretch, colormap, RMS region, and channel windows
- identify spectral-line candidates with redshift-aware Splatalogue overlays
- export reproducible CASA, CARTA, DS9, Python, notebook, and figure artifacts
- show evidence: data URL, headers inspected, assumptions, warnings, and confidence

## Current State

The first slice is implemented through `/api/fits/preview` and the product-table
modal:

- small FITS download with a 50 MB limit
- image/cube detection
- moment-style cube collapse
- channel-map grid
- spectrum extraction
- central PV slice
- contour levels, beam display, RMS estimate
- FITS `RESTFRQ` overlay
- copyable CASA/CARTA/DS9 commands
- local evidence metadata

This is useful, but it is not yet first-class because it is modal-only,
stateless, size-limited, and mostly pixel-space.

The second slice adds the first durable workbench boundary:

- persistent per-user sessions in `services/cube_workbench.py`
- `Open Workbench` action from ALMA product rows
- dedicated `/workbench/{session_id}` route
- metadata, render-plan, spectrum-plan, PV-slice-plan, line-overlay, and export
  API contracts
- downloadable CASA, DS9 region, Python, and notebook artifacts
- bounded FITS product preparation into `data/workbench_cache`
- per-user workbench cache accounting with a default 2 GB quota, age/size
  eviction, and evicted-session state updates
- computed aperture spectra and PV-slice PNGs when a product is prepared
- computed channel/image/moment render PNGs from prepared cubes
- render-time RMS estimates and contour levels
- persisted RMS-region selection used for contour scaling and render statistics
- durable job records for prepare/render/spectrum/PV operations with polling,
  progress, cancellation requests, and terminal result/error state
- curated common-line presets for redshift-aware line ID
- line markers on computed spectrum plots
- line labels on cached channel/moment render images when the rendered spectral
  channel or moment range overlaps the latest line-ID result
- rendered-image selection modes for click-to-set aperture, drag-to-set RMS box,
  and drag-to-set PV line endpoints
- computed spectrum CSV export and current render PNG export for prepared
  products
- bounded remote FITS header inspection for workbench session creation, using
  streamed HTTP range bytes and first image/cube HDU detection without
  downloading full remote products

It still does not yet provide a tiled or background-rendered large-cube engine.
Prepared products can now produce spectrum, PV, channel, and moment outputs, but
tiled/downsampled remote previews, distributed/background worker execution,
sky-coordinate region drawing, and richer multi-format figure export options still need to be
completed.

## Required Subsystems

### 1. Workbench Sessions

A workbench session is a durable, per-user object keyed by a server-generated
`session_id`.

Minimum session fields:

- `session_id`
- `user_id`
- `source_url`
- `filename`
- `archive`
- `project_code`
- `mous_uid`
- `created_at`
- `last_accessed_at`
- `status`
- `cache_path`
- `metadata`
- `evidence`

The frontend should open `/workbench/{session_id}` instead of keeping all state
inside a chat message.

### 2. Cube Engine

The backend should own expensive and science-sensitive operations:

- FITS access, cache, and eviction
- WCS parsing
- spectral-axis parsing and unit conversion
- moment calculations
- channel renders and downsampled tiles
- spectrum extraction
- aperture and RMS-region statistics
- PV extraction
- contour generation
- export artifact generation

Initial implementation should live in `services/cube_workbench.py`. Existing
helpers in `services/fits_service.py`, `services/fits_processing.py`,
`services/splatalogue.py`, and `services/notebook_gen.py` should be reused where
possible instead of duplicating science logic.

### 3. API Surface

First-class endpoints:

- `POST /api/workbench/session`
  Create a session from a FITS/product URL or archive product identifier.

- `GET /api/workbench/{session_id}/metadata`
  Return WCS, beam, pixel scale, cube shape, spectral axis, units, and evidence.

- `POST /api/workbench/{session_id}/render`
  Render channel, moment, or image views with colormap/stretch/contour options.

- `POST /api/workbench/{session_id}/jobs`
  Start a durable asynchronous prepare/render/spectrum/PV/line/export operation.

- `GET /api/workbench/{session_id}/jobs/{job_id}`
  Poll queued/running/completed job state, progress, metrics, result, and errors.

- `DELETE /api/workbench/{session_id}/jobs/{job_id}`
  Request cancellation for queued/running job work.

- `POST /api/workbench/{session_id}/spectrum`
  Extract a spectrum from pixel, sky coordinate, aperture, or region.

- `POST /api/workbench/{session_id}/pv-slice`
  Extract a PV slice from two points or a stored path.

- `POST /api/workbench/{session_id}/line-overlays`
  Query Splatalogue using rest/observed frequency, redshift, tolerance, and
  optional molecule filters.

- `POST /api/workbench/{session_id}/export`
  Generate reproducible artifacts: CASA script, CARTA notes, DS9 region/command,
  Python notebook, PNG/PDF figure, CSV spectrum.

### 4. Frontend Route

Add a dedicated Next.js route:

- `ui-pro/src/app/workbench/[sessionId]/page.tsx`

Expected layout:

- top product identity strip: target, project, MOUS, filename, status
- central image/cube viewer with WCS-aware axes
- left tool rail: channel, moment, aperture, PV, contour, RMS, line ID
- right evidence/export panel
- bottom spectrum and channel strip

The existing modal can stay as a quick-look entry point, but it should offer
`Open Workbench` once a session exists.

### 5. Large-Cube Strategy

The modal path is intentionally capped at 50 MB. The first-class workbench needs
a staged strategy:

1. Header-only inspection with HTTP range requests.
2. If product is small enough, cache full FITS on the server.
3. If product is large, generate downsampled preview products in a background
   job.
4. Keep full-resolution operations opt-in and bounded by user/session quotas.
5. Evict old cache files by size and age.

Recommended starting limits:

- quick-look direct render: 50 MB
- workbench cached product: 500 MB per product
- per-user cache: 2 GB
- session TTL: 24 hours

Current implementation has step 1 for workbench session creation using bounded,
streamed range bytes and first image/cube HDU header detection. It has step 2
with a per-request 500 MB default cap and explicit user-triggered preparation.
It also has in-process durable job polling for long workbench actions plus
per-user quota eviction for staged products. Step 3 and production worker
orchestration are still pending.

### 6. Trust Layer

Every workbench response should include an `evidence` object:

- `source_url`
- `archive`
- `headers_inspected`
- `wcs_status`
- `spectral_axis_status`
- `downloaded_bytes`
- `operations`
- `assumptions`
- `warnings`
- `confidence`

The frontend should display this as a persistent panel, not only as prose.

## Implementation Phases

### Phase A: System Boundary

- Add `services/cube_workbench.py`
- Add workbench session and metadata API endpoints
- Keep current modal behavior working
- Add `Open Workbench` entry point from ALMA product rows
- Status: implemented.

### Phase A2: Staged Data Boundary

- Add bounded server-side FITS cache preparation
- Persist cache status in session metadata
- Compute aperture spectra from prepared cubes
- Compute PV-slice preview images from prepared cubes
- Compute channel/image/moment preview images from prepared cubes
- Compute render-time RMS and contour levels
- Persist and apply an RMS region for render-time noise estimates
- Status: implemented for local/server-staged products under the configured
  cache cap. Bounded remote header range reads are implemented for session
  metadata, but pixel-data range reads, distributed/background worker
  orchestration, and preview-product generation are not complete.

### Phase B: Interactive Science Controls

- Dedicated route
- Channel slider
- Moment 0/1/2 controls
- Contour controls
- Spectrum extraction by click/aperture
- RMS region controls
- Status: partially implemented. The dedicated route, channel/moment controls,
  contour controls, aperture spectrum planning, PV path planning, and cached
  render products exist. RMS-region controls are implemented as persisted pixel
  boxes and are applied to render statistics/contours. The rendered image now
  supports click-to-set aperture plus drag-to-set RMS boxes and PV lines in
  pixel coordinates. Sky-coordinate region drawing and exact plot-margin-aware
  selection remain.

### Phase C: WCS and Line Science

- WCS-aware axes in rendered figures
- Spectral-axis conversion and labeling
- Redshift-aware Splatalogue overlays
- Common-line presets
- Status: partially implemented. Spectral-axis metadata and redshift-aware line
  lookup exist. Common-line presets now drive redshift-aware Splatalogue queries,
  computed spectrum plots show line markers, and cached channel/moment renders
  annotate matching line labels.

### Phase D: Export and Reproducibility

- CASA script
- DS9 command/regions
- CARTA session notes
- Python notebook
- PNG/PDF exports
- CSV spectrum export
- Status: partially implemented. CASA, CARTA notes, DS9 command/region, Python,
  notebook, computed spectrum CSV, and current render PNG artifacts are generated
  and downloadable for prepared products. PDF/multi-panel figure exports remain.

### Phase E: Scale

- Durable operation jobs for long products
- Distributed/background workers for large products
- Cache eviction
- Per-user quotas
- Progress events for long product preparation
- Status: partially implemented. Prepare/render/spectrum/PV can now run through
  durable in-process jobs with polling, progress, cancel requests, and terminal
  results. Staged products now have per-user cache accounting, default 2 GB
  quota enforcement, age/size eviction, and evicted-session state updates.
  Session creation now uses bounded remote FITS header range reads, including
  extension-HDU image/cube header detection. A production worker queue,
  resumability after restart, pixel-data range-read previews, and generated
  downsampled preview products remain.

## Decisions Needed

Defaults are enough to begin, but final production behavior depends on:

- maximum per-user disk cache
- whether full-resolution products are allowed in cloud deployment
- preferred first workflow: line cube, continuum image, or proposal/data triage
- whether users may launch local CASA/CARTA or only receive scripts/instructions

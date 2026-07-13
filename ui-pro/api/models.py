"""Pydantic request models for the QUASAR API.

Extracted verbatim from ``api/main.py`` during the P1 monolith split. These are
pure schema definitions with no application dependencies, so every router / the
SSE pipeline can import them without risk of a circular import.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    model: Optional[str] = "gpt-oss-120b"
    grounded_summary: bool = False
    web_search: bool = True


class RegisterRequest(BaseModel):
    username: str
    password: str
    email: Optional[str] = None
    display_name: Optional[str] = None


class LoginRequest(BaseModel):
    username: str
    password: str


class GoogleLoginRequest(BaseModel):
    credential: str


class FitsPreviewRequest(BaseModel):
    url: str
    filename: Optional[str] = None


class WorkbenchSessionCreate(BaseModel):
    source_url: str
    filename: Optional[str] = None
    project_code: Optional[str] = None
    mous_uid: Optional[str] = None


class WorkbenchRenderRequest(BaseModel):
    mode: Optional[str] = "image"
    channel: Optional[int] = None
    moment: Optional[int] = None
    colormap: Optional[str] = "inferno"
    stretch: Optional[str] = "asinh"
    contour_sigma: Optional[List[float]] = None
    rms_region: Optional[Dict[str, float]] = None


class WorkbenchPrepareRequest(BaseModel):
    max_bytes: Optional[int] = None
    user_cache_bytes: Optional[int] = None
    cache_ttl_seconds: Optional[int] = None
    force: Optional[bool] = False


class WorkbenchSpectrumRequest(BaseModel):
    x_pixel: Optional[float] = None
    y_pixel: Optional[float] = None
    aperture_radius_pixels: Optional[float] = 3.0
    aperture_radius_arcsec: Optional[float] = None
    max_points: Optional[int] = 512


class WorkbenchPvSliceRequest(BaseModel):
    path: Optional[List[Dict[str, float]]] = None
    width_pixels: Optional[float] = 3.0
    max_points: Optional[int] = 512


class WorkbenchLineOverlayRequest(BaseModel):
    observed_frequency_ghz: Optional[float] = None
    line_preset_key: Optional[str] = None
    redshift: Optional[float] = 0.0
    tolerance_ghz: Optional[float] = 0.01
    top_n: Optional[int] = 8


class WorkbenchExportRequest(BaseModel):
    formats: Optional[List[str]] = None


class WorkbenchJobStartRequest(BaseModel):
    operation: str
    payload: Optional[Dict[str, Any]] = None


class SpectralTargetResolveRequest(BaseModel):
    target_name: str
    redshift: Optional[float] = None
    ra_deg: Optional[float] = None
    dec_deg: Optional[float] = None
    # Also probe NOIRLab SPARCL for optical spectra at the resolved position
    # (drives the "optical spectrum available" chip on /spectral-lines).
    include_sparcl: bool = False


class SpectralLineJobRequest(BaseModel):
    operation: str
    payload: Optional[Dict[str, Any]] = None


class ProviderKeySaveRequest(BaseModel):
    provider: str
    api_key: str
    token_limit: Optional[int] = None


class ProviderKeyLimitRequest(BaseModel):
    token_limit: Optional[int] = None


class IssueReportCreateRequest(BaseModel):
    run_id: str
    message_id: str
    category: str
    description: str
    include_context: bool = False
    prompt_excerpt: Optional[str] = ""
    response_excerpt: Optional[str] = ""
    technical_context: Optional[Dict[str, Any]] = None


class IssueReportUpdateRequest(BaseModel):
    status: Optional[str] = None
    admin_notes: Optional[str] = None


class ConversationCreate(BaseModel):
    title: Optional[str] = "New Chat"
    model: Optional[str] = None


class ConversationTitleUpdate(BaseModel):
    title: str


class PlanFeedbackRequest(BaseModel):
    conversation_id: str
    approve: bool = False
    feedback: str = ""


class UserToolRequest(BaseModel):
    name: str
    description: str
    code: str
    api_key_name: Optional[str] = None
    api_key_value: Optional[str] = None


class MCPServerRequest(BaseModel):
    name: str
    transport: str = "stdio"
    command: Optional[str] = None
    args: List[str] = []
    url: Optional[str] = None
    env: Dict[str, str] = {}


class DatalabTiledSearchRequest(BaseModel):
    catalog: str
    table: str
    ra_min: float
    ra_max: float
    dec_min: float
    dec_max: float
    tile_radius_deg: float = 2.0
    step_deg: float = 0.05
    color_cut: Optional[dict] = None
    value_cuts: Optional[list] = None
    morphology: Optional[dict] = None
    peak_threshold: float = 3.0
    max_tiles: int = 64
    candidate_budget: int = 50
    confirm: bool = False

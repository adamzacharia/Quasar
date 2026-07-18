// --- Core Message Types ---
import type { ThoughtStep } from "@/components/ThoughtProcessWidget";

export type MessageRole = "user" | "assistant" | "system";

export type MessageType = "text" | "data" | "papers" | "tool_call" | "image" | "plotly" | "critique" | "notebook" | "web_sources";

/** Card kinds that carry a 1-5 star rating (Feature 4). `web_sources` is
 *  deliberately excluded — it is a provider-attribution strip, not a science
 *  block, so the eval-mode gate must not demand a star on it. Mirrors
 *  services/block_identity.py::RATEABLE_BLOCK_KINDS. */
export type BlockKind = "text" | "data" | "plotly" | "image" | "papers" | "notebook";
export const RATEABLE_BLOCK_KINDS: readonly BlockKind[] = ["text", "data", "plotly", "image", "papers", "notebook"];

/** Standard Plotly figure JSON (traces + layout) from "plotly" SSE events. */
export interface PlotlyFigureSpec {
    data: unknown[];
    layout?: Record<string, unknown>;
}

/** Cross-nav payload riding on "plotly" SSE events (e.g. a SPARCL spectrum
 * card deep-linking into the Spectral Line Explorer). */
export interface PlotlyCardMeta {
    sparcl_id?: string;
    spectype?: string;
    redshift?: number | null;
    line_explorer_url?: string;
    [key: string]: unknown;
}

export interface NotebookData {
    title: string;
    data: Record<string, unknown>;
}

export interface CommandTag {
    type: "archive" | "paper" | "search";
    label: string;
    icon: string;
    color: string;
}

export interface MocOverlay {
    id: string;
    name?: string;
    color?: string;
    order?: number;
    /** MOC geometry in Aladin Lite's A.MOCFromJSON format: {"<order>": [cell, ...]}. */
    mocJson: Record<string, number[]>;
}

export interface BlinkFrame {
    url: string;
    label?: string;
}

export interface HipsImageMeta {
    kind?: string;
    ra?: number;
    dec?: number;
    fovDeg?: number;
    survey?: string;
    /** Raw FITS download URL for SIA cutouts (Data Lab host, linked directly). */
    fitsUrl?: string;
    /** Survey footprint overlays drawn on the interactive view. */
    mocs?: MocOverlay[];
    /** Ordered, WCS-matched frames for the blink comparator (kind "blink"). */
    frames?: BlinkFrame[];
    /** Registration/coverage caveats from the epoch service (CX-18) — shown
     *  on the card so an unregistered frame can't read as a real transient. */
    warnings?: string[];
}

export interface Message {
    id: string;
    role: MessageRole;
    content: string;
    type: MessageType;
    timestamp: Date;
    dataTable?: DataTableResult;
    papers?: Paper[];
    notebookData?: NotebookData;
    toolCall?: ToolCall;
    imageUrl?: string;
    runMeta?: import("./api").ChatRunMeta;
    imageCaption?: string;
    imageMeta?: HipsImageMeta;
    plotlySpec?: PlotlyFigureSpec;   // interactive figure ("plotly" messages)
    plotlyTitle?: string;
    plotlyPngFallback?: string;      // static PNG used if plotly can't render
    plotlyMeta?: PlotlyCardMeta;     // cross-nav payload (e.g. SPARCL → /spectral-lines)
    thinkingSteps?: ThoughtStep[];
    thinking?: string;
    thinkingDuration?: number;
    /** Raw request provenance for this turn's tool calls (Feature 1). Threaded
     *  from the live `tool_trace` SSE event AND rebuilt from the persisted
     *  `rich_meta.toolTrace`, so it survives a page reload. */
    toolTrace?: import("./api").ToolTraceCall[];
    /** The exact request behind THIS card (Feature 1) — papers/plotly/image
     *  messages. Data cards carry it on `dataTable.request` instead. */
    request?: import("./api").ToolRequest;
    /** Stable identity for this block/card (Feature 4). Minted by the backend
     *  at emission and persisted in rich_meta, so the id a rating was keyed to
     *  survives a reload. Absent on turns that predate stable ids, and on
     *  client-only messages (the user's own prompt). */
    blockId?: string;
    /** Which rateable card this is — mirrors `type`, but only for the kinds the
     *  eval gate counts (text/data/plotly/image/papers/notebook). */
    blockKind?: BlockKind;
    attachmentPreviews?: string[];  // base64 data URLs for images
    attachmentNames?: string[];     // names of attached documents
    webSources?: WebSource[];       // clickable source cards from web search
    webImages?: WebImage[];         // image grid from web search
    webProvider?: string;           // confirmed web provider returned by router
    webImageProvider?: string;      // provider used for image enrichment
    webSearchType?: string;         // provider-specific search mode, e.g. Exa deep
    webQuery?: string;              // original web query or URL
    usageTokens?: number;           // provider-reported tokens used for this response
}

export interface WebSource {
    title: string;
    url: string;
    snippet: string;
    evidenceQuality?: EvidenceQuality;
}

export interface EvidenceQuality {
    score: number;
    tier: "primary" | "peer_reviewed" | "preprint" | "institutional" | "reference" | "general" | string;
    label: string;
    reason?: string;
    signals?: string[];
}

export interface WebImage {
    url: string;
    description: string;
    sourceUrl?: string;
    sourceTitle?: string;
}

export interface Conversation {
    id: string;
    title: string;
    createdAt: Date;
    updatedAt: Date;
    messages: Message[];
    model: string;
    isStarred?: boolean;
}

export interface DataTableMetric {
    label: string;
    value: string | number;
    color?: string;
}

export interface DataTableResult {
    metrics: DataTableMetric[];
    columns: string[];
    rows: Record<string, string | number>[];
    sourceName: string;
    warnings?: string[];
    partial?: boolean;
    /** Full-result integrity (Feature 2). `rows` is a PREVIEW — capped at 10k,
     *  display columns only, values clipped. These say what the preview is a
     *  preview OF, so the UI never passes it off as the complete dataset. */
    /** Result-store id addressing the full frame; feeds /api/results/{id}/export.csv.
     *  Absent = no server-side export is possible for this card. */
    resultId?: string;
    /** Row count of the FULL result, not of `rows`. */
    totalRows?: number;
    /** Row count actually present in `rows`. */
    displayedRows?: number;
    /** True when totalRows > displayedRows — `rows` is missing data. */
    truncated?: boolean;
    /** True when the STORED frame is itself a truncated slice of the remote
     *  result (e.g. SIA capped rows before storage) — the export contains
     *  every stored row but not every matching row (f2-CX-21). */
    upstreamPartial?: boolean;
    /** Remote row count before the upstream cap, when the producer knows it. */
    upstreamTotal?: number;
    /** The exact request that produced this table (Feature 1). Persisted with
     *  the card in messages.metadata, so it survives a reload. */
    request?: import("./api").ToolRequest;
    toolName?: string;
    tableKind?: string;   // e.g. "alma_products" or "alma_project_picker"
    archiveLink?: string;  // Footer link to full dataset on ALMA/ESO portal
    hasRowLinks?: boolean; // Whether rows include per-row _link field
    hasPreview?: boolean;  // Whether rows include per-row _preview sky thumbnail URL
    demographics?: {       // Distribution data for auto-generated charts
        bands?: Record<string, number>;
        projects?: Record<string, number>;
        telescopes?: Record<string, number>;
        instruments?: Record<string, number>;
        // RA/Dec positions for the sky map; `i` is the row's positional index
        // in rows[] and `label` its identifier column (click-to-inspect).
        skyCoords?: { ra: number; dec: number; i?: number; label?: string }[];
        // Per-observation STC-S footprints (obscore s_region), drawn on the sky
        // map via A.footprintsFromSTCS; `i` ties a footprint back to its row.
        skyFootprints?: { stcs: string; i?: number; label?: string }[];
        skyFootprintsTruncated?: boolean;
        observationYears?: Record<string, number>;  // Year → count timeline
        scienceCategories?: Record<string, number>; // Category → count
        resolutionBins?: Record<string, number>;    // Resolution range → count
    };
    fitsEstimate?: number; // Estimated number of FITS files available
}

export interface Paper {
    id: string;
    title: string;
    authors: string;
    year: number;
    journal: string;
    citationCount: number;
    type: "journal" | "arxiv" | "radio";
    bibcode?: string;
    arxivId?: string;
    doi?: string;
    abstract?: string;
    relevanceScore?: number;
    isRadioPaper?: boolean;
    // OpenAlex enrichment fields
    fwci?: number | null;
    citationPercentile?: number | null;
    isTop1Percent?: boolean;
    isTop10Percent?: boolean;
    funders?: { name: string; id: string }[];
    oaPdfUrl?: string;
    observationLinks?: ObservationPaperLink[];
}

export interface ObservationPaperLink {
    identifier: string;
    identifierType?: string;
    relation?: string;
    confidence?: string;
    adsQuery?: string;
}

export interface ToolCall {
    id: string;
    name: string;
    displayName: string;
    status: "running" | "completed" | "error";
    input?: Record<string, unknown>;
    output?: string;
    /** The exact request this call made (Feature 1: query provenance). */
    request?: import("./api").ToolRequest;
}

export interface AppSettings {
    model: string;
    availableModels: string[];
}

// ── Task Execution UI Types (Perplexity Computer-style) ──────────────────

export interface TaskItem {
    id: string;
    description: string;
    status: "pending" | "running" | "completed" | "error";
    agentType: string;
    icon?: string;
    detail?: string;
    groupId?: string;
}

export interface TaskGroup {
    groupId: string;
    title: string;
    taskIds: string[];
    tasks: TaskItem[];
}

export interface TaskChecklistItem {
    id: string;
    description: string;
    status: "pending" | "running" | "completed" | "error";
    agentType: string;
}

export interface TaskChecklist {
    title: string;
    tasks: TaskChecklistItem[];
}

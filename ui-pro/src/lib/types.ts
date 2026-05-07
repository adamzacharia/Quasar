// --- Core Message Types ---
import type { ThoughtStep } from "@/components/ThoughtProcessWidget";

export type MessageRole = "user" | "assistant" | "system";

export type MessageType = "text" | "data" | "papers" | "tool_call" | "image" | "critique" | "notebook" | "web_sources";

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
    imageCaption?: string;
    thinkingSteps?: ThoughtStep[];
    attachmentPreviews?: string[];  // base64 data URLs for images
    attachmentNames?: string[];     // names of attached documents
    webSources?: WebSource[];       // clickable source cards from web search
    webImages?: WebImage[];         // image grid from web search
}

export interface WebSource {
    title: string;
    url: string;
    snippet: string;
}

export interface WebImage {
    url: string;
    description: string;
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
    archiveLink?: string;  // Footer link to full dataset on ALMA/ESO portal
    hasRowLinks?: boolean; // Whether rows include per-row _link field
    hasPreview?: boolean;  // Whether rows include per-row _preview sky thumbnail URL
    demographics?: {       // Distribution data for auto-generated charts
        bands?: Record<string, number>;
        projects?: Record<string, number>;
        telescopes?: Record<string, number>;
        instruments?: Record<string, number>;
        skyCoords?: { ra: number; dec: number }[];  // RA/Dec positions for sky map
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
}

export interface ToolCall {
    id: string;
    name: string;
    displayName: string;
    status: "running" | "completed" | "error";
    input?: Record<string, unknown>;
    output?: string;
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

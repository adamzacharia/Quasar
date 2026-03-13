// --- Core Message Types ---
import type { ThoughtStep } from "@/components/ThoughtProcessWidget";

export type MessageRole = "user" | "assistant" | "system";

export type MessageType = "text" | "data" | "papers" | "tool_call" | "image" | "critique" | "notebook";

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

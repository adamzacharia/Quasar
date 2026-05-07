"use client";

import { useState, useEffect } from "react";
import { Download, CheckCircle2 } from "lucide-react";

interface DownloadProgressData {
    filename: string;
    downloaded_bytes: number;
    total_bytes: number | null;
    speed_kbps: number;
    percent: number | null;
}

interface DownloadProgressProps {
    data: DownloadProgressData;
}

function formatBytes(bytes: number): string {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatSpeed(kbps: number): string {
    if (kbps < 1024) return `${kbps.toFixed(0)} KB/s`;
    return `${(kbps / 1024).toFixed(1)} MB/s`;
}

export function DownloadProgress({ data }: DownloadProgressProps) {
    const [done, setDone] = useState(false);
    const [fadeOut, setFadeOut] = useState(false);

    const percent = data.percent ?? 0;
    const isDone = percent >= 100;

    useEffect(() => {
        if (isDone && !done) {
            setDone(true);
            // Auto-fade after 3 seconds
            const t = setTimeout(() => setFadeOut(true), 3000);
            return () => clearTimeout(t);
        }
    }, [isDone, done]);

    return (
        <div
            className={`my-2 rounded-xl border overflow-hidden transition-all duration-500 ${
                fadeOut ? "opacity-0 max-h-0" : "opacity-100 max-h-32"
            }`}
            style={{
                background: "var(--q-glass-bg)",
                borderColor: isDone
                    ? "rgba(16, 185, 129, 0.3)"
                    : "var(--q-glass-border)",
            }}
        >
            <div className="px-4 py-3 flex items-center gap-3">
                {/* Icon */}
                <div
                    className={`w-8 h-8 rounded-lg flex items-center justify-center shrink-0 ${
                        isDone
                            ? "bg-emerald-500/15 text-emerald-400"
                            : "bg-primary/15 text-primary"
                    }`}
                >
                    {isDone ? (
                        <CheckCircle2 className="w-4 h-4" />
                    ) : (
                        <Download className="w-4 h-4 animate-bounce" />
                    )}
                </div>

                {/* Info */}
                <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between mb-1">
                        <span
                            className="text-xs font-medium truncate"
                            style={{ color: "var(--q-text)" }}
                        >
                            {data.filename}
                        </span>
                        <span
                            className="text-[10px] font-mono shrink-0 ml-2"
                            style={{ color: "var(--q-text-muted)" }}
                        >
                            {isDone
                                ? "Complete"
                                : `${formatSpeed(data.speed_kbps)}`}
                        </span>
                    </div>

                    {/* Progress bar */}
                    <div
                        className="h-1.5 rounded-full overflow-hidden"
                        style={{
                            background: "var(--q-border)",
                        }}
                    >
                        <div
                            className={`h-full rounded-full transition-all duration-300 ease-out ${
                                isDone
                                    ? "bg-emerald-500"
                                    : "bg-gradient-to-r from-primary to-accent-purple"
                            }`}
                            style={{ width: `${percent}%` }}
                        />
                    </div>

                    {/* Stats */}
                    <div className="flex items-center justify-between mt-1">
                        <span
                            className="text-[10px] font-mono"
                            style={{ color: "var(--q-text-muted)" }}
                        >
                            {formatBytes(data.downloaded_bytes)}
                            {data.total_bytes
                                ? ` / ${formatBytes(data.total_bytes)}`
                                : ""}
                        </span>
                        <span
                            className="text-[10px] font-mono"
                            style={{ color: "var(--q-text-muted)" }}
                        >
                            {percent.toFixed(0)}%
                        </span>
                    </div>
                </div>
            </div>
        </div>
    );
}

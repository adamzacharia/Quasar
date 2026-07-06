/* eslint-disable @next/next/no-img-element */
"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Download, Minus, Plus, RotateCcw, X } from "lucide-react";

const MIN_ZOOM = 0.5;
const MAX_ZOOM = 8;
const ZOOM_STEP = 1.25;

function clampZoom(value: number): number {
    return Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, value));
}

function downloadNameFor(src: string): string {
    try {
        const pathname = src.startsWith("data:") ? "" : new URL(src, window.location.href).pathname;
        const basename = pathname.split("/").pop() || "";
        if (basename && /\.[a-z0-9]{2,5}$/i.test(basename)) return basename;
    } catch {
        // fall through to default
    }
    return "image.png";
}

interface ImageLightboxProps {
    src: string;
    caption?: string;
    onClose: () => void;
}

/**
 * Full-screen image lightbox: dark backdrop, scroll-wheel / button zoom
 * (0.5x-8x), drag-to-pan when zoomed, Esc / click-outside / X to close,
 * and a Download button. Dependency-free.
 */
export function ImageLightbox({ src, caption, onClose }: ImageLightboxProps) {
    const [zoom, setZoom] = useState(1);
    const [offset, setOffset] = useState({ x: 0, y: 0 });
    const [dragging, setDragging] = useState(false);
    const stageRef = useRef<HTMLDivElement>(null);
    const dragRef = useRef<{ startX: number; startY: number; originX: number; originY: number } | null>(null);
    // Mirror of `zoom` so the (stable) wheel listener can read the latest value.
    const zoomRef = useRef(1);

    // Clamp, apply, and recenter once zoomed back out to fit.
    const applyZoom = useCallback((next: number) => {
        const clamped = clampZoom(next);
        zoomRef.current = clamped;
        setZoom(clamped);
        if (clamped <= 1) setOffset({ x: 0, y: 0 });
    }, []);

    // Esc closes
    useEffect(() => {
        const onKeyDown = (event: KeyboardEvent) => {
            if (event.key === "Escape") onClose();
        };
        window.addEventListener("keydown", onKeyDown);
        return () => window.removeEventListener("keydown", onKeyDown);
    }, [onClose]);

    // Lock body scroll while open
    useEffect(() => {
        const previous = document.body.style.overflow;
        document.body.style.overflow = "hidden";
        return () => {
            document.body.style.overflow = previous;
        };
    }, []);

    // Scroll-wheel zoom (non-passive so we can preventDefault page scroll)
    useEffect(() => {
        const stage = stageRef.current;
        if (!stage) return;
        const onWheel = (event: WheelEvent) => {
            event.preventDefault();
            const factor = event.deltaY < 0 ? ZOOM_STEP : 1 / ZOOM_STEP;
            applyZoom(zoomRef.current * factor);
        };
        stage.addEventListener("wheel", onWheel, { passive: false });
        return () => stage.removeEventListener("wheel", onWheel);
    }, [applyZoom]);

    const onPointerDown = (event: React.PointerEvent<HTMLImageElement>) => {
        if (zoom <= 1) return;
        event.preventDefault();
        event.currentTarget.setPointerCapture(event.pointerId);
        dragRef.current = { startX: event.clientX, startY: event.clientY, originX: offset.x, originY: offset.y };
        setDragging(true);
    };

    const onPointerMove = (event: React.PointerEvent<HTMLImageElement>) => {
        const drag = dragRef.current;
        if (!drag) return;
        setOffset({
            x: drag.originX + (event.clientX - drag.startX),
            y: drag.originY + (event.clientY - drag.startY),
        });
    };

    const endDrag = () => {
        dragRef.current = null;
        setDragging(false);
    };

    const toolbarButton = "p-2 rounded-lg text-slate-300 hover:text-white hover:bg-white/10 transition-colors";

    return (
        <div
            role="dialog"
            aria-modal="true"
            aria-label={caption || "Image viewer"}
            className="fixed inset-0 z-[100] flex flex-col bg-black/85 backdrop-blur-sm"
            onClick={onClose}
        >
            {/* Toolbar */}
            <div
                className="flex items-center justify-end gap-1 px-3 py-2.5 shrink-0"
                onClick={(event) => event.stopPropagation()}
            >
                <button type="button" onClick={() => applyZoom(zoom / ZOOM_STEP)}
                    className={toolbarButton} aria-label="Zoom out" title="Zoom out">
                    <Minus className="h-4 w-4" />
                </button>
                <span className="min-w-[3.25rem] text-center text-xs text-slate-400 tabular-nums select-none">
                    {Math.round(zoom * 100)}%
                </span>
                <button type="button" onClick={() => applyZoom(zoom * ZOOM_STEP)}
                    className={toolbarButton} aria-label="Zoom in" title="Zoom in">
                    <Plus className="h-4 w-4" />
                </button>
                <button type="button" onClick={() => applyZoom(1)}
                    className={toolbarButton} aria-label="Reset zoom" title="Reset zoom">
                    <RotateCcw className="h-4 w-4" />
                </button>
                <a href={src} download={downloadNameFor(src)}
                    className={toolbarButton} aria-label="Download image" title="Download image">
                    <Download className="h-4 w-4" />
                </a>
                <button type="button" onClick={onClose}
                    className={toolbarButton} aria-label="Close" title="Close (Esc)">
                    <X className="h-4 w-4" />
                </button>
            </div>

            {/* Stage — click on empty space closes; image clicks don't */}
            <div ref={stageRef} className="relative flex-1 min-h-0 overflow-hidden flex items-center justify-center px-4 pb-2">
                <img
                    src={src}
                    alt={caption || "Full-size image"}
                    draggable={false}
                    onClick={(event) => event.stopPropagation()}
                    onPointerDown={onPointerDown}
                    onPointerMove={onPointerMove}
                    onPointerUp={endDrag}
                    onPointerCancel={endDrag}
                    className="max-h-full max-w-full object-contain select-none touch-none"
                    style={{
                        transform: `translate(${offset.x}px, ${offset.y}px) scale(${zoom})`,
                        cursor: zoom > 1 ? (dragging ? "grabbing" : "grab") : "default",
                        transition: dragging ? "none" : "transform 120ms ease-out",
                    }}
                />
            </div>

            {/* Caption */}
            {caption && (
                <div
                    className="shrink-0 px-6 py-3 text-center text-xs text-slate-300"
                    onClick={(event) => event.stopPropagation()}
                >
                    {caption}
                </div>
            )}
        </div>
    );
}

"use client";

import { useEffect, useRef } from "react";

/*
 * Research "working" mark: four dots whose move says what the turn is doing.
 * Positions are px inside an 18px box, origin at the centre, y down.
 *
 *   starting  -> swap      2x2; diagonal pairs swing 180deg through the centre
 *   thinking  -> infinity  a fading trail traces a figure-8
 *   searching -> snake     a trail chases round the square's edge
 *   querying  -> cradle    Newton's cradle: the outer dots swing and click
 *   mcp       -> pinwheel  the whole grid turns 90deg at a time
 *   reading   -> fountain  dots arc up and over, one after another
 *   wrapping  -> wave      a row of three, each dot hopping twice in turn
 *   waiting   -> spin      three dots turn 120deg at a time (a slow step)
 *   notice    -> heartbeat the grid double-pulses
 *
 * swap, wave and the morph between them are traced frame by frame from a
 * reference recording. Every move is periodic and returns to its start pose,
 * so a change of activity waits for the end of the current cycle (and at
 * least one full cycle), then blends into the next move's start pose.
 */

export type DotsActivity =
    | "starting" | "thinking" | "searching" | "querying" | "mcp"
    | "reading" | "wrapping" | "waiting" | "notice";

type Pose = [number, number, number]; // x, y, scale (0 = hidden)
type Row = number[];
type Move = { period: number; pose: (t: number) => Pose[] };

/** Piecewise-linear sample of rows shaped [t, v0, v1, ...]. */
function sample(table: Row[], t: number): number[] {
    if (t <= table[0][0]) return table[0].slice(1);
    for (let i = 1; i < table.length; i++) {
        const b = table[i];
        if (t <= b[0]) {
            const a = table[i - 1];
            const k = (t - a[0]) / (b[0] - a[0]);
            return a.slice(1).map((v, j) => v + (b[j + 1] - v) * k);
        }
    }
    return table[table.length - 1].slice(1);
}

const ease = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2);
const clamp01 = (t: number) => Math.max(0, Math.min(1, t));
const RAD = Math.PI / 180;
const HIDDEN: Pose = [0, 0.3, 0];

const H = 4.5; // half the 2x2 spacing
const DOT = 4.5;
// Dot order: 0 = TL, 1 = BR, 2 = TR, 3 = BL.
const CORNERS: [number, number][] = [[-H, -H], [H, H], [H, -H], [-H, H]];

// One 180deg pair swing: [t, degrees, radius factor].
const SWING: Row[] = [
    [0, 0, 1], [0.067, 15, 0.82], [0.133, 54, 0.42], [0.2, 90, 0.4],
    [0.267, 135, 0.74], [0.333, 155, 0.93], [0.4, 171, 1], [0.467, 180, 1],
];

// The wave's row.
const ROW_Y = 2.25;
const ROW_X = 7.1;

// One dot's double hop: [t, fraction of WAVE_AMP].
const HOP: Row[] = [
    [0, 0], [0.033, 0.21], [0.1, 0.53], [0.167, 0.85], [0.233, 0.96], [0.433, 0.96],
    [0.5, 0.85], [0.533, 0.64], [0.6, 0.32], [0.633, 0.11], [0.7, 0.11], [0.733, 0.21],
    [0.8, 0.53], [0.867, 0.75], [0.9, 0.85], [1.1, 0.85], [1.167, 0.74], [1.233, 0.43],
    [1.3, 0.21], [1.367, 0],
];
const WAVE_AMP = 6.2;
const WAVE_DELAYS = [0, 0.15, 0.33];

const HEARTBEAT: Row[] = [[0, 1, 1], [0.12, 1.32, 1.15], [0.24, 0.92, 1], [0.36, 1.18, 1.08], [0.5, 1, 1], [1.3, 1, 1]];

const SNAKE_PATH: [number, number][] = [[-5.5, -5.5], [0, -5.5], [5.5, -5.5], [5.5, 0], [5.5, 5.5], [0, 5.5], [-5.5, 5.5], [-5.5, 0]];
const CRADLE_X = [-DOT, 0, DOT]; // touching balls, strings from y = -3

function squarePose(t: number): Pose[] {
    const c = t % 1.3;
    return CORNERS.map(([x, y], i) => {
        const pairA = i < 2;
        const [deg, r] = sample(SWING, pairA ? c : c - 0.65);
        // Visual counter-clockwise (y down) for pair A, clockwise for pair B.
        const th = (pairA ? 1 : -1) * deg * RAD;
        const cos = Math.cos(th), sin = Math.sin(th);
        return [r * (x * cos + y * sin), r * (-x * sin + y * cos), 1];
    });
}

const MOVES: Record<string, Move> = {
    swap: { period: 1.3, pose: squarePose },
    wave: {
        period: 2,
        pose: (t) => {
            const c = t % 2;
            const row = [-ROW_X, 0, ROW_X].map((x, i): Pose => [x, ROW_Y - WAVE_AMP * sample(HOP, c - WAVE_DELAYS[i])[0], 1]);
            return [...row, HIDDEN];
        },
    },
    spin: {
        period: 1,
        pose: (t) => {
            const n = Math.floor(t), e = ease(clamp01((t % 1) / 0.65));
            const r = 5.2 - 2 * Math.sin(Math.PI * e);
            const tri = [0, 1, 2].map((k): Pose => {
                const th = (-90 + 120 * k + 120 * (n + e)) * RAD;
                return [r * Math.cos(th), r * Math.sin(th), 1];
            });
            return [...tri, HIDDEN];
        },
    },
    snake: {
        period: 8 / 6,
        pose: (t) => [0, 1, 2, 3].map((k): Pose => {
            const q = (((t * 6 - k * 1.1) % 8) + 8) % 8;
            const i = Math.floor(q), f = q - i;
            const a = SNAKE_PATH[i], b = SNAKE_PATH[(i + 1) % 8];
            return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, 1 - k * 0.15];
        }),
    },
    cradle: {
        period: 1.2,
        pose: (t) => {
            const swing = 26 * Math.sin((2 * Math.PI * t) / 1.2);
            const balls = CRADLE_X.map((x0, i): Pose => {
                const a = (i === 0 ? Math.min(0, swing) : i === 2 ? Math.max(0, swing) : 0) * RAD;
                return [x0 + 6 * Math.sin(a), -3 + 6 * Math.cos(a), 1];
            });
            return [...balls, HIDDEN];
        },
    },
    infinity: {
        period: 1.8,
        pose: (t) => [0, 1, 2, 3].map((k): Pose => {
            const u = (2 * Math.PI * t) / 1.8 - k * 0.42;
            return [7 * Math.sin(u), 3.2 * Math.sin(2 * u), 1 - k * 0.13];
        }),
    },
    pinwheel: {
        period: 1,
        pose: (t) => {
            const n = Math.floor(t), e = ease(clamp01((t % 1) / 0.6));
            const r = H * Math.SQRT2 * (1 - 0.4 * Math.sin(Math.PI * e));
            return [-135, 45, -45, 135].map((b): Pose => {
                const th = (b + 90 * (n + e)) * RAD;
                return [r * Math.cos(th), r * Math.sin(th), 1];
            });
        },
    },
    heartbeat: {
        period: 1.3,
        pose: (t) => {
            const [r, s] = sample(HEARTBEAT, t % 1.3);
            return CORNERS.map(([x, y]): Pose => [x * r, y * r, s]);
        },
    },
    fountain: {
        period: 1.5,
        pose: (t) => {
            const arc = [0, 1, 2].map((k): Pose => {
                const f = ((t / 1.5 - k / 3) % 1 + 1) % 1;
                // Grows out of the left end and shrinks into the right, so the
                // wrap back to the start is invisible.
                return [-7 + 14 * f, 4 - 44 * f * (1 - f), Math.min(1, 2.2 * Math.sin(Math.PI * f))];
            });
            return [...arc, HIDDEN];
        },
    },
};

const MOVE_FOR: Record<DotsActivity, keyof typeof MOVES> = {
    starting: "swap",
    thinking: "infinity",
    searching: "snake",
    querying: "cradle",
    mcp: "pinwheel",
    reading: "fountain",
    wrapping: "wave",
    waiting: "spin",
    notice: "heartbeat",
};

// Traced morph from the 2x2 into the wave's row: per dot [t, x, y, scale].
// TL becomes the left dot, BR the middle, TR the right; BL merges in and
// disappears.
const MORPH: Row[][] = [
    [[0, -H, -H, 1], [0.17, 0, 0.3, 1], [0.5, 0, 0.3, 0.62], [0.54, 0, 0.3, 0.62], [0.64, 0, 1.6, 0.65], [0.67, 0, 1.6, 0.65],
        [0.9, -1.9, -2.25, 1], [1.04, -5.8, -1, 1], [1.17, -6.4, 1, 1], [1.27, -7.1, 1.6, 1], [1.37, -ROW_X, ROW_Y, 1]],
    [[0, H, H, 1], [0.17, 0, 0.3, 1], [0.5, 0, 0.3, 0.62], [0.54, 0, 0.3, 0.62], [0.64, 0, 1.6, 0.65], [0.67, 0, 1.6, 0.65],
        [0.9, -1.9, -2.25, 1], [1.04, 1.3, -2.9, 1], [1.17, 0, -1.6, 1], [1.27, 0.6, 1.6, 1], [1.37, 0, ROW_Y, 1]],
    [[0, H, -H, 1], [0.17, 0, 0.3, 1], [0.5, 0, 0.3, 0.62], [0.54, 0, 0.3, 0.62], [0.64, 0, 1.6, 0.65], [0.67, 0, 1.6, 0.65],
        [0.9, -1.9, -2.25, 1], [1.04, 1.3, -2.9, 1], [1.17, 3.9, -2.9, 1], [1.27, 5.8, -0.3, 1], [1.37, ROW_X, ROW_Y, 1]],
    [[0, -H, H, 1], [0.17, 0, 0.3, 1], [0.2, 0, 0.3, 0], [1.37, 0, 0.3, 0]],
];
const MORPH_LEN = 1.37;
const BLEND_LEN = 0.6;

/** Swoop each dot from pose A to pose B through a point near the centre. */
function blend(from: Pose[], to: Pose[], f: number): Pose[] {
    const e = ease(clamp01(f));
    return from.map((a0, i) => {
        const a = a0[2] === 0 ? [0, 0, 0] : a0;
        const b = to[i][2] === 0 ? [0, 0, 0] : to[i];
        const cx = (a[0] + b[0]) * 0.12, cy = (a[1] + b[1]) * 0.12;
        const u = 1 - e;
        const s = (a[2] + (b[2] - a[2]) * e) * (1 - 0.18 * Math.sin(Math.PI * e));
        return [u * u * a[0] + 2 * u * e * cx + e * e * b[0], u * u * a[1] + 2 * u * e * cy + e * e * b[1], s];
    });
}

/** Pose t seconds into the transition from move a to move b, and its length. */
function transition(a: string, b: string): { len: number; pose: (t: number) => Pose[] } {
    if (a === "swap" && b === "wave") return { len: MORPH_LEN, pose: (t) => MORPH.map((tb) => sample(tb, t) as Pose) };
    if (a === "wave" && b === "swap") return { len: MORPH_LEN, pose: (t) => MORPH.map((tb) => sample(tb, MORPH_LEN - t) as Pose) };
    const from = MOVES[a].pose(0), to = MOVES[b].pose(0);
    return { len: BLEND_LEN, pose: (t) => blend(from, to, t / BLEND_LEN) };
}

export function WorkingDots({ live, activity = "starting" }: { live: boolean; activity?: DotsActivity }) {
    const dots = useRef<(HTMLSpanElement | null)[]>([]);
    const target = useRef<string>(MOVE_FOR[activity] ?? "swap");
    useEffect(() => {
        target.current = MOVE_FOR[activity] ?? "swap";
    }, [activity]);

    useEffect(() => {
        const place = (pose: Pose[]) => pose.forEach(([x, y, s], i) => {
            const el = dots.current[i];
            if (el) el.style.transform = `translate(${x}px, ${y}px) scale(${s})`;
        });
        const still = !live || window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
        if (still) {
            place(squarePose(0));
            return;
        }
        // The first move is whatever the turn is doing when the mark mounts.
        let move = target.current;
        let moveStart = performance.now();
        let switchAt: number | null = null; // seconds into `move`
        let trans: { to: string; start: number; len: number; pose: (t: number) => Pose[] } | null = null;
        let frame = 0;

        const tick = (now: number) => {
            if (trans) {
                const t = (now - trans.start) / 1000;
                if (t < trans.len) {
                    place(trans.pose(t));
                    frame = requestAnimationFrame(tick);
                    return;
                }
                move = trans.to;
                moveStart = trans.start + trans.len * 1000;
                trans = null;
            }
            const m = MOVES[move];
            const t = (now - moveStart) / 1000;
            if (target.current === move) {
                switchAt = null;
            } else if (switchAt === null) {
                // Leave at a cycle boundary, after at least one full cycle.
                switchAt = Math.max(1, Math.ceil(t / m.period)) * m.period;
            }
            if (switchAt !== null && t >= switchAt) {
                const tr = transition(move, target.current);
                trans = { to: target.current, start: moveStart + switchAt * 1000, len: tr.len, pose: tr.pose };
                switchAt = null;
                place(tr.pose(Math.min(tr.len, (now - trans.start) / 1000)));
            } else {
                place(m.pose(t));
            }
            frame = requestAnimationFrame(tick);
        };
        frame = requestAnimationFrame(tick);
        return () => cancelAnimationFrame(frame);
    }, [live]);

    return (
        <span
            className={`relative inline-block h-[18px] w-[18px] shrink-0 ${live ? "text-primary" : ""}`}
            style={live ? undefined : { color: "var(--q-text-muted)" }}
            aria-hidden="true"
        >
            {CORNERS.map(([x, y], i) => (
                <span
                    key={i}
                    ref={(el) => { dots.current[i] = el; }}
                    className="absolute rounded-full"
                    style={{
                        left: "50%", top: "50%", width: DOT, height: DOT,
                        margin: `${-DOT / 2}px 0 0 ${-DOT / 2}px`,
                        background: "currentColor",
                        transform: `translate(${x}px, ${y}px)`,
                    }}
                />
            ))}
        </span>
    );
}

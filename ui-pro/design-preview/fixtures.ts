import type { Conversation, Message, DataTableResult } from "../src/lib/types";

export const PREVIEW_USER = { id: "design-preview", username: "preview", display_name: "Preview workspace", is_admin: false };
export const SAMPLE_TABLE: DataTableResult = {
  sourceName: "ALMA archive · sample data",
  metrics: [{ label: "Observations", value: 3 }, { label: "Band", value: 6 }, { label: "Targets", value: 1 }],
  columns: ["Target", "Band", "Frequency (GHz)", "Resolution (arcsec)", "Integration (min)"],
  rows: [
    { Target: "Sz65", Band: 6, "Frequency (GHz)": 230.538, "Resolution (arcsec)": 0.32, "Integration (min)": 42 },
    { Target: "Sz65", Band: 6, "Frequency (GHz)": 231.100, "Resolution (arcsec)": 0.47, "Integration (min)": 28 },
    { Target: "Sz65", Band: 6, "Frequency (GHz)": 229.800, "Resolution (arcsec)": 0.61, "Integration (min)": 36 },
  ],
  totalRows: 3, displayedRows: 3,
  request: { kind: "params", service: "Design preview", text: "Locally supplied sample rows. No archive query was executed." }
};
const time = new Date("2026-09-08T08:00:00Z");
function message(id: string, role: Message["role"], content: string, extra: Partial<Message> = {}): Message {
  return { id, role, content, type: "text", timestamp: time, ...extra };
}
const tableMessages: Message[] = [
  message("sample-table-user", "user", "Find ALMA observations of Sz65 in Band 6."),
  message("sample-table-answer", "assistant", "### Observations of Sz65\n\nThese sample rows show how archive results will appear in Quasar. Compare frequency coverage, angular resolution, and integration time in the table.\n\nThe table controls and query details use the existing components; the values are illustrative."),
  message("sample-table-data", "assistant", "", { type: "data", dataTable: SAMPLE_TABLE }),
];
const frequencies = Array.from({ length: 161 }, (_, i) => 230.1 + i * 0.005);
const flux = frequencies.map((f, i) => +(2.2 + Math.sin(i * 1.9) * 0.6 + Math.cos(i * 0.73) * 0.3 + 35 * Math.exp(-(((f - 230.538) / 0.024) ** 2))).toFixed(3));
const plotMessages: Message[] = [
  message("sample-plot-user", "user", "Inspect the CO(2–1) spectrum and line profile."),
  message("sample-plot-answer", "assistant", "### Spectral line profile\n\nThis synthetic spectrum is a local preview fixture. Hover to inspect values, drag to zoom, or reset the axes using Quasar’s existing Plotly controls."),
  message("sample-plot-card", "assistant", "", { type: "plotly", plotlyTitle: "CO(2–1) · synthetic spectrum", plotlySpec: { data: [{ x: frequencies, y: flux, type: "scatter", mode: "lines", name: "Synthetic flux", line: { color: "#838383", width: 1.5 } }], layout: { xaxis: { title: { text: "Frequency (GHz)" } }, yaxis: { title: { text: "Flux density (mJy)" } }, height: 390, margin: { l: 64, r: 24, t: 32, b: 54 }, showlegend: false } }, request: { kind: "params", service: "Design preview", text: "Synthetic Gaussian plus deterministic variation. No instrument data." } }),
];
const imageMessages: Message[] = [
  message("sample-image-user", "user", "Show the image preview and its inspection controls."),
  message("sample-image-answer", "assistant", "### Image inspection\n\nThis synthetic point-spread function exercises Quasar’s existing image card and full-screen viewer. It is a component fixture, not a sky observation."),
  message("sample-image-card", "assistant", "", { type: "image", imageUrl: "/sample-psf.svg", imageCaption: "Synthetic point-spread function · local preview", imageMeta: { kind: "fits" }, request: { kind: "params", service: "Design preview", text: "Deterministic synthetic PSF rendered locally; not astronomical data." } }),
];
export const SAMPLE_SCENES = [
  { id: "sample-table", label: "Archive results", title: "Sz65 · Band 6 observations", messages: tableMessages },
  { id: "sample-plot", label: "Spectrum", title: "CO(2–1) · line profile", messages: plotMessages },
  { id: "sample-image", label: "Image card", title: "Image inspection", messages: imageMessages },
];
export function sampleConversations(): Conversation[] {
  return SAMPLE_SCENES.map((s, i) => ({ id: s.id, title: s.title, createdAt: time, updatedAt: new Date(time.getTime() - i * 3600000), model: "gpt-oss-120b", messages: structuredClone(s.messages) }));
}

import { NextRequest, NextResponse } from "next/server";
import { PREVIEW_USER, SAMPLE_SCENES, SAMPLE_TABLE, sampleConversations } from "../../../fixtures";
export const dynamic = "force-dynamic";
const removed = new Set<string>();
const json = (body: unknown, status = 200) => NextResponse.json(body, { status, headers: { "Cache-Control": "no-store" } });
async function handle(request: NextRequest) {
  // Deliberately never read cookies, credentials, API keys, or upstream environment.
  const path = request.nextUrl.pathname;
  const method = request.method;
  if (method === "GET") {
    if (path === "/api/auth/me") return json({ user: PREVIEW_USER });
    if (path === "/api/models") return json({ models: ["gpt-oss-120b", "Qwen3-32B", "gemma-4-31B-it"] });
    if (path === "/api/analytics/hit") return json({});
    if (path === "/api/conversations") return json({ conversations: sampleConversations().filter(c => !removed.has(c.id)).map(c => ({ id: c.id, title: c.title, model: c.model, created_at: c.createdAt.toISOString(), updated_at: c.updatedAt.toISOString() })) });
    if (/^\/api\/conversations\/[^/]+\/messages$/.test(path)) {
      const scene = SAMPLE_SCENES.find(s => s.id === path.split("/")[3]);
      const messages = (scene?.messages || []).map(m => ({ role: m.role, type: m.type === "text" ? "text" : "text", content: m.content, metadata: m.type === "data" ? { dataTable: m.dataTable } : m.type === "image" ? { images: [{ url: m.imageUrl, caption: m.imageCaption, meta: m.imageMeta }] } : {} }));
      // Plotly JSON is restored by the preview scene control; normal history in
      // this checkout stores only a PNG fallback for old plot messages.
      return json({ messages });
    }
    if (path === "/api/block-feedback") return json({ ratings: {} });
    if (path === "/api/provider-keys") return json({ keys: [] });
    if (path === "/api/personalization/documents") return json({ documents: [] });
    if (path === "/api/mcp-servers") return json({ servers: [] });
    if (path === "/api/datalab/jobs") return json({ jobs: [] });
    if (path === "/api/datalab/mytables") return json({ tables: [] });
    if (path === "/api/usage-quota" || path === "/api/usage/summary") return json({ platform: {}, byok: {}, is_admin: false, daily: { used_tokens: 0, unlimited: true, exhausted: false } });
  }
  if (method === "POST" && ["/api/chat", "/api/chat/upload"].includes(path)) {
    // Local, deterministic SSE exercises the real ChatArea send/finish path.
    const content = "This is a local design preview. Your message reached Quasar’s existing composer and chat renderer; no model or archive was contacted. Use the Scene menu above to inspect the table, spectrum, and image components.";
    const events = [{ type: "conversation_meta", conversation_id: "sample-table", title: "Local preview conversation" }, { type: "token", content }, { type: "data", ...SAMPLE_TABLE }];
    return new Response(events.map(e => "data: " + JSON.stringify(e) + "\n\n").join("") + "data: [DONE]\n\n", { headers: { "Content-Type": "text/event-stream", "Cache-Control": "no-store" } });
  }
  if (method === "DELETE" && /^\/api\/conversations\/sample-(table|plot|image)$/.test(path)) { removed.add(path.split("/").pop()!); return json({ success: true }); }
  if (method === "POST" && path === "/api/auth/logout") return json({ success: true });
  if (method === "POST" && ["/api/feedback", "/api/block-feedback", "/api/issue-reports"].includes(path)) return json({ success: true, preview: true });
  return json({ detail: "This operation is unavailable in the local design preview. No external request was made." }, 405);
}
export { handle as GET, handle as POST, handle as PUT, handle as PATCH, handle as DELETE };


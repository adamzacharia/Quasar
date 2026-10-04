/**
 * Save an image URL as a file. The `download` attribute is ignored for
 * cross-origin URLs (the API host differs from the app's), so the image is
 * fetched as a blob first; when that fails it opens in a new tab instead.
 * `env` exists for tests (fetch, document, URL, window, setTimeout).
 */
export function downloadName(url, model, now = Date.now()) {
    const ext = (/\.(png|jpe?g|webp|gif)(?:\?|$)/i.exec(String(url || ""))?.[1] || "png").toLowerCase();
    const label = String(model || "generated").replace(/[^\w.-]+/g, "-");
    return `quasar-${label}-${now}.${ext}`;
}

export async function saveImage(url, model, env = globalThis) {
    try {
        const response = await env.fetch(url);
        if (!response.ok) throw new Error(String(response.status));
        const objectUrl = env.URL.createObjectURL(await response.blob());
        const anchor = env.document.createElement("a");
        anchor.href = objectUrl;
        anchor.download = downloadName(url, model);
        env.document.body.appendChild(anchor);
        anchor.click();
        anchor.remove();
        env.setTimeout(() => env.URL.revokeObjectURL(objectUrl), 1000);
        return "downloaded";
    } catch {
        env.window.open(url, "_blank", "noopener,noreferrer");
        return "opened";
    }
}

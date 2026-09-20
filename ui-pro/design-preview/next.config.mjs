import path from "node:path";
import { fileURLToPath } from "node:url";
const root = path.dirname(fileURLToPath(import.meta.url));
export default {
  devIndicators: false,
  distDir: "../.next/design-preview",
  turbopack: { root: path.resolve(root, "..") },
  env: { NEXT_PUBLIC_API_URL: "http://127.0.0.1:3011", NEXT_PUBLIC_ENABLE_EVAL_MODE: "0" },
  async headers() { return [{ source: "/:path*", headers: [
    { key: "Content-Security-Policy", value: "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval' blob:; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; font-src 'self' https://fonts.gstatic.com; img-src 'self' data: blob:; connect-src 'self'; worker-src 'self' blob:; frame-src 'self'; object-src 'none'; base-uri 'self'; form-action 'self'" },
    { key: "X-Robots-Tag", value: "noindex, nofollow" }
  ] }]; }
};


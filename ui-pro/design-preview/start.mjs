import { spawn } from "node:child_process";
import { copyFileSync, mkdirSync } from "node:fs";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import path from "node:path";
const root = path.dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
mkdirSync(path.join(root, "public"), { recursive: true });
for (const asset of ["quasar_logo.png", "favicon.png", "favicon.ico"]) {
  copyFileSync(path.join(root, "../public", asset), path.join(root, "public", asset));
}
const child = spawn(process.execPath, [require.resolve("next/dist/bin/next"), "dev", root, "--hostname", "127.0.0.1", "--port", "3011"], {
  cwd: root, stdio: "inherit", windowsHide: true,
  env: { ...process.env, NEXT_TELEMETRY_DISABLED: "1", NEXT_PUBLIC_API_URL: "http://127.0.0.1:3011" }
});
for (const signal of ["SIGINT", "SIGTERM"]) process.on(signal, () => child.kill(signal));
child.on("exit", code => process.exit(code ?? 0));


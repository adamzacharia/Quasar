"use client";
import { useEffect, useRef, useState } from "react";
import { usePathname, useRouter } from "next/navigation";
import { ArrowLeftRight, Check, ChevronDown, Moon, Sun } from "lucide-react";
import { useChatStore } from "../src/lib/store";
import { useThemeStore } from "../src/lib/theme-store";
import { SAMPLE_SCENES, sampleConversations } from "./fixtures";

type Design = "current" | "refined" | "compact";
export function PreviewChrome({ children }: { children: React.ReactNode }) {
  const [design, setDesign] = useState<Design>("refined");
  const [ready, setReady] = useState(false);
  const [notice, setNotice] = useState("");
  const [scene, setScene] = useState("empty");
  const [inspector, setInspector] = useState(false);
  const [measurements, setMeasurements] = useState<string[]>([]);
  const queuedScene = useRef<string | null>(null);
  const router = useRouter();
  const pathname = usePathname();
  const theme = useThemeStore(s => s.theme);
  const setTheme = useThemeStore(s => s.setTheme);
  const activeId = useChatStore(s => s.activeConversationId);
  const streaming = useChatStore(s => s.isStreaming);

  useEffect(() => {
    const saved = localStorage.getItem("quasar_preview_design");
    const selected: Design = saved === "current" || saved === "compact" ? saved : "refined";
    document.documentElement.dataset.design = selected;
    localStorage.setItem("quasar_onboarded", "true");
    localStorage.removeItem("quasar_starting_points_day");
    useChatStore.setState({ sidebarOpen: window.innerWidth >= 768 });
    // The preview holds rendering until its origin-local setup is ready.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setDesign(selected);
    setReady(true);
  }, []);

  function applyScene(id: string) {
    const store = useChatStore.getState();
    if (id === "empty") {
      localStorage.removeItem("quasar_starting_points_day");
      store.createNewConversation();
    } else {
      const conversations = sampleConversations();
      const selected = conversations.find(c => c.id === id);
      if (!selected) return;
      useChatStore.setState({ conversations, activeConversationId: id, messages: selected.messages, isStreaming: false, thinkingSteps: [], thinkingStatus: "idle" });
    }
    setScene(id);
  }
  useEffect(() => {
    if (pathname === "/" && queuedScene.current) {
      const id = queuedScene.current;
      queuedScene.current = null;
      // This synchronizes a requested preview scene after route navigation.
      // eslint-disable-next-line react-hooks/set-state-in-effect
      applyScene(id);
    }
  }, [pathname]);
  useEffect(() => {
    if (!notice) return;
    const timer = setTimeout(() => setNotice(""), 4500);
    return () => clearTimeout(timer);
  }, [notice]);
  function chooseScene(id: string) {
    if (streaming) { setNotice("Wait for the local sample response to finish before changing scenes."); return; }
    if (pathname !== "/") { queuedScene.current = id; router.push("/"); }
    else applyScene(id);
  }
  function chooseDesign(value: Design) {
    setDesign(value);
    document.documentElement.dataset.design = value;
    localStorage.setItem("quasar_preview_design", value);
  }
  function inspectColors() {
    const root = getComputedStyle(document.documentElement);
    const resolve = (token: string) => {
      const probe = document.createElement("span");
      probe.style.color = root.getPropertyValue(token).trim();
      document.body.appendChild(probe);
      const value = getComputedStyle(probe).color;
      probe.remove();
      return value;
    };
    const luminance = (color: string) => {
      const channels = (color.match(/[\\d.]+/g) || []).slice(0, 3).map(v => Number(v) / 255).map(v => v <= .04045 ? v / 12.92 : ((v + .055) / 1.055) ** 2.4);
      return channels[0] * .2126 + channels[1] * .7152 + channels[2] * .0722;
    };
    const contrast = (a: string, b: string) => { const x = luminance(a), y = luminance(b); return ((Math.max(x, y) + .05) / (Math.min(x, y) + .05)).toFixed(2); };
    setMeasurements(["--q-bg", "--q-surface", "--q-input-bg"].map(bg => `${bg}: body ${contrast(resolve("--q-text"), resolve(bg))}:1 · secondary ${contrast(resolve("--q-text-secondary"), resolve(bg))}:1`));
    setInspector(v => !v);
  }
  return <>
    <div className="preview-toolbar" role="region" aria-label="Design preview controls">
      <div className="preview-identity"><ArrowLeftRight size={16} strokeWidth={1.5}/><strong>Quasar</strong><span>Design preview</span></div>
      <div className="preview-segments" aria-label="Appearance">{(["current", "refined", "compact"] as const).map(value => <button key={value} aria-pressed={design === value} onClick={() => chooseDesign(value)}>{value === "current" ? "Current" : value === "refined" ? "Refined" : "Compact"}</button>)}</div>
      <label className="preview-scene"><span>Scene</span><select value={activeId?.startsWith("sample-") ? activeId : scene === "empty" || !activeId ? "empty" : scene} onChange={e => chooseScene(e.target.value)} disabled={streaming}><option value="empty">New chat</option>{SAMPLE_SCENES.map(s => <option key={s.id} value={s.id}>{s.label}</option>)}</select><ChevronDown size={13}/></label>
      <button className="preview-theme" onClick={() => setTheme(theme === "dark" ? "light" : "dark")} aria-label={theme === "dark" ? "Preview light theme" : "Preview dark theme"}>{theme === "dark" ? <Sun size={16}/> : <Moon size={16}/>}</button>
      <button className="preview-inspect" onClick={inspectColors} aria-expanded={inspector}>Contrast</button>
      <span className="preview-data"><Check size={12}/> Sample data · local only</span>
    </div>
    {inspector && <div className="preview-measurements" role="status"><strong>{theme === "dark" ? "Dark" : "Light"} theme · computed token colors</strong>{measurements.map(m => <span key={m}>{m}</span>)}<small>Token pairs only; not a whole-page accessibility audit.</small></div>}
    <div className="preview-stage quasar-backdrop fixed inset-0 flex overflow-hidden" onClickCapture={event => {
      const anchor = (event.target as HTMLElement).closest?.("a");
      if (!anchor?.href || anchor.href.startsWith("blob:") || anchor.href.startsWith("data:")) return;
      if (new URL(anchor.href, location.href).origin !== location.origin) {
        event.preventDefault(); event.stopPropagation(); setNotice("External links stay closed in this local design preview.");
      }
    }}>{ready ? children : <div className="preview-loading">Loading Quasar’s components…</div>}</div>
    {notice && <div className="preview-notice" role="status">{notice}</div>}
  </>;
}


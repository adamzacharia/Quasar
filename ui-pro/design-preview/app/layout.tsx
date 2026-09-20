import type { Metadata } from "next";
import { PreviewChrome } from "../preview-chrome";
import "../../src/app/globals.css";
import "../preview.css";
export const metadata: Metadata = { title: "Quasar — Design preview", robots: { index: false, follow: false } };
export default function PreviewLayout({ children }: { children: React.ReactNode }) {
  return <html lang="en" data-theme="light" data-design="refined" suppressHydrationWarning>
    <head><script dangerouslySetInnerHTML={{ __html: `try{localStorage.setItem("quasar_onboarded","true");var t=localStorage.getItem("quasar_theme");document.documentElement.dataset.theme=t==="dark"?"dark":"light";document.documentElement.dataset.design=localStorage.getItem("quasar_preview_design")||"refined"}catch(e){}` }} /></head>
    <body className="font-display antialiased" suppressHydrationWarning><PreviewChrome>{children}</PreviewChrome></body>
  </html>;
}


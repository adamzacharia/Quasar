import type { Metadata, Viewport } from "next";
import { SpeedInsights } from "@vercel/speed-insights/next";
import "./globals.css";

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  maximumScale: 1,
  userScalable: false,
};

export const metadata: Metadata = {
  title: "QUASAR — AI Radio Astronomy Research Assistant",
  description: "Professional AI-powered interface for radio astronomy research, data analysis, and literature review.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className="dark" suppressHydrationWarning>
      <body className="bg-bg-dark font-display text-slate-100 h-screen flex overflow-hidden antialiased" suppressHydrationWarning>
        {children}
        <SpeedInsights />
      </body>
    </html>
  );
}

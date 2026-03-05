import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "QUASAR — AI Radio Astronomy Research Assistant",
  description: "Professional AI-powered interface for radio astronomy research, data analysis, and literature review.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className="dark" suppressHydrationWarning>
      <body className="bg-bg-dark font-display text-slate-100 h-screen flex overflow-hidden antialiased" suppressHydrationWarning>
        {children}
      </body>
    </html>
  );
}

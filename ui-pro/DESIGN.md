# QUASAR Design System

> Auto-generated design specification for the QUASAR UI (`ui-pro/`).
> Use this as the single source of truth when building or modifying components.

---

## Stack

| Layer        | Technology                          |
|------------- |-------------------------------------|
| Framework    | Next.js 16 (App Router)             |
| UI Library   | React 19                            |
| Styling      | Tailwind CSS 4, CSS custom props    |
| State        | Zustand 5                           |
| Icons        | Lucide React + custom SVGs          |
| Typography   | Google Fonts (Inter, JetBrains Mono)|
| Markdown     | react-markdown + remark-gfm         |
| Analytics    | Vercel Speed Insights               |

---

## Typography

| Token            | Font                    | Usage                        |
|----------------- |-------------------------|------------------------------|
| `--font-display` | `"Inter", sans-serif`   | All UI text, headings, body  |
| `--font-mono`    | `"JetBrains Mono", mono`| Code blocks, pricing, stats  |

### Fluid base font size

| Breakpoint               | `font-size` |
|--------------------------|-------------|
| Default (< 768px)        | 14px        |
| `min-width: 768px`       | 16px        |
| `min-width: 1280px`      | 17px        |
| `min-width: 1600px + 1000px height` | 18px |
| `min-width: 2200px + 1200px height` | 19px |

All `rem` values scale with these breakpoints.

---

## Color System

### Brand colors (constant across themes)

| Token                  | Value       | Usage                              |
|----------------------- |-------------|------------------------------------|
| `--color-primary`      | `#f471b5`   | Primary pink, CTAs, active states  |
| `--color-primary-dark` | `#22111a`   | Text on primary buttons            |
| `--color-accent-purple`| `#a855f7`   | Secondary accent, gradient partner |
| `--color-emerald-accent`| `#10b981`  | Success, connected, grounded mode  |

### Text gradient

```css
background-image: linear-gradient(to right, #f471b5, #c084fc);
```

Used via the `.text-gradient` utility class on the QUASAR wordmark.

### Dark theme (default)

| Token                 | Value                           | Usage                          |
|---------------------- |---------------------------------|--------------------------------|
| `--q-bg`              | `#0A1628`                       | Page background                |
| `--q-sidebar`         | `#112340`                       | Sidebar background             |
| `--q-card`            | `#1d2e4a`                       | Card surfaces                  |
| `--q-surface`         | `#0f1f38`                       | Elevated surfaces, inputs      |
| `--q-text`            | `#f1f5f9`                       | Primary text (slate-100)       |
| `--q-text-secondary`  | `#94a3b8`                       | Secondary text (slate-400)     |
| `--q-text-muted`      | `#64748b`                       | Muted/disabled text (slate-500)|
| `--q-border`          | `rgba(255,255,255,0.08)`        | Default borders                |
| `--q-input-bg`        | `#0f1f38`                       | Input fields background        |
| `--q-code-bg`         | `#0a0a0f`                       | Code block background          |
| `--q-shadow-glow`     | `rgba(244,113,181,0.15)`        | Pink glow on hover             |
| `--q-scrollbar-track` | `#112340`                       | Scrollbar track                |
| `--q-scrollbar-thumb` | `#334155`                       | Scrollbar thumb                |

### Light theme

| Token                 | Value                           |
|---------------------- |---------------------------------|
| `--q-bg`              | `#f8fafc`                       |
| `--q-sidebar`         | `#f1f5f9`                       |
| `--q-card`            | `#ffffff`                       |
| `--q-surface`         | `#f0f4f8`                       |
| `--q-text`            | `#0f172a`                       |
| `--q-text-secondary`  | `#475569`                       |
| `--q-text-muted`      | `#94a3b8`                       |
| `--q-border`          | `rgba(0,0,0,0.08)`             |
| `--q-input-bg`        | `#ffffff`                       |
| `--q-code-bg`         | `#f1f5f9`                       |

Theme is toggled via `data-theme="dark|light"` on `<html>`, persisted in `localStorage("quasar_theme")`, managed by `useThemeStore` (Zustand).

---

## Glassmorphism System

The UI uses a layered frosted-glass aesthetic. Each layer has increasing blur and opacity.

| Class            | Blur    | Saturate | Purpose                        |
|----------------- |---------|----------|--------------------------------|
| `.glass-card`    | 18px    | 140%     | Interactive content cards      |
| `.glass-surface` | 18px    | 145%     | Chat input, elevated areas     |
| `.glass-panel`   | 20px    | 145%     | Chat header bar                |
| `.glass-sidebar` | 22px    | 150%     | Sidebar navigation             |
| `.glass-popover` | 24px    | 155%     | Dropdowns, menus, tooltips     |
| `.glass-control` | none    | none     | Buttons, toggles, small inputs |
| `.glass-active`  | none    | none     | Active sidebar item (pink left border) |

### Glass token map (dark)

| Token                    | Value                             |
|------------------------- |-----------------------------------|
| `--q-glass-bg`           | `rgba(255,255,255,0.03)`          |
| `--q-glass-hover`        | `rgba(255,255,255,0.08)`          |
| `--q-glass-border`       | `rgba(255,255,255,0.08)`          |
| `--q-glass-hover-border` | `rgba(244,113,181,0.4)` (pink)    |
| `--q-glass-surface`      | `rgba(13,28,50,0.88)`             |
| `--q-glass-strong`       | `rgba(13,28,50,0.94)`             |
| `--q-glass-sidebar`      | `rgba(17,35,64,0.95)`             |
| `--q-glass-popover`      | `rgba(8,18,34,0.96)`              |
| `--q-glass-control`      | `rgba(255,255,255,0.09)`          |
| `--q-glass-control-hover`| `rgba(255,255,255,0.14)`          |
| `--q-glass-highlight`    | `rgba(255,255,255,0.18)`          |
| `--q-glass-shadow`       | `0 18px 48px rgba(0,0,0,0.28), inset 0 1px 0 rgba(255,255,255,0.09)` |

### Hover behavior

- `.glass-card:hover` — stronger background, pink border, `translateY(-2px)`, pink glow shadow
- `.glass-control:hover` — lighter bg, pink border highlight

---

## Layout

### Responsive layout variables

| Variable                    | Default         | 1600px+   | 2200px+   |
|---------------------------- |-----------------|-----------|-----------|
| `--q-sidebar-width`         | `280px`         | `320px`   | `360px`   |
| `--q-empty-state-width`     | `64rem`         | `72rem`   | `80rem`   |
| `--q-suggestion-grid-width` | `48rem`         | `56rem`   | `64rem`   |
| `--q-table-max-height`      | `420px`         | `520px`   | `620px`   |
| `--q-chat-content-width`    | `calc(100% - clamp(0rem, 4vw, 6rem))`  | — | — |
| `--q-chat-input-width`      | `calc(100% - clamp(0rem, 8vw, 12rem))` | — | — |

### Page structure

```
<html data-theme="dark|light">
  <body class="bg-[var(--q-bg)] font-display text-[var(--q-text)]">
    <div class="fixed inset-0 flex overflow-hidden quasar-backdrop">
      <Sidebar />          <!-- fixed width, glass-sidebar -->
      <main>               <!-- flex-1 -->
        <ChatArea />       <!-- or page content -->
      </main>
    </div>
  </body>
</html>
```

### Background gradient

```css
.quasar-backdrop {
  background-color: var(--q-bg);
  background-image: linear-gradient(135deg,
    rgba(244,113,181,0.08),   /* pink top-left */
    transparent 34%,
    rgba(16,185,129,0.045)    /* emerald bottom-right */
  );
}
```

Light theme uses a blue tint instead of emerald: `rgba(14,165,233,0.055)`.

---

## Component Patterns

### Buttons

| Variant           | Classes                                                                     |
|------------------ |-----------------------------------------------------------------------------|
| **Primary CTA**   | `bg-primary hover:bg-primary/90 text-primary-dark font-semibold rounded-full shadow-lg shadow-primary/20` |
| **Icon button**   | `p-2.5 rounded-full` + contextual colors                                   |
| **Ghost nav**     | `text-slate-400 hover:bg-white/10 hover:text-white rounded-lg transition-colors` |
| **Active nav**    | `glass-active text-white` (pink left border)                               |
| **Danger**        | `text-red-500/70 hover:bg-red-500/10 hover:text-red-500`                   |
| **Glass button**  | `glass-control` base + hover states                                        |

### Cards

```
glass-card rounded-2xl md:rounded-3xl p-3 md:p-5 text-left group
```

Suggestion cards use per-icon colored pill backgrounds:
- Primary pink: `bg-primary/10 text-primary`
- Indigo: `bg-indigo-500/10 text-indigo-400`
- Emerald: `bg-emerald-500/10 text-emerald-400`
- Orange: `bg-orange-500/10 text-orange-400`
- Cyan: `bg-cyan-500/10 text-cyan-200` (spectral line nav)

### Inputs

```
glass-surface rounded-2xl ring-1 ring-white/10
focus-within:border-primary/50 focus-within:ring-primary/50
```

Inner `<input>`: `bg-transparent border-none text-white placeholder-slate-500 text-sm`

### Dropdowns / Popovers

```
glass-popover rounded-xl p-1.5
```

Menu items: `rounded-lg px-3 py-2 text-sm text-slate-200 hover:bg-white/10`

### Toggles

Custom switch using spans:
- Track: `h-5 w-9 rounded-full border`
- Thumb: `h-3.5 w-3.5 rounded-full transition-transform`
- Active: colored border + bg + translated thumb

### Sidebar navigation items

```
w-full flex items-center gap-3 px-3 py-2.5 rounded-xl transition-colors text-sm
```

Active: theme-specific highlight (e.g., `bg-cyan-500/10 text-cyan-200 border border-cyan-500/20`)

---

## Icon System

### Primary: Lucide React

Standard icons from `lucide-react`. Common imports:

`Plus`, `MessageSquare`, `History`, `Bookmark`, `Settings`, `HelpCircle`, `ChevronDown`,
`Bot`, `X`, `ExternalLink`, `Github`, `BookOpen`, `Search`, `Telescope`, `FileText`,
`Zap`, `Check`, `LogOut`, `User`, `Trash2`, `Cpu`, `Waves`, `Send`, `PlusCircle`,
`Square`, `ShieldCheck`, `Globe`, `Satellite`, `AudioWaveform`, `Image`

Default size: `w-4 h-4` or `w-5 h-5`.

### Custom SVG icons (`src/components/icons/QuasarIcons.tsx`)

Stroke-based 16x16 SVGs with `strokeWidth="1.3"`:

| Component        | Usage                              |
|------------------ |------------------------------------|
| `IconNeuralNet`  | Thinking/reasoning UI states       |
| `IconOpenBook`   | Journal/documentation references   |
| `IconWebGlobe`   | Web search indicators              |
| `IconChainLink`  | External links, connected resources|

### Provider logos

Inline SVGs for model providers: OpenAI, DeepSeek. Fallback: `<Bot>` from Lucide.
TACC models use `<Cpu>` with `text-cyan-300`.

---

## Spacing Conventions

| Context               | Value           |
|---------------------- |-----------------|
| Sidebar padding       | `p-4` to `p-6`  |
| Card padding          | `p-3` / `p-5`   |
| Section gap           | `space-y-3` to `space-y-6` |
| Icon-text gap         | `gap-2` to `gap-3` |
| Button padding        | `px-3 py-2` (nav), `py-3 px-4` (CTA) |
| Border radius — small | `rounded-lg` (8px) |
| Border radius — medium| `rounded-xl` (12px) |
| Border radius — large | `rounded-2xl` (16px) / `rounded-3xl` (24px) |
| Border radius — full  | `rounded-full` (CTA buttons, avatars) |

---

## Animation & Transitions

| Pattern                | Implementation                                           |
|----------------------- |----------------------------------------------------------|
| Default transition     | `transition-colors` (200ms ease)                         |
| Card hover             | `transition: all 0.3s ease` + `translateY(-2px)`         |
| Slide-in panel         | `animate-in fade-in slide-in-from-left-2 duration-200`   |
| Dropdown open          | `animate-in fade-in slide-in-from-bottom-2 duration-150` |
| Chevron rotate         | `transition-transform duration-200 rotate-180`           |
| Icon spin on hover     | `group-hover:rotate-90` (plus icon)                      |
| Status pulse           | `animate-pulse` (connection dot, stop button)            |
| Theme switch           | `transition-colors duration-300` on `<body>`             |

---

## Prose / Markdown Rendering

### Table styling

- Rounded corners: `border-radius: 0.75rem` with `border-collapse: separate`
- Header: pink-to-purple gradient background, uppercase, `letter-spacing: 0.05em`
- Rows: subtle even/odd striping, pink hover highlight
- Font size: `0.8rem` body, `0.72rem` headers

### Heading spacing

- Top margin: `1.6em` (double separation from body text)
- Bottom margin: `0.6em`
- First heading in message: no top margin

### Paragraph/list spacing

- Paragraphs: `0.5em` top/bottom
- Lists: `0.5em` top/bottom
- Horizontal rules: `1.5em` top/bottom

---

## Scrollbar Styling

| Context          | Width | Track                 | Thumb           | Hover          |
|----------------- |-------|-----------------------|-----------------|----------------|
| Default          | 8px   | `var(--q-scrollbar-track)` | `var(--q-scrollbar-thumb)` | `#f471b5` |
| `.custom-scrollbar` | 4px | transparent           | `var(--q-scrollbar-thumb)` | `#f471b5` |
| `.hide-scrollbar`   | hidden | — | — | — |

---

## Accessibility Notes

- All interactive elements use `transition-colors` for smooth state changes
- Focus-visible ring on interactive controls: `focus-visible:ring-1 focus-visible:ring-emerald-400/50`
- ARIA roles on menus: `role="menu"`, `role="menuitem"`, `role="menuitemcheckbox"`, `aria-checked`, `aria-expanded`, `aria-haspopup`
- Tooltips use `role="tooltip"` with expand-on-hover
- Disabled state: `disabled:opacity-40 disabled:cursor-not-allowed`
- `referrerPolicy="no-referrer"` on external user avatars
- `suppressHydrationWarning` on theme-switching elements

---

## File Structure

```
ui-pro/src/
  app/
    layout.tsx              # Root layout, theme init, backdrop
    page.tsx                # Main chat page
    globals.css             # Theme tokens, glass classes, prose
    help/page.tsx           # Help documentation
    terms/page.tsx          # Terms of service
    spectral-lines/page.tsx # Spectral line explorer
    workbench/[sessionId]/page.tsx
  components/
    Sidebar.tsx             # Navigation, model picker, panels
    ChatArea.tsx            # Main chat container
    ChatInput.tsx           # Message composer with attachments
    ChatMessage.tsx         # Message bubbles, markdown
    EmptyState.tsx          # Landing state with suggestions
    AuthModal.tsx           # Login/signup modal
    SettingsModal.tsx       # Settings dialog
    ErrorBoundary.tsx       # Error fallback
    ThemeInitializer.tsx    # Client-side theme hydration
    OnboardingOverlay.tsx   # First-run experience
    PaperCard.tsx           # Research paper display
    DataTableCard.tsx       # Data table visualization
    WebSourcesCard.tsx      # Web search results
    DownloadProgress.tsx    # File download indicator
    SpectralLineTable.tsx   # Spectral line results
    PlanReviewWidget.tsx    # Research plan review
    TaskExecutionWidget.tsx # Task execution progress
    ThoughtProcessWidget.tsx# Thinking/reasoning display
    ObservationPaperGraph.tsx # Research graph visualization
    icons/QuasarIcons.tsx   # Custom SVG icons
  lib/
    store.ts                # Main Zustand store
    auth-store.ts           # Auth state
    theme-store.ts          # Theme toggle
    api.ts                  # API client
    types.ts                # TypeScript types
    models.ts               # Model utilities
    content-safety.js       # Content filtering
    evidence-quality.js     # Evidence scoring
    chat-message-updaters.js# Message state helpers
    research-graph.js       # Graph data structures
    feedback-report.js      # Feedback utilities
```

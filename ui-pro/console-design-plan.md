# Console design palette — awaiting approval

No application source or CSS has been changed for this task.

## Palette

Use Radix Colors' published sRGB Gray and Gray Dark scales verbatim, with Blue / Blue Dark as the sole interaction accent. Semantic roles use Green, Amber, Red, and Blue and their official dark counterparts. No scale value will be hand-tuned.

Sources checked 2026-09-08:
- [Radix scale usage](https://www.radix-ui.com/colors/docs/palette-composition/understanding-the-scale)
- [Official light scales](https://github.com/radix-ui/colors/blob/main/src/light.ts)
- [Official dark scales](https://github.com/radix-ui/colors/blob/main/src/dark.ts)

A step below always means the matching light/dark scale unless explicitly marked fixed. All existing token names remain defined. The final accepted map will be recorded in DESIGN.md.

## Existing color and effect token mapping

| Existing token(s) | Proposed mapping |
|---|---|
| --q-bg, --q-content-scrim | Gray 1, opaque |
| --q-sidebar, --q-card, --q-surface, --q-code-bg, --q-scrollbar-track | Gray 2 |
| --q-glass-bg, --q-glass-surface, --q-glass-sidebar, --q-glass-drawer | Gray 2, opaque |
| --q-glass-strong, --q-glass-popover | Gray 3, opaque raised surfaces |
| --q-input-bg, --q-glass-control | Gray 3 |
| --q-glass-hover, --q-glass-control-hover | Gray 4 |
| --q-glass-highlight | Transparent; decorative highlight removed |
| --q-text, --q-observation-graph-node-text | Gray 12 |
| --q-text-secondary, --q-text-muted, --q-text-faint, --q-mono-accent, --q-observation-graph-node-detail | Gray 11; contextual Gray 12 where Gray 11 misses 4.5:1 |
| --q-border, --q-glass-border, --q-drawer-border, --q-observation-graph-edge | Gray 10; contextual Gray 11 where Gray 10 misses 3:1 |
| --q-glass-hover-border | Gray 11 |
| --q-scrollbar-thumb | Gray 10; Gray 11 hover |
| --q-observation-graph-node-bg | Gray 2, opaque; follows the current theme |
| --q-shadow-glow | Transparent, retained as an inert compatibility token |
| --q-glass-shadow, --q-drawer-shadow | Alias to the new md shadow; only menus, popovers, modals, and drawers consume it |
| --q-drawer-scrim | Fixed Radix Gray Dark 1 with 48% opacity in dark mode, 24% in light mode |
| --color-primary | Blue 9, primary action fill only |
| --color-primary-dark | Fixed Radix Gray Dark 1 in both themes, for text on Blue 9 |
| --color-accent-purple | Gray 11 compatibility alias; no purple rendering |
| --color-emerald-accent | Green 11; contextual Green 12 where needed, status text only |

Gray 5 remains available for pressed/selected interactive surfaces, using the stronger contextual border/text aliases where needed.

## Added role aliases

| Role | Proposed mapping |
|---|---|
| Links and active navigation | Blue 11; contextual Blue 12 where Blue 11 misses 4.5:1 |
| Focus | Blue 11, 2px outline, 2px offset |
| Success / warning / danger / info | Green / Amber / Red / Blue: step 9 dots and step 11 text; step 12 text fallback where required |
| Icon color | Inherit the accessible label/text color |
| sm shadow | Radix Gray Dark 1 at 8% opacity in dark / 4% in light; 0 1px 2px |
| md shadow | Radix Gray Dark 1 at 16% opacity in dark / 8% in light; 0 4px 12px |
| Strong border / secondary / accent / semantic text variants | Same role in both themes; use the next documented step only as specified above |

Step 12 text fallbacks will be explicit CSS role variants chosen for the known surface, with the same DOM/class structure in both themes. No runtime color or behavior logic is needed. Cards have borders and no shadow. Accent is limited to primary actions, focus, active navigation, and links. Prose token wiring remains, with links using an accessible accent-text alias and quote borders using a neutral alias.

## Existing layout token mapping

| Existing token | Proposed value |
|---|---|
| --q-sidebar-width | 260px |
| --q-sidebar-rail-width | 56px |
| --q-drawer-width | min(260px, 85vw) |
| --q-chat-content-width | Preserve current responsive width; constrain prose itself to 72ch |
| --q-chat-input-width | Preserve current responsive width |
| --q-empty-state-width | 72ch, capped by available width |
| --q-suggestion-grid-width | 72ch, capped by available width; now a plain list |
| --q-table-max-height | Preserve current responsive heights |

Typography: IBM Plex Sans and IBM Plex Mono through one Google Fonts import, system fallbacks, 14px base / 15px at 1280px+, heading weight 600, tabular numerals for numeric data. Controls 6px radius, cards 8px, modals/drawers 10px. List rows 32px, 8px spacing grid, card padding 12px. Lucide icons 16px / 1.5 stroke. Motion 120–160ms ease-out for interaction/open/close and reduced-motion support.

## Contrast exceptions requiring approval

The supplied fixed step rules conflict with its contrast targets. These are calculated sRGB ratios from the published colors, not browser-computed verification of the finished application.

1. Gray 8 against light Gray 1 is 1.87:1; use Gray 10 for normal borders and Gray 11 when needed. Gray 10 against light Gray 3 is 3.33:1, but against Gray 5 it is 2.87:1.
2. Blue 8 against light Gray 1 is 2.27:1. Blue 10 still fails on Gray 4 (2.96:1), so use Blue 11 for focus: at least 3.61:1 across Gray 1–5 in light mode and 6.19:1 in dark.
3. Step 11 text sometimes needs step 12: light Blue 11 on Gray 3 is 4.18:1; Green 11 on Gray 2 is 4.480:1; Amber 11 on Gray 1 is 4.494:1; Gray 11 on Gray 5 is 4.483:1. Values below a threshold are failures even when rounding would display the threshold.

Proposed primary text / secondary text ratios on background, surface, raised:
- Light: 15.88 / 5.77; 15.48 / 5.62; 14.30 / 5.19.
- Dark: 16.28 / 9.11; 15.15 / 8.48; 13.71 / 7.67.
- Fixed Gray Dark 1 text on Blue 9: 5.78:1 in either theme.

An independent read-only Codex review confirmed these conflicts and the need to preserve graph/scrim/shadow token families. Its concern that Gray 10 borders and Blue 10 focus do not pass on every interactive surface is incorporated above; no finding is treated as waived.

## Work after approval

Implement presentation changes only within ui-pro/, preserving existing behavior, data flow, routes, stores, props, and pre-existing edits. Run lint and TypeScript, check prohibited source patterns and token preservation, then check the requested pages/cards in both themes and at 375px. Measure browser-computed colors including interactive states and finish an independent diff review. Application implementation and live acceptance checks are pending.


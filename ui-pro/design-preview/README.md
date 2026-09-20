# Quasar browser design preview

Run `node design-preview/start.mjs` from ui-pro, then open http://127.0.0.1:3011.

This is an isolated Next app. Its pages import the existing Quasar React pages and components directly; no production source is copied or modified. The preview-only stylesheet changes presentation under a data-design attribute. Current removes those overrides. Refined and Compact are exploratory directions, not approved final designs.

The fixed preview API base is http://127.0.0.1:3011. All /api requests terminate in local fixtures; there is no proxy or upstream fetch. The preview ignores cookies and never sets them. Storage is isolated by origin. CSP confines network API connections to this origin. Only the existing and preview Google Fonts styles/fonts are allowed remotely. Sample data and locally simulated sends are labelled by the preview toolbar. Unsupported server actions return an explicit preview-only response.

QA inventory: current/refined/compact full cycle; light/dark full cycle; initial and filled chat; suggestion prefill; local send completion; sidebar expand/collapse; settings tabs and close; composer options open/toggle/close; gallery navigation/filter/prompt handoff; table sorting; real Plotly card zoom/reset; desktop and 375px viewport fit; no network connection to :8000 or external APIs. Exploratory cases: switch conversations during a local response, and narrow viewport with the sidebar open.

Colors follow the published Radix Gray / Gray Dark and Blue / Blue Dark scales: https://github.com/radix-ui/colors/blob/main/src/light.ts and https://github.com/radix-ui/colors/blob/main/src/dark.ts . The prior palette plan and contrast exceptions remain proposals. This preview is not final whole-app accessibility acceptance.


# PAI26 Conference Context & LaTeX Viewing Guide

## PAI26 Is the First Edition

> [!IMPORTANT]
> **There is no PAI25 or PAI24.** PAI26 is the **inaugural** Physics and AI conference. The "26" refers to the year (2026), not the edition number.

PAI26 is a **new conference** co-organized by three groups:
1. Stanford's **Center for Decoding the Universe** (CDU) — existed since ~2024
2. **APS Group on Data Science**
3. **NeurIPS ML and Physical Sciences (ML4PS) Workshop** team

### Related Prior Events (not PAI editions)

| Event | Date | Format | Details |
|-------|------|--------|---------|
| **CDU Annual Conference** (1st edition) | June 2025 | Talks + posters | "Data-driven Discovery in the Rubin Era" |
| **CDU Quarterly Forum — Fall 2025** | Fall 2025 | Invited talks | AI+astrophysics sessions |
| **CDU Quarterly Forum — Fall 2024** | Fall 2024 | Inaugural forum | ML applied to astrophysical data |
| **NeurIPS ML4PS Workshop** | 2017–2024 | 4-page papers | ~150 papers/year, same organizer team |

The CDU forums had recorded talks and posters but **no peer-reviewed published papers**. The ML4PS workshop is the closest venue with published proceedings.

### ML4PS 2024 Best Papers (for reference)

| Award | Title | Link |
|-------|-------|------|
| 🏅 Best AI for Physics | Robust Emulator for Compressible Navier-Stokes (Gregory et al.) | [PDF](https://ml4physicalsciences.github.io/2024/files/NeurIPS_ML4PS_2024_176.pdf) |
| 🏅 Best Physics for AI | Higher-order Cumulants in Diffusion Models (Aarts et al.) | [PDF](https://ml4physicalsciences.github.io/2024/files/NeurIPS_ML4PS_2024_71.pdf) |

Full ML4PS 2024 proceedings (~150 papers): [ml4physicalsciences.github.io/2024](https://ml4physicalsciences.github.io/2024/)

---

## How to View LaTeX as PDF

Antigravity **does not have a built-in LaTeX compiler**. Three options:

### Option A: LaTeX Workshop Extension (Best for in-editor preview)
1. **Ctrl+Shift+X** → search **"LaTeX Workshop"** → Install
2. Install [MiKTeX](https://miktex.org/download) (`winget install MiKTeX.MiKTeX`)
3. Open [paper/main.tex](file:///c:/Users/adama/Desktop/Quasar-main/paper/main.tex) → auto-compiles → PDF preview side-by-side

### Option B: Overleaf (No install needed)
1. [overleaf.com](https://overleaf.com) → New Project → Upload → upload `paper/` folder
2. Instant live PDF preview

### Option C: Command line
```powershell
winget install MiKTeX.MiKTeX   # one-time install
cd c:\Users\adama\Desktop\Quasar-main\paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
# Open main.pdf in your default viewer
```

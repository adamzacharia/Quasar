"""
QUASAR Benchmark Runner
=======================
Automatically tests all benchmark questions against the Quasar API,
scores responses using an LLM judge, and generates charts/reports.

Usage:
    python run_benchmark.py                             # Run against local backend
    python run_benchmark.py --api-url https://quasar-7812.onrender.com  # Remote
    python run_benchmark.py --model gpt-4.1             # Specify model
    python run_benchmark.py --judge-model gpt-4o        # Specify judge model
    python run_benchmark.py --output-dir ./results       # Custom output directory
"""

import os
import sys
import json
import time
import re
import argparse
import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

# ---------------------------------------------------------------------------
# Benchmark question definitions
# ---------------------------------------------------------------------------

QUESTIONS = [
    # ── General Knowledge — Easy ──
    {
        "id": "GK-E-01",
        "category": "General Knowledge",
        "difficulty": "Easy",
        "title": "ALMA Configurations and Angular Resolutions",
        "question": "What are the different ALMA configurations and their corresponding angular resolutions?",
        "criteria": [
            "Lists the main 12m array configurations (C-1 through C-10, or compact to extended)",
            "Mentions that compact configurations give lower angular resolution and extended configurations give higher angular resolution",
            "Mentions that angular resolution depends on observing frequency/band",
            "Optionally mentions ACA (7m array) and Total Power configurations",
            "Provides approximate resolution ranges for at least a few configurations",
            "Information is factually correct",
        ],
    },
    {
        "id": "GK-E-02",
        "category": "General Knowledge",
        "difficulty": "Easy",
        "title": "CASA Version Used for Data Processing",
        "question": "How do I find out the information that indicates which CASA version was used to process the data from a given ALMA project?",
        "criteria": [
            "Mentions checking the README file or processing logs in the delivered data package",
            "References the casa*.log files or pipeline weblog",
            "Mentions that the ALMA Science Archive metadata or QA2 reports may contain this information",
            "Mentions the pipeline_manifest.xml or equivalent metadata file",
            "Factually accurate guidance",
        ],
    },
    # ── General Knowledge — Medium ──
    {
        "id": "GK-M-01",
        "category": "General Knowledge",
        "difficulty": "Medium",
        "title": "Types of Downloadable ALMA Data",
        "question": "What types of data can I download for observations made with ALMA?",
        "criteria": [
            "Distinguishes between raw data (ASDM) and calibrated data",
            "Mentions pipeline-calibrated Measurement Sets (MS)",
            "Mentions science-ready data products (images, cubes, continuum maps)",
            "References auxiliary data (pipeline weblog, scripts, calibration tables, README)",
            "Optionally mentions quality assurance (QA2) reports",
            "Mentions that availability depends on the project and cycle",
            "Factually accurate",
        ],
    },
    {
        "id": "GK-M-02",
        "category": "General Knowledge",
        "difficulty": "Medium",
        "title": "ALMA Observing Bands",
        "question": "How many observing bands are available with ALMA? How many bands have public data available in the ALMA Science Archive as of March 25, 2026?",
        "criteria": [
            "States the total number of ALMA receiver bands (10 bands: Band 1-10)",
            "Correctly identifies which bands are commissioned and offered for observations",
            "Attempts a live query or provides knowledge about which bands have public data",
            "Distinguishes between bands that exist and bands with public data in the archive",
            "Provides band frequency ranges for at least the most common bands",
            "Factually accurate",
        ],
    },
    # ── General Knowledge — Hard ──
    {
        "id": "GK-H-01",
        "category": "General Knowledge",
        "difficulty": "Hard",
        "title": "Bandwidth Switching for Calibration",
        "question": "Which projects in the ALMA Science Archive likely needed Bandwidth Switching for calibration?",
        "criteria": [
            "Correctly explains what bandwidth switching is",
            "Identifies conditions that trigger bandwidth switching (high spectral resolution, narrow channel widths)",
            "Attempts to query the archive for projects matching these criteria",
            "Provides a strategy or query for identifying such projects",
            "Demonstrates understanding that this information may not be directly tagged and requires inference",
        ],
    },
    # ── Data-Specific — Easy ──
    {
        "id": "DS-E-01",
        "category": "Data-Specific",
        "difficulty": "Easy",
        "title": "Cycle 10 Solar Observations",
        "question": "How many projects in Cycle 10 observed the sun?",
        "criteria": [
            "Constructs a valid query filtering by Cycle 10",
            "Filters for solar observations (source name Sun or solar science category)",
            "Returns a specific number of projects",
            "Mentions the method used to arrive at the count",
            "Factually verifiable against the archive",
        ],
    },
    # ── Data-Specific — Medium ──
    {
        "id": "DS-M-01",
        "category": "Data-Specific",
        "difficulty": "Medium",
        "title": "Cycle 9 Multi-Array Projects",
        "question": "How many projects in Cycle 9 used the 12m, 7m, and total power array for their observations?",
        "criteria": [
            "Correctly identifies all three array types: 12m, 7m (ACA), and Total Power (TP)",
            "Constructs a query that finds projects using all three arrays (not just any one)",
            "Filters correctly for Cycle 9",
            "Returns a specific count",
            "Explains the methodology used",
        ],
    },
    # ── Data-Specific — Hard ──
    {
        "id": "DS-H-01",
        "category": "Data-Specific",
        "difficulty": "Hard",
        "title": "HH212 Band 7 Deep Continuum Data",
        "question": "I'm interested in the source HH212. Give me a summary of the Band 7 data from any project that can be used for creating a deep, high-resolution (better than 1 arcsec) image of the continuum.",
        "criteria": [
            "Searches for source HH212 in Band 7 (~275-373 GHz)",
            "Filters for observations with angular resolution better than 1 arcsec",
            "Identifies projects suitable for continuum imaging (bandwidth, sensitivity, integration time)",
            "Provides a structured summary (project codes, PI, resolution, bandwidth, integration time)",
            "Mentions possibility of combining data from multiple projects",
            "Discusses calibration or compatibility considerations",
        ],
    },
    # ── Analysis & Methods — Easy ──
    {
        "id": "AM-E-01",
        "category": "Analysis & Methods",
        "difficulty": "Easy",
        "title": "Python Query of ALMA Science Archive",
        "question": "Is there a way to make a query of the ALMA Science Archive to retrieve ALMA data for a specific object through Python?",
        "criteria": [
            "Confirms that Python-based query is possible",
            "Mentions at least one method: Astroquery (astroquery.alma)",
            "Optionally mentions ALMiner, pyvo (TAP), or direct HTTP/API access",
            "Provides a working code snippet demonstrating the query",
            "Code example is syntactically correct and runnable",
            "Explains what the code does",
        ],
    },
    # ── Analysis & Methods — Medium ──
    {
        "id": "AM-M-01",
        "category": "Analysis & Methods",
        "difficulty": "Medium",
        "title": "Astroquery for M83 Observations",
        "question": "How do I use Astroquery to find ALMA observations of the source M83?",
        "criteria": [
            "Provides a working astroquery.alma code example",
            "Uses Alma.query_object('M83') or equivalent",
            "Explains the returned table structure (proposal_id, target_name, band_list, etc.)",
            "Code is syntactically correct and uses current API",
            "Mentions setup requirements (installation, imports)",
        ],
    },
    {
        "id": "AM-M-02",
        "category": "Analysis & Methods",
        "difficulty": "Medium",
        "title": "TAP vs ALMiner Source Query",
        "question": "Show me two ways to query for a source in the ALMA Science Archive, with one using TAP and the other using ALMiner.",
        "criteria": [
            "Provides a working TAP/ADQL query example",
            "Provides a working ALMiner query example",
            "Both examples query for the same source/concept",
            "Explains the differences or trade-offs between the two approaches",
            "Code is syntactically correct for both methods",
        ],
    },
    {
        "id": "AM-M-03",
        "category": "Analysis & Methods",
        "difficulty": "Medium",
        "title": "M83 Band 6 Observations",
        "question": "Find ALMA observations of the source M83 using Band 6.",
        "criteria": [
            "Constructs a query with both source (M83) and band (Band 6) filters",
            "Returns actual results from the archive",
            "Presents results in a readable format (table, summary)",
            "Mentions Band 6 frequency range (~211-275 GHz) for context",
            "Code or method used is correct",
        ],
    },
    {
        "id": "AM-M-04",
        "category": "Analysis & Methods",
        "difficulty": "Medium",
        "title": "Generate ALMA Archive URL for M83 Band 6",
        "question": "Generate the URL for the ALMA Science Archive for the query of M83 using Band 6.",
        "criteria": [
            "Generates a valid, clickable URL pointing to the ALMA Science Archive",
            "URL includes source name parameter (M83)",
            "URL includes band filter (Band 6)",
            "URL is for the correct archive endpoint",
            "URL, when opened, returns relevant results",
        ],
    },
    # ── Analysis & Methods — Hard ──
    {
        "id": "AM-H-01",
        "category": "Analysis & Methods",
        "difficulty": "Hard",
        "title": "ALMA and JWST Protostars in Perseus",
        "question": "Show me locations of protostars in Perseus that have been observed with ALMA and JWST.",
        "criteria": [
            "Queries ALMA archive for observations in the Perseus star-forming region",
            "Queries JWST / MAST archive for observations in the same region",
            "Cross-matches sources observed by both observatories",
            "Identifies which sources are protostars",
            "Generates a visual map showing source locations (RA/Dec plot)",
            "Labels or distinguishes ALMA-only, JWST-only, and both-observed sources",
            "Output is visually clear and informative",
        ],
    },
    {
        "id": "AM-H-02",
        "category": "Analysis & Methods",
        "difficulty": "Hard",
        "title": "ALMA and JWST Hubble Ultra Deep Field Overlay",
        "question": "I want to see two images of the Hubble Ultra Deep Field, one with ALMA and one with JWST. Ideally, overlap the two images in a way that I can see ALMA in contours and JWST in colorscale.",
        "criteria": [
            "Correctly identifies the HUDF coordinates",
            "Retrieves or references ALMA data/image of the HUDF",
            "Retrieves or references JWST data/image of the HUDF",
            "Generates or provides code to generate an overlay image",
            "Output image is scientifically meaningful and visually appealing",
            "Handles coordinate alignment (WCS reprojection if needed)",
            "Labels axes, provides colorbar, and adds legend/annotations",
        ],
    },
    # ── Scientific Questions — Easy ──
    {
        "id": "SQ-E-01",
        "category": "Scientific Questions",
        "difficulty": "Easy",
        "title": "Protostellar Outflows Publications",
        "question": "I'm interested in studying protostellar outflows. Please make a table of the 10 most recent publications on this topic that used data from the ALMA Science Archive, and summarize them. Use the ALMA Science Archive to get this information.",
        "criteria": [
            "Retrieves publications linked to ALMA data on the topic of protostellar outflows",
            "Presents at least 10 publications (or all available if fewer)",
            "Each entry includes: Title, Authors, Year, Journal, ALMA Project Code (if available)",
            "Provides a brief summary of each publication's key findings",
            "Publications are sorted by recency (most recent first)",
            "Uses an appropriate data source (ALMA telbib, ADS, or archive bibliography)",
            "Output is well-formatted as a table",
        ],
    },
    # ── Scientific Questions — Medium ──
    {
        "id": "SQ-M-01",
        "category": "Scientific Questions",
        "difficulty": "Medium",
        "title": "Protostellar Disks with CO Isotopologues in Band 6",
        "question": "List the protostellar disks in the ALMA Science Archive for which 12CO, 13CO, and C18O in Band 6 have been observed in the same project.",
        "criteria": [
            "Correctly identifies the rest frequencies of the three CO isotopologues in Band 6",
            "Constructs a query that filters for projects covering all three lines",
            "Filters for Band 6 observations",
            "Identifies the specific projects and their target names",
            "Recognizes that the targets should be protostellar disks",
            "Returns a structured list of projects and targets",
        ],
    },
    # ── Scientific Questions — Hard ──
    {
        "id": "SQ-H-01",
        "category": "Scientific Questions",
        "difficulty": "Hard",
        "title": "Galaxies at z=1-2 with CO Observations",
        "question": "Show me all galaxies at a redshift between z=1 and z=2 that were observed with ALMA and included the Carbon Monoxide rest frequency in their spectral setup.",
        "criteria": [
            "Correctly computes the observed frequency ranges for CO transitions at z=1 to z=2",
            "Constructs queries covering the appropriate observed frequency ranges",
            "Filters for extragalactic / galaxy targets",
            "Handles the complexity of multiple possible CO transitions being redshifted into ALMA bands",
            "Returns a meaningful list of projects / sources",
            "Explains the methodology and which CO transitions are accessible at these redshifts",
        ],
    },
]

DIFFICULTY_WEIGHTS = {"Easy": 1.0, "Medium": 1.5, "Hard": 2.0}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class QuestionResult:
    id: str
    category: str
    difficulty: str
    title: str
    question: str
    response: str = ""
    response_time_s: float = 0.0
    error: Optional[str] = None
    # Scores (0-5)
    correctness: int = 0
    completeness: int = 0
    code_quality: Optional[int] = None  # None if N/A
    presentation: int = 0
    tool_usage: int = 0
    judge_reasoning: str = ""
    criteria_met: list = field(default_factory=list)

    @property
    def total_score(self) -> int:
        s = self.correctness + self.completeness + self.presentation + self.tool_usage
        if self.code_quality is not None:
            s += self.code_quality
        return s

    @property
    def max_score(self) -> int:
        return 25 if self.code_quality is not None else 20

    @property
    def percentage(self) -> float:
        return (self.total_score / self.max_score * 100) if self.max_score else 0

    @property
    def grade(self) -> str:
        p = self.percentage
        if p >= 90: return "A"
        if p >= 75: return "B"
        if p >= 60: return "C"
        if p >= 40: return "D"
        return "F"


# ---------------------------------------------------------------------------
# API interaction
# ---------------------------------------------------------------------------

def query_quasar_api(question: str, api_url: str, model: str, timeout: int = 300) -> tuple[str, float]:
    """Send a question to the Quasar SSE endpoint, collect the full response."""
    import httpx

    url = f"{api_url.rstrip('/')}/api/chat"
    payload = {"message": question, "model": model}

    t0 = time.time()
    full_text = ""

    with httpx.Client(timeout=timeout) as client:
        with client.stream("POST", url, json=payload, headers={"Content-Type": "application/json"}) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                try:
                    parsed = json.loads(data_str)
                    if parsed.get("type") == "token":
                        full_text += parsed.get("content", "")
                except json.JSONDecodeError:
                    full_text += data_str

    elapsed = time.time() - t0
    return full_text, elapsed


# ---------------------------------------------------------------------------
# LLM Judge
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator for an AI-powered astronomical research assistant called QUASAR.
You will be given a question, the evaluation criteria, and the system's response.
Score the response on each dimension (0-5 scale) and provide reasoning.

Return your evaluation as a JSON object with these fields:
{
    "correctness": <0-5>,
    "completeness": <0-5>,
    "code_quality": <0-5 or null if no code expected>,
    "presentation": <0-5>,
    "tool_usage": <0-5>,
    "criteria_met": [<list of criteria indices (0-based) that were met>],
    "reasoning": "<brief explanation of scores>"
}

Scoring guide:
- 0: Completely absent/wrong
- 1: Major issues
- 2: Significant gaps
- 3: Mostly adequate, minor issues
- 4: Good with only trivial issues
- 5: Excellent, comprehensive

For code_quality, set to null if the question doesn't require code.
For tool_usage, evaluate whether the system used appropriate tools (archive queries, RAG, etc.) based on evidence in the response.
"""


def judge_response(question_data: dict, response: str, judge_model: str) -> dict:
    """Use an LLM to judge the response quality."""
    import openai

    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    criteria_text = "\n".join(f"  {i}. {c}" for i, c in enumerate(question_data["criteria"]))

    user_prompt = f"""## Question
{question_data['question']}

## Evaluation Criteria
{criteria_text}

## System Response
{response if response else "[ERROR: No response received]"}

Please evaluate this response and return your JSON scoring."""

    resp = client.chat.completions.create(
        model=judge_model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
    )

    try:
        return json.loads(resp.choices[0].message.content)
    except (json.JSONDecodeError, IndexError):
        return {
            "correctness": 0, "completeness": 0, "code_quality": None,
            "presentation": 0, "tool_usage": 0, "criteria_met": [],
            "reasoning": "Judge failed to return valid JSON",
        }


# ---------------------------------------------------------------------------
# Chart generation
# ---------------------------------------------------------------------------

def generate_charts(results: list[QuestionResult], output_dir: Path):
    """Generate benchmark result visualizations."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # ── Color palette ──
    COLORS = {
        "Easy": "#4ade80",
        "Medium": "#facc15",
        "Hard": "#f87171",
        "A": "#22c55e", "B": "#84cc16", "C": "#eab308", "D": "#f97316", "F": "#ef4444",
    }
    CAT_COLORS = {
        "General Knowledge": "#818cf8",
        "Data-Specific": "#38bdf8",
        "Analysis & Methods": "#a78bfa",
        "Scientific Questions": "#fb923c",
    }

    plt.rcParams.update({
        "figure.facecolor": "#0f172a",
        "axes.facecolor": "#1e293b",
        "text.color": "#e2e8f0",
        "axes.labelcolor": "#e2e8f0",
        "xtick.color": "#94a3b8",
        "ytick.color": "#94a3b8",
        "axes.edgecolor": "#334155",
        "font.family": "sans-serif",
        "font.size": 11,
    })

    # ── 1. Overall scores by question ──
    fig, ax = plt.subplots(figsize=(14, 6))
    ids = [r.id for r in results]
    pcts = [r.percentage for r in results]
    colors = [COLORS.get(r.difficulty, "#94a3b8") for r in results]
    bars = ax.bar(range(len(ids)), pcts, color=colors, edgecolor="#475569", linewidth=0.5)
    ax.set_xticks(range(len(ids)))
    ax.set_xticklabels(ids, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Score (%)")
    ax.set_title("QUASAR Benchmark — Score by Question", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylim(0, 105)
    ax.axhline(y=90, color="#22c55e", linestyle="--", alpha=0.4, label="A (≥90%)")
    ax.axhline(y=75, color="#84cc16", linestyle="--", alpha=0.4, label="B (≥75%)")
    ax.axhline(y=60, color="#eab308", linestyle="--", alpha=0.4, label="C (≥60%)")
    # Legend for difficulty
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS["Easy"], label="Easy"),
        Patch(facecolor=COLORS["Medium"], label="Medium"),
        Patch(facecolor=COLORS["Hard"], label="Hard"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", framealpha=0.3)
    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{pct:.0f}%", ha="center", va="bottom", fontsize=7, color="#cbd5e1")
    plt.tight_layout()
    fig.savefig(output_dir / "scores_by_question.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── 2. Radar chart: average scores by dimension ──
    dimensions = ["Correctness", "Completeness", "Code Quality", "Presentation", "Tool Usage"]
    code_results = [r for r in results if r.code_quality is not None]
    avg_scores = [
        np.mean([r.correctness for r in results]),
        np.mean([r.completeness for r in results]),
        np.mean([r.code_quality for r in code_results]) if code_results else 0,
        np.mean([r.presentation for r in results]),
        np.mean([r.tool_usage for r in results]),
    ]

    angles = np.linspace(0, 2 * np.pi, len(dimensions), endpoint=False).tolist()
    avg_scores_loop = avg_scores + [avg_scores[0]]
    angles += [angles[0]]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.fill(angles, avg_scores_loop, alpha=0.25, color="#818cf8")
    ax.plot(angles, avg_scores_loop, "o-", color="#818cf8", linewidth=2, markersize=6)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dimensions, fontsize=10)
    ax.set_ylim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(["1", "2", "3", "4", "5"], fontsize=8)
    ax.set_title("Average Scores by Dimension", fontsize=14, fontweight="bold", pad=20)
    ax.grid(color="#475569", alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / "radar_dimensions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── 3. Scores by category ──
    categories = list(dict.fromkeys(r.category for r in results))
    fig, ax = plt.subplots(figsize=(10, 5))
    cat_avgs = []
    for cat in categories:
        cat_results = [r for r in results if r.category == cat]
        cat_avgs.append(np.mean([r.percentage for r in cat_results]))
    bars = ax.barh(categories, cat_avgs,
                   color=[CAT_COLORS.get(c, "#64748b") for c in categories],
                   edgecolor="#475569", height=0.6)
    ax.set_xlim(0, 105)
    ax.set_xlabel("Average Score (%)")
    ax.set_title("Average Score by Category", fontsize=14, fontweight="bold", pad=15)
    for bar, val in zip(bars, cat_avgs):
        ax.text(val + 1.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=10, color="#cbd5e1")
    plt.tight_layout()
    fig.savefig(output_dir / "scores_by_category.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── 4. Scores by difficulty ──
    difficulties = ["Easy", "Medium", "Hard"]
    fig, ax = plt.subplots(figsize=(8, 5))
    diff_avgs = []
    for d in difficulties:
        d_results = [r for r in results if r.difficulty == d]
        diff_avgs.append(np.mean([r.percentage for r in d_results]) if d_results else 0)
    bars = ax.bar(difficulties, diff_avgs,
                  color=[COLORS[d] for d in difficulties],
                  edgecolor="#475569", width=0.5)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Average Score (%)")
    ax.set_title("Average Score by Difficulty", fontsize=14, fontweight="bold", pad=15)
    for bar, val in zip(bars, diff_avgs):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 2,
                f"{val:.1f}%", ha="center", fontsize=11, color="#cbd5e1")
    plt.tight_layout()
    fig.savefig(output_dir / "scores_by_difficulty.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── 5. Response time chart ──
    fig, ax = plt.subplots(figsize=(14, 5))
    times = [r.response_time_s for r in results]
    colors_time = [COLORS.get(r.difficulty, "#94a3b8") for r in results]
    ax.bar(range(len(ids)), times, color=colors_time, edgecolor="#475569", linewidth=0.5, alpha=0.85)
    ax.set_xticks(range(len(ids)))
    ax.set_xticklabels(ids, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Response Time (seconds)")
    ax.set_title("Response Time by Question", fontsize=14, fontweight="bold", pad=15)
    avg_time = np.mean(times) if times else 0
    ax.axhline(y=avg_time, color="#818cf8", linestyle="--", alpha=0.6, label=f"Avg: {avg_time:.1f}s")
    ax.legend(loc="upper right", framealpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / "response_times.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── 6. Grade distribution pie ──
    from collections import Counter
    grades = Counter(r.grade for r in results)
    fig, ax = plt.subplots(figsize=(6, 6))
    grade_order = ["A", "B", "C", "D", "F"]
    sizes = [grades.get(g, 0) for g in grade_order]
    grade_colors = [COLORS.get(g, "#64748b") for g in grade_order]
    non_zero = [(g, s, c) for g, s, c in zip(grade_order, sizes, grade_colors) if s > 0]
    if non_zero:
        labels, sizes, colors_pie = zip(*non_zero)
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, colors=colors_pie,
            autopct="%1.0f%%", startangle=90,
            textprops={"color": "#e2e8f0", "fontsize": 12},
        )
        for t in autotexts:
            t.set_color("#0f172a")
            t.set_fontweight("bold")
    ax.set_title("Grade Distribution", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    fig.savefig(output_dir / "grade_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  [OK] 6 charts saved to {output_dir}/")


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(results: list[QuestionResult], output_dir: Path, model: str, judge_model: str):
    """Generate a markdown report with embedded charts."""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    total_weighted = sum(r.percentage * DIFFICULTY_WEIGHTS[r.difficulty] for r in results)
    max_weighted = sum(100 * DIFFICULTY_WEIGHTS[r.difficulty] for r in results)
    overall_pct = total_weighted / max_weighted * 100 if max_weighted else 0
    overall_grade = "A" if overall_pct >= 90 else "B" if overall_pct >= 75 else "C" if overall_pct >= 60 else "D" if overall_pct >= 40 else "F"

    avg_time = sum(r.response_time_s for r in results) / len(results) if results else 0
    errors = sum(1 for r in results if r.error)

    lines = [
        f"# QUASAR Benchmark Report",
        f"",
        f"**Date:** {now}",
        f"**Model Under Test:** `{model}`",
        f"**Judge Model:** `{judge_model}`",
        f"**Questions:** {len(results)}",
        f"",
        f"---",
        f"",
        f"## Overall Results",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| **Weighted Score** | **{overall_pct:.1f}%** |",
        f"| **Overall Grade** | **{overall_grade}** |",
        f"| Avg Response Time | {avg_time:.1f}s |",
        f"| Errors | {errors}/{len(results)} |",
        f"",
        f"---",
        f"",
        f"## Visualizations",
        f"",
        f"### Scores by Question",
        f"![Scores by Question](scores_by_question.png)",
        f"",
        f"### Average Scores by Dimension",
        f"![Radar](radar_dimensions.png)",
        f"",
        f"### Scores by Category",
        f"![Category](scores_by_category.png)",
        f"",
        f"### Scores by Difficulty",
        f"![Difficulty](scores_by_difficulty.png)",
        f"",
        f"### Response Times",
        f"![Times](response_times.png)",
        f"",
        f"### Grade Distribution",
        f"![Grades](grade_distribution.png)",
        f"",
        f"---",
        f"",
        f"## Detailed Results",
        f"",
        f"| ID | Category | Difficulty | Score | Grade | Time |",
        f"|----|----------|------------|-------|-------|------|",
    ]

    for r in results:
        lines.append(
            f"| {r.id} | {r.category} | {r.difficulty} | "
            f"{r.total_score}/{r.max_score} ({r.percentage:.0f}%) | {r.grade} | {r.response_time_s:.1f}s |"
        )

    lines += [
        f"",
        f"---",
        f"",
        f"## Question Details",
        f"",
    ]

    for r in results:
        status = "❌ ERROR" if r.error else f"{r.grade} ({r.percentage:.0f}%)"
        lines += [
            f"### {r.id}: {r.title}",
            f"",
            f"**Status:** {status}  ",
            f"**Response Time:** {r.response_time_s:.1f}s",
            f"",
            f"| Dimension | Score |",
            f"|-----------|-------|",
            f"| Correctness | {r.correctness}/5 |",
            f"| Completeness | {r.completeness}/5 |",
            f"| Code Quality | {'N/A' if r.code_quality is None else f'{r.code_quality}/5'} |",
            f"| Presentation | {r.presentation}/5 |",
            f"| Tool Usage | {r.tool_usage}/5 |",
            f"| **Total** | **{r.total_score}/{r.max_score}** |",
            f"",
            f"**Judge Reasoning:** {r.judge_reasoning}",
            f"",
            f"<details><summary>Full Response</summary>",
            f"",
            f"```",
            f"{r.response[:3000]}{'...' if len(r.response) > 3000 else ''}",
            f"```",
            f"</details>",
            f"",
            f"---",
            f"",
        ]

    report_path = output_dir / "benchmark_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [OK] Report saved to {report_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="QUASAR Benchmark Runner")
    parser.add_argument("--api-url", default="http://localhost:8000",
                        help="Quasar API base URL (default: http://localhost:8000)")
    parser.add_argument("--model", default="gpt-4.1",
                        help="Model to test (default: gpt-4.1)")
    parser.add_argument("--judge-model", default="gpt-4o",
                        help="Model for LLM judge scoring (default: gpt-4o)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory for results (default: Benchmark/results/<timestamp>)")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Max seconds to wait per question (default: 300)")
    parser.add_argument("--questions", nargs="*", default=None,
                        help="Specific question IDs to run (e.g., GK-E-01 AM-M-02). Default: all")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print questions without running them")
    args = parser.parse_args()

    # Select questions
    if args.questions:
        selected = [q for q in QUESTIONS if q["id"] in args.questions]
        if not selected:
            print(f"No questions matched IDs: {args.questions}")
            sys.exit(1)
    else:
        selected = QUESTIONS

    if args.dry_run:
        print(f"\n{'='*60}")
        print(f"  QUASAR Benchmark — Dry Run ({len(selected)} questions)")
        print(f"{'='*60}\n")
        for q in selected:
            print(f"  [{q['id']}] ({q['difficulty']}) {q['title']}")
            print(f"    -> {q['question'][:80]}...")
            print()
        return

    # Setup output
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(__file__).parent / "results" / f"{timestamp}_{args.model.replace('/', '_')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  QUASAR Benchmark Runner")
    print(f"{'='*60}")
    print(f"  API:        {args.api_url}")
    print(f"  Model:      {args.model}")
    print(f"  Judge:      {args.judge_model}")
    print(f"  Questions:  {len(selected)}")
    print(f"  Output:     {output_dir}")
    print(f"{'='*60}\n")

    results: list[QuestionResult] = []

    for i, q in enumerate(selected, 1):
        print(f"  [{i}/{len(selected)}] {q['id']}: {q['title']} ({q['difficulty']})")

        result = QuestionResult(
            id=q["id"],
            category=q["category"],
            difficulty=q["difficulty"],
            title=q["title"],
            question=q["question"],
        )

        # ── Step 1: Query Quasar ──
        try:
            print(f"    -> Querying Quasar API...", end=" ", flush=True)
            response, elapsed = query_quasar_api(q["question"], args.api_url, args.model, args.timeout)
            result.response = response
            result.response_time_s = round(elapsed, 2)
            print(f"[OK] ({elapsed:.1f}s, {len(response)} chars)")
        except Exception as e:
            result.error = str(e)
            result.response_time_s = 0
            print(f"[ERR] Error: {e}")

        # ── Step 2: Judge the response ──
        try:
            print(f"    -> Judging response...", end=" ", flush=True)
            scores = judge_response(q, result.response, args.judge_model)
            result.correctness = scores.get("correctness", 0)
            result.completeness = scores.get("completeness", 0)
            result.code_quality = scores.get("code_quality")
            result.presentation = scores.get("presentation", 0)
            result.tool_usage = scores.get("tool_usage", 0)
            result.criteria_met = scores.get("criteria_met", [])
            result.judge_reasoning = scores.get("reasoning", "")
            print(f"[OK] Score: {result.total_score}/{result.max_score} ({result.percentage:.0f}%) [{result.grade}]")
        except Exception as e:
            print(f"[ERR] Judge error: {e}")

        results.append(result)
        print()

    # ── Save raw JSON results ──
    json_path = output_dir / "results.json"
    json_data = {
        "metadata": {
            "timestamp": timestamp,
            "api_url": args.api_url,
            "model": args.model,
            "judge_model": args.judge_model,
            "total_questions": len(results),
        },
        "results": [asdict(r) for r in results],
    }
    json_path.write_text(json.dumps(json_data, indent=2, default=str), encoding="utf-8")
    print(f"  [OK] Raw results saved to {json_path}")

    # ── Generate charts ──
    print("\n  Generating charts...")
    try:
        generate_charts(results, output_dir)
    except ImportError:
        print("  [ERR] matplotlib/numpy not installed — skipping charts")
        print("    Install with: pip install matplotlib numpy")

    # ── Generate report ──
    print("\n  Generating report...")
    generate_report(results, output_dir, args.model, args.judge_model)

    # ── Summary ──
    total_weighted = sum(r.percentage * DIFFICULTY_WEIGHTS[r.difficulty] for r in results)
    max_weighted = sum(100 * DIFFICULTY_WEIGHTS[r.difficulty] for r in results)
    overall_pct = total_weighted / max_weighted * 100 if max_weighted else 0

    print(f"\n{'='*60}")
    print(f"  BENCHMARK COMPLETE")
    print(f"{'='*60}")
    print(f"  Overall Weighted Score: {overall_pct:.1f}%")
    print(f"  Results: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

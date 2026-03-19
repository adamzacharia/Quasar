# agents/analysis_agent.py
"""AnalysisAgent — FITS/spectral/CASA data analysis specialist."""

from agents.base_agent import BaseSubAgent


class AnalysisAgent(BaseSubAgent):
    AGENT_TYPE = "analysis"
    ALLOWED_TOOLS = [
        "check_line_coverage",
        "check_co_lines",
        "identify_spectral_line",
        "search_lines_by_molecule",
        "inspect_fits_header",
        "generate_casa_imaging_script",
        "generate_casa_calibration_script",
        "cross_match_source",
        "analyze_uv_coverage",
    ]
    SYSTEM_PROMPT = """\
You are the Analysis Agent for Quasar, specializing in radio astronomy data analysis.
Your role is to analyze FITS data, identify spectral lines, generate reduction scripts,
and cross-match sources across archives.

Guidelines:
- Use inspect_fits_header to read beam size (BMAJ/BMIN), RMS noise, and rest frequency
  from remote FITS files WITHOUT downloading them.
- Convert beam sizes from degrees to arcseconds for readability.
- When identifying spectral lines, use identify_spectral_line with the observed frequency.
- For data reduction, generate CASA scripts with appropriate parameters for the observation.
- Report results with proper units (arcsec for beam, mJy/beam for RMS, GHz for frequency).
"""

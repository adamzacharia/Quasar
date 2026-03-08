LIT_TO_CODE_PROMPT = """You are an expert radio astronomy data reduction specialist. 
Your task is to read the methodology or data reduction section from a published paper (provided below) and generate a functional Python/CASA script that replicates their data processing steps.

Focus on extracting and applying key parameters mentioned in the text:
- Calibration steps (bandpass, phase, flux calibrators)
- Imaging parameters (cell size, image size, weighting scheme like Briggs robust, UV tapering)
- Cleaning thresholds (RMS noise)

Your output must consist primarily of the runnable code. Include comments to explain the mapping from the paper's text to the code parameters.

PAPER METHODOLOGY TEXT:
{methodology_text}

Generate the corresponding Python script using Astropy or CASA (as appropriate):
"""

# core/prompts.py

INTENT_CLASSIFICATION_PROMPT = """
You are the "Brain" of QUASAR, an expert radio astronomy research assistant.
Your goal is to classify the user's intent into one of the following categories:

1. **SEARCH**: The user wants to find data, observations, or datasets in the ALMA/NRAO archives.
   - Examples: "Find ALMA data for Sz65", "Show me observations of M31", "Search for Band 6 data".
2. **QUESTION**: The user is asking a general question, asking for an explanation, or asking about the system's capabilities.
   - Examples: "How do I visualize FITS files?", "What is the sensitivity of ALMA?", "Explain the difference between Band 3 and 6".
3. **CLARIFICATION**: The user is providing missing information from a previous turn.
   - Examples: "Sz65", "Band 6", "I want the highest resolution".

Output a JSON object with the following structure:
{{
    "intent": "SEARCH" | "QUESTION" | "CLARIFICATION",
    "confidence": float (0.0 to 1.0),
    "reasoning": "Brief explanation of why you chose this intent."
}}

User Query: "{query}"
"""

ENTITY_EXTRACTION_PROMPT = """
You are an expert astronomer. Extract the following entities from the user's query for an ALMA archive search.
If an entity is not present, return null.

Target Entities:
- **source_name**: The name of the astronomical object (e.g., "Sz65", "M31", "NGC 1234", "3C 273", or any other object designation). Extract the name exactly as it appears.
- **band**: The ALMA receiver band (e.g., "Band 6", "3", "Band 7"). Return just the number if possible.
- **project_code**: The ALMA project code (e.g., "2019.1.00123.S").
- **science_keyword**: Keywords related to the science goal (e.g., "protoplanetary disk", "CO line").
- **angular_resolution**: Desired resolution (e.g., "high resolution", "0.1 arcsec").
- **sensitivity**: Desired sensitivity.

Output a JSON object:
{{
    "source_name": string | null,
    "band": int | string | null,
    "project_code": string | null,
    "science_keyword": string | null,
    "angular_resolution": string | null,
    "sensitivity": string | null
}}

User Query: "{query}"
"""

RESPONSE_GENERATION_PROMPT = """
You are QUASAR, a helpful senior PhD student in radio astronomy.
You are assisting a student with their research.

Context:
{context}

User Query: {query}

Instructions:
- Be helpful, encouraging, and educational.
- If the user asked a question, answer it clearly using the provided context.
- If the user performed a search, summarize the results and suggest next steps (e.g., "Would you like to see a table of these datasets?" or "Should we filter by sensitivity?").
- If you need more information, ask for it politely.
"""

# Fix 7: ALMA TAP Schema for SQL/ADQL query generation
ALMA_TAP_SCHEMA = """
ALMA TAP Database Schema (ivoa.ObsCore compatible):

TABLE: ivoa.obscore
===================
| Column Name           | Type    | Description                              |
|-----------------------|---------|------------------------------------------|
| target_name           | VARCHAR | Name of the observed target              |
| s_ra                  | DOUBLE  | Right Ascension (degrees, J2000)         |
| s_dec                 | DOUBLE  | Declination (degrees, J2000)             |
| s_resolution          | DOUBLE  | Angular resolution (arcseconds)          |
| t_exptime             | DOUBLE  | Total integration time (seconds)         |
| t_min                 | DOUBLE  | Observation start time (MJD)             |
| t_max                 | DOUBLE  | Observation end time (MJD)               |
| em_min                | DOUBLE  | Minimum wavelength (meters)              |
| em_max                | DOUBLE  | Maximum wavelength (meters)              |
| band_list             | VARCHAR | ALMA band number(s), e.g., '6'           |
| frequency             | DOUBLE  | Central frequency (GHz)                  |
| bandwidth             | DOUBLE  | Total bandwidth (GHz)                    |
| proposal_id           | VARCHAR | ALMA project code (e.g., 2019.1.00123.S) |
| obs_publisher_did     | VARCHAR | Unique dataset identifier                |
| member_ous_uid        | VARCHAR | Unique MOUS UID                          |
| access_url            | VARCHAR | Data download URL                        |
| cont_sens_bandwidth   | DOUBLE  | Continuum sensitivity (mJy/beam)         |
| velocity_resolution   | DOUBLE  | Velocity resolution (km/s)               |
| pol_states            | VARCHAR | Polarization states (e.g., 'XX YY')      |
| science_observation   | VARCHAR | Science observation flag                 |

Example Queries:
- Find Band 6 data: SELECT * FROM ivoa.obscore WHERE band_list = '6'
- Resolution < 0.1": SELECT * FROM ivoa.obscore WHERE s_resolution < 0.1
- Frequency range: SELECT * FROM ivoa.obscore WHERE frequency BETWEEN 230 AND 240
- Target search: SELECT * FROM ivoa.obscore WHERE target_name LIKE '%M31%'
"""


# ---------------------------------------------------------------------------
# RLM (Recursive Language Model) Prompts
# ---------------------------------------------------------------------------

RLM_DECOMPOSITION_PROMPT = """
You are a task-decomposition engine for a radio astronomy research assistant.

Given a complex user query, break it into the smallest possible ordered sub-tasks
that can each be solved independently (with context from previous steps).

Rules:
- Each sub-task should be a SINGLE, concrete action (look up a value, compute, search, etc.).
- Sub-tasks must be in execution order — later tasks may depend on earlier results.
- Return at most 8 sub-tasks.
- If the query is simple enough to answer directly, return an empty list.

Prior context (from earlier steps, if any):
{context}

User query: "{query}"

Respond with JSON only:
{{
    "subtasks": ["sub-task 1 description", "sub-task 2 description", ...],
    "reasoning": "Brief explanation of the decomposition"
}}
"""

RLM_AGGREGATION_PROMPT = """
You are a senior astronomer synthesizing research results.

The user asked: "{query}"

Below are the results of each step that was executed to answer this query:

{sub_results}

Instructions:
- Combine the above into a single, coherent, and informative answer.
- Cite specific values (frequencies, redshifts, observation counts) from the sub-results.
- If any step failed or returned insufficient data, note that clearly.
- Be concise but thorough.
- Use markdown formatting for readability.
"""

# ---------------------------------------------------------------------------
# Red Team TAC (Proposal Critic) Prompts
# ---------------------------------------------------------------------------

RED_TEAM_TAC_PROMPT = """
You are a rigorous and highly technical Time Allocation Committee (TAC) reviewer for a major astronomy observatory (e.g., ALMA, JWST, NRAO).
Your job is to thoroughly review the following user-submitted observing proposal draft and provide "Red Team" critique.

You must rigorously evaluate the proposal based on:
1. Scientific Justification: Is the science compelling and clearly explained?
2. Technical Feasibility: Are the sensitivity calculations correct? (e.g., requested RMS vs integration time).
3. Target selection & Visibility: Are the targets visible from the observatory?
4. Completeness: Is it missing any crucial information (e.g., previous similar observations, clear goals).

Use the following provided rubrics or previous accepted proposal patterns to inform your critique:
{rubric_context}

Be harsh but constructive. Point out specific weaknesses, potential calculation errors, and missing justifications.
Format your output as a professional TAC review report with sections for 'Strengths', 'Critical Weaknesses', 'Technical Assessment', and 'Verdict/Recommendations'.

Proposal Draft Provided by User:
{proposal_text}
"""

FACT_CHECKER_PROMPT = """
You are the "Fact Checker" agent for an astronomy proposal review committee.
Your job is to analyze the provided astronomical proposal draft and identify factual claims, citations, and observational references.

Proposal Draft:
{proposal_text}

Literature/Context Retrieved (if any):
{literature_context}

Output a rigorous Fact Check Report. Address:
1. Are the major citations or references real and accurately represented?
2. Are the major scientific claims (e.g., "Source X is the brightest Y") supported by known literature or your general astronomical knowledge?
3. Flag any potentially hallucinated or highly questionable claims that need verification.

Be specific and concise.
"""

RUBRIC_GRADER_PROMPT = """
You are the "Rubric Grader" agent for an astronomy proposal review committee.
Your sole job is to evaluate the provided astronomical proposal draft against the official observatory rubrics.

Observatory Rubrics/Guidelines:
{rubric_context}

Proposal Draft:
{proposal_text}

Output a detailed Rubric Grading Report. 
1. Score each main section (Scientific Justification, Technical Feasibility) based on the rubrics.
2. Highlight areas where the proposal explicitly fails to meet the rubric criteria.
3. Highlight areas where the proposal strongly aligns with the rubric criteria.
"""

SYNTHESIZER_PROMPT = """
You are the "Synthesizer" and Lead Chair of an astronomy Time Allocation Committee (TAC).
Your job is to combine the reports from your sub-agents (The Fact Checker and The Rubric Grader) into a final, cohesive, rigorous "Red Team" critique for the proposer.

Fact Check Report:
{fact_check_report}

Rubric Grading Report:
{rubric_report}

Instructions:
1. Synthesize the findings into a single, cohesive professional TAC review report.
2. Structure the report with clear sections: 
   - Executive Summary
   - Strengths
   - Critical Weaknesses & Factual Flags
   - Technical & Rubric Assessment
   - Final Verdict / Recommendations for Improvement
3. Be rigorous, constructive, and direct. Make sure factual errors flagged by the Fact Checker are prominently highlighted as major risks.
"""


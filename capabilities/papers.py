"""
capabilities/papers.py — the ADS / papers / OpenAlex literature family as
transport-pure capabilities (P1 family migration #3, after datalab and alma).

Logic relocated VERBATIM from ``core/agent.py`` (the ``_search_papers`` /
``_search_papers_by_observation_identifier`` / ``_lookup_researcher`` /
``_get_research_trends`` / ``_evaluate_consensus`` / ``_extract_paper_details``
/ ``_reproduce_paper_methods`` methods plus the
``_derive_archive_identifiers_for_paper_search`` helper). The only changes are
structural (same rules as capabilities/alma.py):

  * clients/services arrive via ``CallContext.services`` (``ads_client``,
    ``openalex_client``, ``search_service``, ``pdf_service``, ``llm_client``,
    ``agent_config``) — never ``self.*`` on the agent. They are fetched with
    ``ctx.services.get(...)`` (not ``ctx.service(...)``) at the SAME code
    position the legacy body read the attribute, so a missing client flows
    into the identical guard/except path instead of raising the adapter's
    generic context error (the CX-03/CX-04 locality rule);
  * the two tools whose legacy registration LAMBDA carried an
    ``if self.ads_client else {"error": "ADS client not configured"}`` gate
    (``search_papers``, ``search_papers_by_observation_id``) reproduce that
    gate — including its success-key-less dict — as the first statement of
    ``run()``, so the whole legacy function surface (lambda + method) lives in
    ONE implementation. The method's own inner ads_client guard is kept
    verbatim below it (dead via the tool path, exactly as it was);
  * the agent's thread-local ``last_run_result`` is written ONLY through the
    injected ``set_last_run_result`` accessor (TLS semantics stay agent-side);
  * the legacy ``print()`` enrichment traces go through the injected
    ``console_log`` (the agent binds ``print``) — byte-identical stdout;
  * the LLM calls (``self.client.responses.create(model=self.config.model)``
    in evaluate_consensus / reproduce_paper_methods) read the injected
    ``llm_client`` + ``agent_config`` at the same in-body position, so a
    partially-constructed agent produces the same catch-and-return error
    shape as the legacy attribute access did;
  * every path returns a :class:`ToolResult` whose ``native`` payload is the
    exact legacy output dict (byte-parity). Exceptions the legacy method did
    NOT catch still propagate.

Known benign divergence (documented, observability-only): the legacy
``@log_tool`` span for ``search_papers_by_observation_id`` fired only AFTER
the registration lambda's ads_client gate; re-applied at the adapter boundary
(``_papers_tool_fn(log_name=...)``) it now also wraps the gate path. The
returned dict on that path is unchanged.

``LIT_TO_CODE_PROMPT`` is imported from ``core.prompts.lit_to_code`` — a pure
string-constant module (no core machinery, no import cycle; the agent imports
this module lazily inside ``_papers_tool_fn``).
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict

from capabilities.base import BaseCapability, Provenance, ToolResult
from core.prompts.lit_to_code import LIT_TO_CODE_PROMPT
from services.alma_science_queries import (
    publication_join_where,
    select_obscore_query_extended,
    summarize_publication_links,
)

logger = logging.getLogger(__name__)


def _native(out: Dict[str, Any], *, ads_query: Optional[str] = None) -> ToolResult:
    """Wrap a legacy output dict as a byte-parity ToolResult.

    ``ads_query`` (the TRANSLATED ADS query string actually executed, not the
    user's natural-language arguments) is carried as provenance so the query-
    provenance surface shows the real request (Feature 1). It never enters the
    model-facing dict — it rides the adapter sidecar.
    """
    ok = bool(isinstance(out, dict) and out.get("success"))
    err = out.get("error") if isinstance(out, dict) else None
    prov = None
    if ads_query:
        prov = Provenance(service="ads", query=str(ads_query),
                          endpoint="https://api.adsabs.harvard.edu/v1/search/query")
    return ToolResult(
        success=ok,
        error=(str(err) if (err is not None and not ok) else None),
        native=out,
        provenance=prov,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Shared helper (relocated verbatim from agent.py)
# ─────────────────────────────────────────────────────────────────────────────
def derive_archive_identifiers_for_paper_search(
    identifier: str,
    *,
    ads_client: Any,
    search_service: Any,
) -> List[str]:
    """Resolve a MOUS/dataset identifier to proposal/project IDs when possible."""
    raw = str(identifier or "").strip()
    if not raw or not search_service:
        return []

    id_type = ads_client.classify_observation_identifier(raw) if ads_client else "identifier"
    if id_type == "project_code":
        return []

    keyword_by_type = {
        "mous_uid": "member_ous_uid",
        "asdm_uid": "asdm_uid",
        "dataset_id": "obs_publisher_did",
        "uid": "member_ous_uid",
    }
    keyword = keyword_by_type.get(id_type)
    if not keyword:
        return []

    df = search_service.search_alma_with_keywords({keyword: raw})
    if df is None or not hasattr(df, "columns") or df.empty:
        return []

    derived = []
    for col in ("proposal_id", "project_code"):
        if col in df.columns:
            for value in df[col].dropna().astype(str).unique().tolist()[:5]:
                clean = value.strip()
                if clean and clean not in derived:
                    derived.append(clean)
    return derived


# ─────────────────────────────────────────────────────────────────────────────
# Input models + capabilities (one Input right above its capability; required
# fields are REQUIRED-BUT-NULLABLE — Optional[...] with no default — so an
# explicit JSON null flows into the verbatim legacy body. guard CX-01)
# ─────────────────────────────────────────────────────────────────────────────
class _In(BaseModel):
    model_config = ConfigDict(extra="ignore")


class SearchPapersInput(_In):
    query: Optional[str]
    # The legacy registration lambda's defaults (max_results=15), NOT the
    # method signature's (10) — the lambda was the wire surface.
    max_results: Optional[int] = 15
    sort: Optional[str] = "date desc"


class SearchPapers(BaseCapability):
    name = "search_papers"
    description = (
        "Search the NASA ADS database for astronomical papers. "
        "Returns titles, authors, abstracts, citation counts, DOIs, and a direct link to each paper on NASA ADS. "
        "IMPORTANT: Pass the user's request as natural language — an internal AI query builder will "
        "automatically translate it into optimal ADS syntax using keyword searches, bibgroup filters, "
        "SIMBAD object linking, second-order discovery operators (trending, similar, useful), and more.\n"
        "Examples of what to pass as query:\n"
        "- 'recent papers on protoplanetary disks'\n"
        "- 'best ALMA papers on disk gaps'\n"
        "- 'papers about HL Tau'\n"
        "- 'what are people reading about FRBs right now'\n"
        "- 'foundational papers on planet formation'\n"
        "- 'review articles on AGN feedback'\n"
        "- 'papers by Sean Andrews on disk surveys'\n"
        "Do NOT try to construct ADS field syntax yourself — just pass the natural language query."
    )
    category = "literature"
    InputModel = SearchPapersInput
    annotations = {"read_only": True, "cost": "network"}

    def run(self, inp, ctx) -> ToolResult:
        ads_client = ctx.services.get("ads_client")
        # The legacy registration lambda's gate, verbatim (success-key-less dict).
        if not ads_client:
            return _native({"error": "ADS client not configured"})
        _log = ctx.service("console_log")
        set_last_run_result = ctx.service("set_last_run_result")
        query, max_results, sort = inp.query, inp.max_results, inp.sort
        try:
            if not ads_client:
                # Dead via the tool path (the lambda gate above already returned)
                # — preserved verbatim from the inline method body.
                return _native({"success": False, "error": "NASA ADS Client not initialized (check API Key)"})

            # Use the smart NL→ADS query builder for rich query translation
            result = ads_client.search_natural_language(
                question=query,
                max_results=max_results,
                sort=sort,
            )

            papers_list = result.get("papers", [])
            ads_query = result.get("query", query)

            # ── Silent OpenAlex enrichment ────────────────────────────
            # Batch-enrich papers with funding data, FWCI scores, citation
            # percentiles, and OA PDF URLs that ADS doesn't provide.
            # Failures are silently swallowed — enrichment is best-effort.
            try:
                oalex = ctx.services.get("openalex_client")
                dois = [p.get("doi", "") for p in papers_list if p.get("doi")]
                if dois and oalex:
                    enrichments = oalex.enrich_batch_dois(dois)
                    if enrichments:
                        _enriched_count = 0
                        for paper in papers_list:
                            doi = paper.get("doi", "")
                            if doi and doi in enrichments:
                                e = enrichments[doi]
                                paper["fwci"] = e.get("fwci")
                                paper["citation_percentile"] = e.get("citation_percentile")
                                paper["is_top_1_percent"] = e.get("is_top_1_percent", False)
                                paper["is_top_10_percent"] = e.get("is_top_10_percent", False)
                                paper["funders"] = e.get("funders", [])
                                paper["oa_pdf_url"] = e.get("oa_pdf_url", "")
                                paper["openalex_topics"] = e.get("topics", [])
                                _enriched_count += 1
                        _log(f"[OpenAlex] Enriched {_enriched_count}/{len(papers_list)} papers")
            except Exception as _enrich_err:
                _log(f"[OpenAlex] Enrichment failed (non-fatal): {_enrich_err}")

            # Store result for the UI backend to pick up
            set_last_run_result({
                "type": "papers",
                "papers": papers_list,
                "source": f"ADS: {ads_query}",
            })
            return _native({
                "success": True,
                "count": len(papers_list),
                "ads_query": ads_query,
                "papers": papers_list,
                "top_title": papers_list[0]["title"] if papers_list else "No results",
            }, ads_query=ads_query)
        except Exception as e:
            return _native({"success": False, "error": str(e)})


class SearchPapersByObservationIdInput(_In):
    identifier: Optional[str]
    max_results: Optional[int] = 20
    facility: Optional[str] = "ALMA"


class SearchPapersByObservationId(BaseCapability):
    name = "search_papers_by_observation_id"
    description = (
        "Find NASA ADS papers explicitly connected to a specific archive identifier. "
        "Use this instead of generic search_papers when the user provides an ALMA project/proposal code "
        "(e.g. 2019.1.00123.S), MOUS/member_ous_uid (uid://...), ASDM UID, or archive dataset ID. "
        "Also works in REVERSE: pass an ADS bibcode (e.g. 2018ApJ...869L..41A) to find the archived "
        "ALMA data that paper used (via the ObsCore bib_reference join). "
        "The lookup uses exact identifier searches and returns provenance metadata for the graph."
    )
    category = "literature"
    InputModel = SearchPapersByObservationIdInput
    annotations = {"read_only": True, "cost": "network"}

    def run(self, inp, ctx) -> ToolResult:
        ads_client = ctx.services.get("ads_client")
        # The legacy registration lambda's gate, verbatim (success-key-less dict).
        if not ads_client:
            return _native({"error": "ADS client not configured"})
        _log = ctx.service("console_log")
        set_last_run_result = ctx.service("set_last_run_result")
        identifier, max_results, facility = inp.identifier, inp.max_results, inp.facility
        try:
            if not ads_client:
                # Dead via the tool path — preserved verbatim from the method body.
                return _native({"success": False, "error": "NASA ADS Client not initialized (check API Key)"})

            identifiers_to_search = [str(identifier or "").strip()]
            derived_identifiers = []
            try:
                derived_identifiers = derive_archive_identifiers_for_paper_search(
                    identifier,
                    ads_client=ads_client,
                    search_service=ctx.services.get("search_service"),
                )
                for derived in derived_identifiers:
                    if derived and derived not in identifiers_to_search:
                        identifiers_to_search.append(derived)
            except Exception as _derive_err:
                _log(f"[ADS identifier] Archive identifier derivation failed (non-fatal): {_derive_err}")

            merged_papers: Dict[str, Dict[str, Any]] = {}
            ads_queries = []
            identifier_types = {}
            for search_identifier in identifiers_to_search:
                result = ads_client.search_by_observation_identifier(
                    identifier=search_identifier,
                    max_results=max_results,
                    facility=facility,
                )
                ads_queries.append(result.get("query", ""))
                identifier_types[search_identifier] = result.get("identifier_type", "identifier")
                for paper in result.get("papers", []):
                    key = paper.get("bibcode") or paper.get("doi") or paper.get("title") or str(id(paper))
                    if key not in merged_papers:
                        merged_papers[key] = paper
                    else:
                        current_links = merged_papers[key].setdefault("observation_links", [])
                        for link in paper.get("observation_links", []):
                            if link not in current_links:
                                current_links.append(link)

            papers_list = list(merged_papers.values())[:max_results]
            ads_query = " OR ".join(q for q in ads_queries if q)

            try:
                oalex = ctx.services.get("openalex_client")
                dois = [p.get("doi", "") for p in papers_list if p.get("doi")]
                if dois and oalex:
                    enrichments = oalex.enrich_batch_dois(dois)
                    if enrichments:
                        for paper in papers_list:
                            doi = paper.get("doi", "")
                            if doi and doi in enrichments:
                                e = enrichments[doi]
                                paper["fwci"] = e.get("fwci")
                                paper["citation_percentile"] = e.get("citation_percentile")
                                paper["is_top_1_percent"] = e.get("is_top_1_percent", False)
                                paper["is_top_10_percent"] = e.get("is_top_10_percent", False)
                                paper["funders"] = e.get("funders", [])
                                paper["oa_pdf_url"] = e.get("oa_pdf_url", "")
                                paper["openalex_topics"] = e.get("topics", [])
            except Exception as _enrich_err:
                _log(f"[OpenAlex] Enrichment failed (non-fatal): {_enrich_err}")

            # ── R2 reverse direction: bibcode → archived ALMA data ────────
            # Best-effort ObsCore bib_reference join; failures never break
            # the ADS paper search this tool has always performed.
            archival_data = None
            archive_query = None
            try:
                clean_id = str(identifier or "").strip()
                if ads_client.classify_observation_identifier(clean_id) == "bibcode":
                    search_service = ctx.services.get("search_service")
                    alminer_client = getattr(search_service, "alminer_client", None)
                    if alminer_client is not None:
                        where, _kind = publication_join_where(clean_id)
                        archive_query = select_obscore_query_extended(
                            where,
                            extra_columns=("bib_reference", "pub_title",
                                           "publication_year", "first_author"),
                            top=2000,
                        )
                        df = alminer_client.search_by_sql(archive_query)
                        if df is not None and hasattr(df, "empty") and not df.empty:
                            summary = summarize_publication_links(df)
                            archival_data = summary.head(50).to_dict("records")
            except Exception as _rev_err:
                _log(f"[ALMA bib_reference] Reverse data lookup failed (non-fatal): {_rev_err}")

            set_last_run_result({
                "type": "papers",
                "papers": papers_list,
                "source": f"ADS identifier: {identifier}",
                "paper_provenance": {
                    "identifier": identifier,
                    "identifier_type": identifier_types.get(str(identifier or "").strip(), "identifier"),
                    "derived_identifiers": derived_identifiers,
                    "ads_query": ads_query,
                    "facility": facility,
                },
            })
            out = {
                "success": True,
                "count": len(papers_list),
                "identifier": identifier,
                "identifier_type": identifier_types.get(str(identifier or "").strip(), "identifier"),
                "derived_identifiers": derived_identifiers,
                "ads_query": ads_query,
                "papers": papers_list,
                "top_title": papers_list[0]["title"] if papers_list else "No results",
            }
            if archival_data is not None:
                out["archival_data"] = archival_data
                out["archival_data_note"] = (
                    "ALMA observations whose ObsCore bib_reference lists this bibcode "
                    "(the archived data this paper used)."
                )
                out["archive_query"] = archive_query
            return _native(out, ads_query=ads_query)
        except Exception as e:
            return _native({"success": False, "error": str(e)})


class LookupResearcherInput(_In):
    query: Optional[str]
    max_results: Optional[int] = 3


class LookupResearcher(BaseCapability):
    name = "lookup_researcher"
    description = (
        "Look up a researcher/scientist by name or ORCID to get their full academic profile: "
        "current institution, h-index, i10-index, total publications, total citations, "
        "ORCID, Scopus ID, research topics, affiliation history, and publication trend "
        "over the last 10 years.  Powered by OpenAlex (90M+ disambiguated authors).\n"
        "Use this when the user asks about a person, wants to know who someone is, "
        "or wants contact/institutional information about a researcher.\n"
        "Examples: 'Who is Andrea Isella?', 'Tell me about Crystal Brogan', "
        "'Look up ORCID 0000-0001-2345-6789'"
    )
    category = "literature"
    InputModel = LookupResearcherInput
    annotations = {"read_only": True, "cost": "network"}

    def run(self, inp, ctx) -> ToolResult:
        query, max_results = inp.query, inp.max_results
        try:
            oalex = ctx.services.get("openalex_client")

            # Check if the query looks like an ORCID
            clean = query.strip().replace("https://orcid.org/", "")
            is_orcid = (
                clean.replace("-", "").isdigit() and len(clean) >= 16
            )

            if is_orcid:
                author = oalex.get_author(clean)
                if author:
                    return _native({
                        "success": True,
                        "count": 1,
                        "researchers": [author],
                    })
                return _native({"success": False, "error": f"No author found for ORCID {clean}"})

            # Name search
            authors = oalex.search_authors(query, max_results=max_results)
            if not authors:
                return _native({"success": False, "error": f"No researchers found matching '{query}'"})

            return _native({
                "success": True,
                "count": len(authors),
                "researchers": authors,
            })
        except Exception as e:
            return _native({"success": False, "error": str(e)})


class GetResearchTrendsInput(_In):
    query: Optional[str]
    year_from: Optional[int] = 2015
    year_to: Optional[int] = 2026


class GetResearchTrends(BaseCapability):
    name = "get_research_trends"
    description = (
        "Get a bibliometric trend showing papers-per-year for a given topic or search "
        "query.  Returns total paper count and yearly breakdown.\n"
        "Use when the user asks 'How much research is being done on X?', "
        "'Is interest in X growing?', 'Publication trends for FRBs'.\n"
        "Also returns the funding landscape — which funders (NSF, NASA, ESA, etc.) "
        "have funded research on the topic."
    )
    category = "literature"
    InputModel = GetResearchTrendsInput
    annotations = {"read_only": True, "cost": "network"}

    def run(self, inp, ctx) -> ToolResult:
        query, year_from, year_to = inp.query, inp.year_from, inp.year_to
        try:
            oalex = ctx.services.get("openalex_client")

            # Publication trends
            trends = oalex.get_topic_trends(
                query, year_from=year_from, year_to=year_to,
            )

            # Funding landscape (top funded works)
            funded = oalex.get_funding_landscape(query, max_results=15)

            return _native({
                "success": True,
                "trends": trends,
                "funded_works_count": len(funded),
                "top_funded_works": funded[:8],
            })
        except Exception as e:
            return _native({"success": False, "error": str(e)})


class EvaluateConsensusInput(_In):
    question: Optional[str]
    max_papers: Optional[int] = 20


class EvaluateConsensus(BaseCapability):
    name = "evaluate_consensus"
    description = (
        "Evaluate the scientific consensus on a research question by searching NASA ADS for the most-cited "
        "papers on the topic, reading all their abstracts, and producing a structured analysis. "
        "The output includes: overall consensus level (Strong Agreement / Divided / etc.), "
        "which specific papers agree vs disagree, WHY they disagree (different methods, data, assumptions), "
        "key evidence from each side with proper citations, how the consensus has evolved over time, "
        "and what open questions remain. "
        "Use this when the user asks questions like: 'Do scientists agree on X?', 'What does the field think about X?', "
        "'Is there consensus on X?', 'What's the current understanding of X?', or any question where "
        "a literature-wide summary would be more useful than individual paper results."
    )
    category = "literature"
    InputModel = EvaluateConsensusInput
    annotations = {"read_only": True, "cost": "llm"}

    def run(self, inp, ctx) -> ToolResult:
        set_last_run_result = ctx.service("set_last_run_result")
        question, max_papers = inp.question, inp.max_papers
        try:
            ads_client = ctx.services.get("ads_client")
            if not ads_client:
                return _native({"success": False, "error": "NASA ADS Client not initialized"})

            # Step 1: Search for highly-cited papers on the topic for authoritative consensus
            result = ads_client.search_natural_language(
                question=question,
                max_results=max_papers,
                sort="citation_count desc",
            )
            papers_list = result.get("papers", [])

            if not papers_list:
                return _native({"success": False, "error": "No papers found for this question."})

            # Step 2: Build a structured context with all abstracts + metadata
            paper_contexts = []
            for i, p in enumerate(papers_list, 1):
                paper_ctx = (
                    f"[{i}] {p.get('title', 'Unknown')} "
                    f"({p.get('authors', 'Unknown')}, {p.get('year', '?')})\n"
                    f"    Journal: {p.get('journal', 'Unknown')} | "
                    f"Citations: {p.get('citations', 0)} | "
                    f"Bibcode: {p.get('bibcode', '')}\n"
                    f"    Abstract: {p.get('abstract', 'No abstract')}\n"
                )
                paper_contexts.append(paper_ctx)

            all_papers_text = "\n".join(paper_contexts)

            # Step 3: Ask the LLM to evaluate consensus with detailed citations
            consensus_prompt = f"""\
You are an expert scientific literature analyst. You have been given {len(papers_list)} \
peer-reviewed papers retrieved from NASA ADS on the following question:

QUESTION: "{question}"

YOUR TASK: Analyze all the abstracts below and produce a detailed CONSENSUS EVALUATION.

PAPERS:
{all_papers_text}

PRODUCE YOUR ANALYSIS IN THIS EXACT FORMAT:

## 📊 Field Consensus: [Strong Agreement / Moderate Agreement / Divided / Strong Disagreement]
**Confidence:** [High / Medium / Low] (based on {len(papers_list)} papers, weighted by citation count)
**Papers Analyzed:** {len(papers_list)}

### Majority Position
State the dominant view clearly in 2-3 sentences. Cite the specific papers that support it using their [number] references.

Example: "The majority of the literature ([1], [3], [5], [7], [8], [12]) concludes that..."

### Dissenting/Alternative Views
If papers disagree, group them by their position. For EACH dissenting view:
- State the alternative conclusion
- List which papers support it (by [number])
- Explain WHY they reach a different conclusion (different methodology? different data? different assumptions? different telescope?)
- Cite the specific evidence or reasoning from their abstracts

If there are no dissenting views, state "No significant dissent found in the analyzed literature."

### Key Evidence Summary
Bullet-point the strongest pieces of evidence from both sides, citing specific papers:
- "[1] found that..."
- "[5] measured X and concluded..."
- "[9] used ALMA data showing..."

### Evolution Over Time
If the consensus has shifted over time (older papers say X, newer papers say Y), note this trend.

### Open Questions
What does the literature identify as unresolved? What would settle the debate?

IMPORTANT RULES:
- ALWAYS cite papers by their [number] reference
- Include the author name and year when first citing a paper
- Be specific about evidence — don't just say "some papers agree"
- Weight highly-cited papers more heavily in your assessment
- If a question is too narrow or the papers don't directly address it, say so honestly
"""

            llm_client = ctx.services.get("llm_client")
            config = ctx.services.get("agent_config")
            response = llm_client.responses.create(
                model=config.model,
                instructions="You are a meticulous scientific literature analyst who produces rigorous, well-cited consensus evaluations.",
                input=consensus_prompt,
                temperature=0.1,
                max_output_tokens=4000,
            )

            analysis = response.output_text.strip()

            # Store for UI rendering
            set_last_run_result({
                "type": "consensus",
                "text": analysis,
                "papers": papers_list,
                "question": question,
                "source": f"Consensus Analysis: {len(papers_list)} papers",
            })

            return _native({
                "success": True,
                "question": question,
                "papers_analyzed": len(papers_list),
                "analysis": analysis,
            })

        except Exception as e:
            return _native({"success": False, "error": str(e)})


# CAP-05: a real ADS bibcode is EXACTLY 19 characters, starts with a 4-digit
# year, and always contains dots (e.g. 2019ApJ...883..170M), so the legacy gate
# `'.' not in id or len(id) > 20` never matched one — every bibcode was sent
# straight to arxiv.org/pdf/<bibcode>.pdf and 404'd. Detect bibcodes
# positively, and keep a fast path for new-style (1812.04040) and old-style
# (astro-ph/9901001) arXiv ids that must never hit ADS resolution.
_ARXIV_NEW_STYLE_RE = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")
_ARXIV_OLD_STYLE_RE = re.compile(r"^[a-z\-]+(\.[A-Za-z]{2})?/\d{7}(v\d+)?$", re.IGNORECASE)
_ADS_BIBCODE_RE = re.compile(r"^\d{4}[A-Za-z]")


def _needs_ads_resolution(identifier: str) -> bool:
    """True when ``identifier`` should be resolved to an arXiv id via ADS (CAP-05)."""
    if _ARXIV_NEW_STYLE_RE.match(identifier) or _ARXIV_OLD_STYLE_RE.match(identifier):
        return False
    if len(identifier) == 19 and _ADS_BIBCODE_RE.match(identifier):
        return True  # canonical ADS bibcode
    # Legacy heuristic retained for anything else (dotless ids, long DOIs, ...).
    return "." not in identifier or len(identifier) > 20


class ExtractPaperDetailsInput(_In):
    identifier: Optional[str]
    query: Optional[str]


class ExtractPaperDetails(BaseCapability):
    name = "extract_paper_details"
    description = (
        "Download a scientific paper by its arXiv ID or ADS bibcode and extract specific details "
        "(e.g. beam size, flux density, telescope configuration) using an LLM QA pass over the full text. "
        "Use when the user asks specific questions about the contents of a published paper."
    )
    category = "literature"
    InputModel = ExtractPaperDetailsInput
    annotations = {"read_only": True, "cost": "network"}

    def run(self, inp, ctx) -> ToolResult:
        set_last_run_result = ctx.service("set_last_run_result")
        identifier, query = inp.identifier, inp.query
        try:
            ads_client = ctx.services.get("ads_client")
            arxiv_id = identifier.strip()
            if _needs_ads_resolution(arxiv_id):  # CAP-05: bibcodes now match
                if ads_client:
                    try:
                        details = ads_client.get_paper_details(arxiv_id)
                        if isinstance(details, dict) and details.get("arxiv_id"):
                            arxiv_id = details["arxiv_id"]
                        elif isinstance(details, dict) and details.get("doi"):
                            return _native({"success": False, "error": f"Paper has DOI ({details['doi']}) but no arXiv ID. PDF download requires an open access arXiv ID."})
                    except Exception:
                        pass

            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            pdf_service = ctx.services.get("pdf_service")
            result = pdf_service.query_paper_pdf(pdf_url, query)

            if not result.get("success"):
                return _native({"success": False, "error": f"Could not extract details: {result.get('error')}", "identifier": identifier})

            set_last_run_result({"type": "text", "text": result["answer"], "source": f"Paper Extractor: {identifier}"})
            return _native({"success": True, "identifier": identifier, "extracted_answer": result["answer"]})
        except Exception as e:
            return _native({"success": False, "error": str(e)})


class ReproducePaperMethodsInput(_In):
    identifier: Optional[str]


class ReproducePaperMethods(BaseCapability):
    name = "reproduce_paper_methods"
    description = (
        "Extract the methodology from a published paper (by arXiv ID or ADS bibcode) and construct a "
        "Python/CASA data reduction script that replicates its steps. Use when a user asks "
        "'how did they reduce the data for this paper' or 'reproduce this paper'."
    )
    category = "literature"
    InputModel = ReproducePaperMethodsInput
    annotations = {"read_only": True, "cost": "llm"}

    def run(self, inp, ctx) -> ToolResult:
        set_last_run_result = ctx.service("set_last_run_result")
        identifier = inp.identifier
        try:
            ads_client = ctx.services.get("ads_client")
            # Step 1: Resolve identifier to a PDF URL
            # Try arXiv first (most common for astro papers)
            arxiv_id = identifier.strip()
            # If it looks like a bibcode, try to get the arXiv ID from ADS
            if _needs_ads_resolution(arxiv_id):  # CAP-05: bibcodes now match
                # Likely an ADS bibcode — try to resolve via ADS
                if ads_client:
                    try:
                        details = ads_client.get_paper_details(arxiv_id)
                        if isinstance(details, dict) and details.get("arxiv_id"):
                            arxiv_id = details["arxiv_id"]
                        elif isinstance(details, dict) and details.get("doi"):
                            return _native({
                                "success": False,
                                "error": f"Paper has DOI ({details['doi']}) but no arXiv ID. "
                                         "PDF download is only supported for arXiv papers currently."
                            })
                    except Exception:
                        pass  # Fall through and try the identifier as-is

            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

            # Step 2: Download and extract methodology
            pdf_service = ctx.services.get("pdf_service")
            result = pdf_service.get_paper_methodology_from_url(pdf_url)

            if not result.get("success"):
                return _native({
                    "success": False,
                    "error": f"Could not extract methodology: {result.get('error', 'Unknown error')}",
                    "identifier": identifier
                })

            methodology_text = result["methodology"]

            # Step 3: Generate reproduction script using LIT_TO_CODE_PROMPT
            prompt = LIT_TO_CODE_PROMPT.format(methodology_text=methodology_text)

            llm_client = ctx.services.get("llm_client")
            config = ctx.services.get("agent_config")
            response = llm_client.responses.create(
                model=config.model,
                instructions="You are an expert radio astronomy data reduction specialist.",
                input=prompt,
                temperature=0.2,
                max_output_tokens=4000
            )

            script = response.output_text.strip()

            set_last_run_result({
                "type": "code",
                "code": script,
                "source": f"Reproduce: {identifier}"
            })

            return _native({
                "success": True,
                "identifier": identifier,
                "methodology_summary": methodology_text[:500] + "..." if len(methodology_text) > 500 else methodology_text,
                "generated_script": script
            })

        except Exception as e:
            return _native({"success": False, "error": str(e), "identifier": identifier})


CAPABILITIES: List[BaseCapability] = [
    SearchPapers(),
    SearchPapersByObservationId(),
    LookupResearcher(),
    GetResearchTrends(),
    EvaluateConsensus(),
    ExtractPaperDetails(),
    ReproducePaperMethods(),
]

__all__ = [
    "CAPABILITIES",
    "derive_archive_identifiers_for_paper_search",
    "SearchPapers", "SearchPapersByObservationId", "LookupResearcher",
    "GetResearchTrends", "EvaluateConsensus", "ExtractPaperDetails",
    "ReproducePaperMethods",
]

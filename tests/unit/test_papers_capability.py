"""Unit tests for capabilities/papers.py (P1 family migration #3).

Mirrors tests/unit/test_alma_capability.py: byte-parity checks on the relocated
legacy bodies, the lambda-gate reproduction, CX-01 null flow, the OpenAlex
enrichment traces, and the agent wiring (_papers_tool_fn / _papers_ctx_provider
/ registration surface).
"""

import threading

import pandas as pd
import pytest

from capabilities.papers import (
    CAPABILITIES,
    EvaluateConsensus,
    ExtractPaperDetails,
    GetResearchTrends,
    LookupResearcher,
    ReproducePaperMethods,
    SearchPapers,
    SearchPapersByObservationId,
    derive_archive_identifiers_for_paper_search,
)
from capabilities.base import CallContext


# ─────────────────────────────────────────────────────────────────────────────
# Harness
# ─────────────────────────────────────────────────────────────────────────────
class _State:
    def __init__(self):
        self.last_run_result = None
        self.console = []


def _ctx(state=None, **service_overrides):
    state = state or _State()

    def _set_lrr(v):
        state.last_run_result = v

    services = {
        "console_log": state.console.append,
        "set_last_run_result": _set_lrr,
    }
    services.update(service_overrides)
    return CallContext(services=services), state


def _run(cap, ctx, **kwargs):
    return cap.run(cap.InputModel(**kwargs), ctx).to_native()


_PAPERS = [
    {"title": "Rings in HL Tau", "bibcode": "2015ApJ...808L...3A", "doi": "10.1/a",
     "authors": "Brogan et al.", "year": 2015, "journal": "ApJ",
     "citations": 2000, "abstract": "Rings."},
    {"title": "Disk gaps", "bibcode": "2018ApJ...869L..41A", "doi": "",
     "authors": "Andrews et al.", "year": 2018, "journal": "ApJL",
     "citations": 900, "abstract": "Gaps."},
]


class _FakeAds:
    def __init__(self, papers=None, id_type="project_code"):
        self.papers = papers if papers is not None else [dict(p) for p in _PAPERS]
        self.id_type = id_type
        self.calls = []
        self.details = {}

    def search_natural_language(self, question, max_results, sort):
        self.calls.append(("nl", question, max_results, sort))
        return {"papers": [dict(p) for p in self.papers], "query": f"ads({question})"}

    def search_by_observation_identifier(self, identifier, max_results, facility):
        self.calls.append(("obsid", identifier, max_results, facility))
        papers = [dict(p, observation_links=[f"link-{identifier}"]) for p in self.papers]
        return {"papers": papers, "query": f"id:{identifier}", "identifier_type": "project_code"}

    def classify_observation_identifier(self, raw):
        self.calls.append(("classify", raw))
        return self.id_type

    def get_paper_details(self, bibcode):
        self.calls.append(("details", bibcode))
        return self.details


class _FakeOalex:
    def __init__(self, enrich=None, author=None, authors=None, raise_on_enrich=False):
        self.enrich = enrich or {}
        self.author = author
        self.authors = authors or []
        self.raise_on_enrich = raise_on_enrich
        self.calls = []

    def enrich_batch_dois(self, dois):
        self.calls.append(("enrich", tuple(dois)))
        if self.raise_on_enrich:
            raise RuntimeError("openalex down")
        return self.enrich

    def get_author(self, orcid):
        self.calls.append(("get_author", orcid))
        return self.author

    def search_authors(self, query, max_results):
        self.calls.append(("search_authors", query, max_results))
        return self.authors

    def get_topic_trends(self, query, year_from, year_to):
        self.calls.append(("trends", query, year_from, year_to))
        return {"total": 42, "by_year": {2020: 10}}

    def get_funding_landscape(self, query, max_results):
        self.calls.append(("funding", query, max_results))
        return [{"id": i} for i in range(12)]


class _FakePdf:
    def __init__(self, qa=None, methodology=None):
        self.qa = qa or {"success": True, "answer": "0.035 arcsec"}
        self.methodology = methodology or {"success": True, "methodology": "CASA tclean, robust=0.5"}
        self.calls = []

    def query_paper_pdf(self, url, query):
        self.calls.append(("qa", url, query))
        return self.qa

    def get_paper_methodology_from_url(self, url):
        self.calls.append(("methodology", url))
        return self.methodology


class _FakeLLM:
    def __init__(self, text="ANALYSIS TEXT"):
        self.text = text
        self.kwargs = None
        outer = self

        class _Responses:
            def create(self, **kw):
                outer.kwargs = kw

                class _R:
                    output_text = outer.text
                return _R()

        self.responses = _Responses()


class _FakeConfig:
    model = "test-model"
    user_id = None


# ─────────────────────────────────────────────────────────────────────────────
# search_papers
# ─────────────────────────────────────────────────────────────────────────────
def test_search_papers_success_sets_papers_card():
    ads = _FakeAds()
    ctx, state = _ctx(ads_client=ads)
    out = _run(SearchPapers(), ctx, query="rings in HL Tau")

    assert out["success"] is True and out["count"] == 2
    assert out["ads_query"] == "ads(rings in HL Tau)"
    assert out["top_title"] == "Rings in HL Tau"
    assert state.last_run_result["type"] == "papers"
    assert state.last_run_result["source"] == "ADS: ads(rings in HL Tau)"


def test_search_papers_lambda_gate_dict_is_verbatim():
    # The legacy registration lambda returned this success-key-less dict.
    ctx, _ = _ctx(ads_client=None)
    res = SearchPapers().run(SearchPapers.InputModel(query="x"), ctx)
    assert res.to_native() == {"error": "ADS client not configured"}
    assert res.success is False


def test_search_papers_wire_default_is_the_lambdas_15():
    ads = _FakeAds()
    ctx, _ = _ctx(ads_client=ads)
    _run(SearchPapers(), ctx, query="q")
    assert ads.calls[0] == ("nl", "q", 15, "date desc")


def test_search_papers_null_query_flows_into_the_legacy_body():
    # CX-01: explicit JSON null must reach the client call, not die in pydantic.
    ads = _FakeAds()
    ctx, _ = _ctx(ads_client=ads)
    out = _run(SearchPapers(), ctx, query=None)
    assert ads.calls[0][1] is None
    assert out["success"] is True  # the fake tolerates None like ADS would error


def test_search_papers_openalex_enrichment_and_trace():
    ads = _FakeAds()
    oalex = _FakeOalex(enrich={"10.1/a": {
        "fwci": 12.3, "citation_percentile": 99.9, "is_top_1_percent": True,
        "funders": ["NSF"], "oa_pdf_url": "http://oa/x.pdf", "topics": ["disks"],
    }})
    ctx, state = _ctx(ads_client=ads, openalex_client=oalex)
    out = _run(SearchPapers(), ctx, query="q")

    enriched = out["papers"][0]
    assert enriched["fwci"] == 12.3 and enriched["is_top_1_percent"] is True
    assert enriched["is_top_10_percent"] is False  # .get default, verbatim
    assert out["papers"][1].get("fwci") is None  # no doi → untouched
    assert state.console == ["[OpenAlex] Enriched 1/2 papers"]


def test_search_papers_enrichment_failure_is_nonfatal():
    ads = _FakeAds()
    oalex = _FakeOalex(raise_on_enrich=True)
    ctx, state = _ctx(ads_client=ads, openalex_client=oalex)
    out = _run(SearchPapers(), ctx, query="q")
    assert out["success"] is True
    assert state.console == ["[OpenAlex] Enrichment failed (non-fatal): openalex down"]


def test_search_papers_client_error_is_typed():
    class _Boom(_FakeAds):
        def search_natural_language(self, question, max_results, sort):
            raise RuntimeError("ADS 500")

    ctx, _ = _ctx(ads_client=_Boom())
    out = _run(SearchPapers(), ctx, query="q")
    assert out == {"success": False, "error": "ADS 500"}


# ─────────────────────────────────────────────────────────────────────────────
# search_papers_by_observation_id
# ─────────────────────────────────────────────────────────────────────────────
def test_obsid_gate_and_merge_dedup():
    ads = _FakeAds(id_type="mous_uid")
    df = pd.DataFrame([{"proposal_id": "2019.1.00123.S"}])

    class _Svc:
        def search_alma_with_keywords(self, kw):
            assert kw == {"member_ous_uid": "uid://A/X1/X1"}
            return df

    ctx, state = _ctx(ads_client=ads, search_service=_Svc())
    out = _run(SearchPapersByObservationId(), ctx, identifier="uid://A/X1/X1")

    assert out["success"] is True
    assert out["derived_identifiers"] == ["2019.1.00123.S"]
    # Two identifier searches ran (raw + derived) but papers dedup by bibcode:
    assert out["count"] == 2
    # observation_links from the second search merged onto the first copy:
    links = out["papers"][0]["observation_links"]
    assert links == ["link-uid://A/X1/X1", "link-2019.1.00123.S"]
    assert out["ads_query"] == "id:uid://A/X1/X1 OR id:2019.1.00123.S"
    prov = state.last_run_result["paper_provenance"]
    assert prov["derived_identifiers"] == ["2019.1.00123.S"]
    assert state.last_run_result["source"] == "ADS identifier: uid://A/X1/X1"


def test_obsid_lambda_gate_dict_is_verbatim():
    ctx, _ = _ctx(ads_client=None)
    out = _run(SearchPapersByObservationId(), ctx, identifier="x")
    assert out == {"error": "ADS client not configured"}


def test_obsid_derivation_failure_is_nonfatal():
    ads = _FakeAds(id_type="mous_uid")

    class _Svc:
        def search_alma_with_keywords(self, kw):
            raise RuntimeError("TAP down")

    ctx, state = _ctx(ads_client=ads, search_service=_Svc())
    out = _run(SearchPapersByObservationId(), ctx, identifier="uid://A/X1/X1")
    assert out["success"] is True and out["derived_identifiers"] == []
    assert state.console == [
        "[ADS identifier] Archive identifier derivation failed (non-fatal): TAP down"
    ]


def test_derive_helper_short_circuits():
    ads = _FakeAds(id_type="project_code")
    assert derive_archive_identifiers_for_paper_search(
        "2019.1.00123.S", ads_client=ads, search_service=object()) == []
    # No search service → no derivation at all (and no classify call).
    n_calls = len(ads.calls)
    assert derive_archive_identifiers_for_paper_search(
        "uid://A/X1/X1", ads_client=ads, search_service=None) == []
    assert len(ads.calls) == n_calls
    # Unknown identifier type → no keyword mapping → [].
    ads2 = _FakeAds(id_type="weird")
    assert derive_archive_identifiers_for_paper_search(
        "abc", ads_client=ads2, search_service=object()) == []


# ─────────────────────────────────────────────────────────────────────────────
# lookup_researcher / get_research_trends
# ─────────────────────────────────────────────────────────────────────────────
def test_lookup_researcher_orcid_path():
    oalex = _FakeOalex(author={"display_name": "A. Isella"})
    ctx, _ = _ctx(openalex_client=oalex)
    out = _run(LookupResearcher(), ctx, query="https://orcid.org/0000-0001-2345-6789")
    assert out == {"success": True, "count": 1, "researchers": [{"display_name": "A. Isella"}]}
    assert oalex.calls == [("get_author", "0000-0001-2345-6789")]


def test_lookup_researcher_orcid_not_found():
    ctx, _ = _ctx(openalex_client=_FakeOalex(author=None))
    out = _run(LookupResearcher(), ctx, query="0000-0001-2345-6789")
    assert out == {"success": False, "error": "No author found for ORCID 0000-0001-2345-6789"}


def test_lookup_researcher_name_path_and_empty():
    oalex = _FakeOalex(authors=[{"n": 1}, {"n": 2}])
    ctx, _ = _ctx(openalex_client=oalex)
    out = _run(LookupResearcher(), ctx, query="Crystal Brogan")
    assert out["count"] == 2
    assert oalex.calls == [("search_authors", "Crystal Brogan", 3)]

    ctx2, _ = _ctx(openalex_client=_FakeOalex(authors=[]))
    out2 = _run(LookupResearcher(), ctx2, query="Nobody")
    assert out2 == {"success": False, "error": "No researchers found matching 'Nobody'"}


def test_lookup_researcher_null_query_is_caught_like_legacy():
    # Legacy: query=None → .strip() AttributeError → typed error dict.
    ctx, _ = _ctx(openalex_client=_FakeOalex())
    out = _run(LookupResearcher(), ctx, query=None)
    assert out["success"] is False and "NoneType" in out["error"]


def test_get_research_trends_shapes_the_legacy_dict():
    oalex = _FakeOalex()
    ctx, _ = _ctx(openalex_client=oalex)
    out = _run(GetResearchTrends(), ctx, query="FRBs")
    assert out["success"] is True
    assert out["trends"] == {"total": 42, "by_year": {2020: 10}}
    assert out["funded_works_count"] == 12
    assert len(out["top_funded_works"]) == 8  # [:8], verbatim
    assert oalex.calls[0] == ("trends", "FRBs", 2015, 2026)


# ─────────────────────────────────────────────────────────────────────────────
# evaluate_consensus
# ─────────────────────────────────────────────────────────────────────────────
def test_consensus_requires_ads_client():
    ctx, _ = _ctx(ads_client=None)
    out = _run(EvaluateConsensus(), ctx, question="q")
    assert out == {"success": False, "error": "NASA ADS Client not initialized"}


def test_consensus_no_papers_is_typed():
    ads = _FakeAds(papers=[])
    ctx, _ = _ctx(ads_client=ads)
    out = _run(EvaluateConsensus(), ctx, question="q")
    assert out == {"success": False, "error": "No papers found for this question."}


def test_consensus_happy_path_uses_config_model_and_sets_card():
    ads = _FakeAds()
    llm = _FakeLLM(text="  ## Consensus  ")
    ctx, state = _ctx(ads_client=ads, llm_client=llm, agent_config=_FakeConfig())
    out = _run(EvaluateConsensus(), ctx, question="Do disks have rings?")

    assert out["success"] is True and out["papers_analyzed"] == 2
    assert out["analysis"] == "## Consensus"
    assert llm.kwargs["model"] == "test-model"
    assert llm.kwargs["temperature"] == 0.1 and llm.kwargs["max_output_tokens"] == 4000
    assert 'QUESTION: "Do disks have rings?"' in llm.kwargs["input"]
    assert "[1] Rings in HL Tau" in llm.kwargs["input"]
    # Sorted by citations, verbatim sort param:
    assert ads.calls[0] == ("nl", "Do disks have rings?", 20, "citation_count desc")
    assert state.last_run_result["type"] == "consensus"
    assert state.last_run_result["source"] == "Consensus Analysis: 2 papers"


# ─────────────────────────────────────────────────────────────────────────────
# extract_paper_details / reproduce_paper_methods
# ─────────────────────────────────────────────────────────────────────────────
def test_extract_details_arxiv_id_skips_ads_resolution():
    pdf = _FakePdf()
    ads = _FakeAds()
    ctx, state = _ctx(ads_client=ads, pdf_service=pdf)
    out = _run(ExtractPaperDetails(), ctx, identifier="1812.04040", query="beam size?")

    assert out == {"success": True, "identifier": "1812.04040", "extracted_answer": "0.035 arcsec"}
    assert pdf.calls == [("qa", "https://arxiv.org/pdf/1812.04040.pdf", "beam size?")]
    assert ("details", "1812.04040") not in ads.calls  # '.' in id → no ADS lookup
    assert state.last_run_result == {
        "type": "text", "text": "0.035 arcsec", "source": "Paper Extractor: 1812.04040"}


def test_extract_details_dotted_bibcode_skips_resolution_verbatim_quirk():
    # Preserved legacy quirk: the gate is `'.' not in id or len(id) > 20`, and a
    # standard 19-char bibcode CONTAINS dots — so it is never resolved via ADS
    # and goes straight (wrongly but verbatim) into the arxiv URL template.
    pdf = _FakePdf()
    ads = _FakeAds()
    ads.details = {"arxiv_id": "1812.04040"}
    ctx, _ = _ctx(ads_client=ads, pdf_service=pdf)
    out = _run(ExtractPaperDetails(), ctx, identifier="2018ApJ...869L..41A", query="q")
    assert out["success"] is True
    assert pdf.calls[0][1] == "https://arxiv.org/pdf/2018ApJ...869L..41A.pdf"
    assert ("details", "2018ApJ...869L..41A") not in ads.calls


def test_extract_details_dotless_identifier_resolves_via_ads():
    pdf = _FakePdf()
    ads = _FakeAds()
    ads.details = {"arxiv_id": "1812.04040"}
    ctx, _ = _ctx(ads_client=ads, pdf_service=pdf)
    out = _run(ExtractPaperDetails(), ctx, identifier="somebibcode", query="q")
    assert out["success"] is True
    assert ("details", "somebibcode") in ads.calls
    assert pdf.calls[0][1] == "https://arxiv.org/pdf/1812.04040.pdf"


def test_extract_details_doi_only_paper_is_typed():
    ads = _FakeAds()
    ads.details = {"doi": "10.3847/x"}
    ctx, _ = _ctx(ads_client=ads, pdf_service=_FakePdf())
    out = _run(ExtractPaperDetails(), ctx, identifier="somebibcode", query="q")
    assert out == {"success": False, "error": (
        "Paper has DOI (10.3847/x) but no arXiv ID. PDF download requires an open access arXiv ID.")}


def test_extract_details_pdf_failure_is_typed():
    pdf = _FakePdf(qa={"success": False, "error": "404"})
    ctx, _ = _ctx(ads_client=_FakeAds(), pdf_service=pdf)
    out = _run(ExtractPaperDetails(), ctx, identifier="1812.04040", query="q")
    assert out == {"success": False, "error": "Could not extract details: 404",
                   "identifier": "1812.04040"}


def test_reproduce_methods_generates_script_and_code_card():
    pdf = _FakePdf()
    llm = _FakeLLM(text="import casa\n")
    ctx, state = _ctx(ads_client=_FakeAds(), pdf_service=pdf,
                      llm_client=llm, agent_config=_FakeConfig())
    out = _run(ReproducePaperMethods(), ctx, identifier="1812.04040")

    assert out["success"] is True and out["generated_script"] == "import casa"
    assert out["methodology_summary"] == "CASA tclean, robust=0.5"  # <500 chars, no ellipsis
    assert llm.kwargs["model"] == "test-model" and llm.kwargs["temperature"] == 0.2
    assert "CASA tclean, robust=0.5" in llm.kwargs["input"]
    assert state.last_run_result == {
        "type": "code", "code": "import casa", "source": "Reproduce: 1812.04040"}


def test_reproduce_methods_truncates_long_methodology():
    pdf = _FakePdf(methodology={"success": True, "methodology": "x" * 501})
    ctx, _ = _ctx(ads_client=_FakeAds(), pdf_service=pdf,
                  llm_client=_FakeLLM(), agent_config=_FakeConfig())
    out = _run(ReproducePaperMethods(), ctx, identifier="1812.04040")
    assert out["methodology_summary"] == "x" * 500 + "..."


def test_reproduce_methods_pdf_failure_is_typed():
    pdf = _FakePdf(methodology={"success": False, "error": "no PDF"})
    ctx, _ = _ctx(ads_client=_FakeAds(), pdf_service=pdf)
    out = _run(ReproducePaperMethods(), ctx, identifier="1812.04040")
    assert out == {"success": False,
                   "error": "Could not extract methodology: no PDF",
                   "identifier": "1812.04040"}


# ─────────────────────────────────────────────────────────────────────────────
# agent wiring (real module surface)
# ─────────────────────────────────────────────────────────────────────────────
def _wiring_agent():
    from core.tools import ToolRegistry
    from tests.integration.test_agent_archive_tools import _load_agent_module

    module = _load_agent_module()
    agent = module.QuasarAgent.__new__(module.QuasarAgent)
    agent._tls = threading.local()
    agent.tool_registry = ToolRegistry()
    agent.last_search_results = None
    agent.last_run_result = None
    agent.ads_client = None
    agent.openalex_client = None
    return agent


def test_papers_registrations_keep_their_legacy_surface():
    agent = _wiring_agent()
    agent._register_tools()

    sp = agent.tool_registry.get_tool("search_papers")
    assert sp is not None and sp.category == "literature"
    assert sp.parameters["required"] == ["query"]
    assert "Do NOT try to construct ADS field syntax yourself" in sp.description

    obs = agent.tool_registry.get_tool("search_papers_by_observation_id")
    assert obs.parameters["required"] == ["identifier"]

    cons = agent.tool_registry.get_tool("evaluate_consensus")
    assert cons.parameters["required"] == ["question"]

    lr = agent.tool_registry.get_tool("lookup_researcher")
    assert "Powered by OpenAlex" in lr.description

    ep = agent.tool_registry.get_tool("extract_paper_details")
    assert ep.parameters["required"] == ["identifier", "query"]


def test_papers_ctx_provider_wires_the_agents_tls_state():
    agent = _wiring_agent()
    ctx = agent._papers_ctx_provider()
    ctx.service("set_last_run_result")({"type": "papers", "papers": []})
    assert agent.last_run_result == {"type": "papers", "papers": []}
    # Missing clients arrive as None (capabilities guard/except like legacy):
    assert ctx.services.get("ads_client") is None
    assert ctx.services.get("pdf_service") is None


def test_papers_tool_fn_end_to_end_through_the_agent():
    agent = _wiring_agent()
    agent.ads_client = _FakeAds()

    fn = agent._papers_tool_fn("search_papers", log_name="_search_papers")
    out = fn(query="rings")

    assert out["success"] is True and out["count"] == 2
    assert agent.last_run_result["type"] == "papers"


def test_papers_tool_fn_gate_without_client_matches_the_legacy_lambda():
    agent = _wiring_agent()
    fn = agent._papers_tool_fn("search_papers", log_name="_search_papers")
    assert fn(query="rings") == {"error": "ADS client not configured"}


def test_every_family_capability_is_registered():
    agent = _wiring_agent()
    agent._register_tools()
    for cap in CAPABILITIES:
        assert agent.tool_registry.get_tool(cap.name) is not None, cap.name


def test_unknown_capability_name_raises():
    agent = _wiring_agent()
    with pytest.raises(KeyError):
        agent._papers_tool_fn("not_a_tool")

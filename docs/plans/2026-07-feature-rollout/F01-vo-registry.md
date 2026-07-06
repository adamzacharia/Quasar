# F01 — VO Registry autonomous discovery (pyvo)

Read `docs/plans/2026-07-feature-rollout/CONVENTIONS.md` and the exemplars it
names BEFORE coding. Biggest strategic feature of the rollout: it turns
"supports N archives" into "can discover and query ~any VO service".
Executed LAST — treat every external interaction as potentially flaky.

## Objective

Four agent tools: (1) search the IVOA registry for services; (2) list/
inspect tables of a discovered TAP service; (3) run guarded ADQL against an
arbitrary TAP service; (4) cone-search an arbitrary SCS service.

## Files

- CREATE `services/vo_registry.py`
- CREATE `tests/unit/test_vo_registry.py`
- CREATE `scripts/smoke/smoke_f01_vo_registry.py`
- EDIT `core/agent.py` (4 anchors)

## Design

All pyvo imports LAZY. Reuse the timeout pattern from
`integrations/tap.py::_TimeoutHTTPSession` (copy the ~12-line class into the
new module rather than importing across layers): construct
`pyvo.dal.TAPService(url, session=_TimeoutHTTPSession(timeout))`.
Env: `VO_REGISTRY_TIMEOUT` (default 45 s), `VO_ADQL_MAXREC_CAP` (default
1000).

```python
class VoRegistryService:
    def __init__(self, *, timeout=None, maxrec_cap=None,
                 regsearch_fn=None, tap_factory=None, scs_factory=None):
        # regsearch_fn / tap_factory / scs_factory: injectables for tests

    def registry_search(self, keywords, service_type=None, waveband=None,
                        max_rows=30) -> dict:
        # VERIFIED LIVE 2026-07-04 on pyvo 1.8.1 — follow exactly:
        # - keyword-only: registry.search(keywords=<list>) works (GLEAM -> 23).
        # - WITH a service type you MUST use the constraint API AND include
        #   auxiliary capabilities, or VizieR-style records vanish:
        #     registry.search(registry.Freetext(*kw),
        #         registry.Servicetype(st).include_auxiliary_services())
        #   (plain servicetype='tap' returned 0 for GLEAM because VizieR
        #    catalogs are vs:catalogservice records whose TAP is tap#aux).
        # - service_type in {None,"tap","sia","ssa","scs"}; pass "scs" to
        #   pyvo as "conesearch" (normalize both aliases on input).
        # rows (defensive getattr, every field may be missing):
        #   {"ivoid", "short_name", "title" (res_title), "service_type"
        #    (res_type), "access_url", "waveband" (join list with ','),
        #    "description" (res_description truncated to 200 chars)}
        # access_url extraction (verified): prefer
        #   rec.get_service(service_type or "tap", lax=True).baseurl
        #   inside try/except; fall back to the first vs:paramhttp entry of
        #   rec.list_interfaces(); skip resources with no resolvable URL
        #   (count them in a warning)
        # clamp max_rows <= 100 (warn); dedupe by access_url

    def list_tables(self, access_url, keyword=None, max_tables=50) -> dict:
        # svc = tap_factory(access_url); tables = svc.tables
        # keyword: case-insensitive substring filter on table name +
        #   description (essential for VizieR's ~60k tables — add a warning
        #   recommending a keyword when truncation happens)
        # rows: {"table_name", "description" (trunc 150), "n_columns"
        #        (len(t.columns) guarded — some services lazy-load; use
        #        try/except -> None)}

    def describe_table(self, access_url, table_name) -> dict:
        # find table (exact then case-insensitive); rows per column:
        #   {"name", "datatype", "unit", "ucd", "description" (trunc 80)}
        # cap 200 columns with warning

    def run_adql(self, access_url, adql, max_rows=200) -> dict:
        # GUARDS (in order):
        #   1. strip leading whitespace/comments (`-- ...` lines and
        #      /* */ blocks) then require upper().startswith("SELECT")
        #      -> else {"success": False, "error": "Only SELECT queries
        #         are allowed."}
        #   2. maxrec = min(max_rows, maxrec_cap)
        # result = svc.run_sync(adql, maxrec=maxrec)
        # rows: list of dicts from result.to_table(); stringify values that
        #   are not (str,int,float,bool,None) — bytes -> utf-8/replace,
        #   masked -> None, numpy -> .item()
        # returns {"success", "rows", "count", "columns",
        #          "truncated": count == maxrec, "warnings", "provenance"
        #          (access_url + the adql)}
        # DALQueryError -> success False with the server's message text
        #   (that is what tells the model how to fix its ADQL)

    def cone_search(self, access_url, ra, dec, radius_deg,
                    max_rows=100) -> dict:
        # scs = scs_factory(access_url); scs.search(pos=(ra,dec),
        #   radius=radius_deg) -> normalize like run_adql; clamp radius<=5
```

## agent.py wiring

Tools (category `"archive"`), names chosen to read as a workflow:

1. `vo_find_services` — `_vo_find_services(keywords, service_type=None,
   waveband=None, max_rows=30)` (keywords required, string). Table columns
   `["short_name","title","service_type","waveband","access_url"]`,
   source `"IVOA Registry"`.
   Description: "Discover Virtual Observatory services (TAP/SIA/SSA/cone)
   by keyword and waveband — finds archives Quasar has no built-in client
   for. Follow with vo_list_tables / vo_adql_query on the access_url."
2. `vo_list_tables` — `_vo_list_tables(access_url, keyword=None,
   max_tables=50)`; table columns `["table_name","n_columns",
   "description"]`, source access_url.
   Description: "List (and filter) the tables of any TAP service found via
   vo_find_services."
3. `vo_describe_table` — `_vo_describe_table(access_url, table_name)`;
   table columns `["name","datatype","unit","ucd","description"]`.
   Description: "Column schema of a table on any TAP service — call before
   writing ADQL."
4. `vo_adql_query` — `_vo_adql_query(access_url, adql, max_rows=200)`;
   table card from returned rows (columns = result columns, cap the column
   list at 12 for display with a warning), source `f"TAP: {access_url}"`,
   filter_label = first 120 chars of the ADQL.
   Description: "Run a guarded SELECT-only ADQL query against any TAP
   service URL. On ADQL errors the server message is returned — read it and
   fix the query."
5. `vo_cone_search` — `_vo_cone_search(access_url, target_name=None,
   ra=None, dec=None, radius_deg=0.1, max_rows=100)`.
   Description: "Cone search any VO simple-cone-search service by position."

Status labels: `"vo_find_services": "Searching the IVOA registry"`,
`"vo_list_tables": "Listing TAP service tables"`,
`"vo_describe_table": "Inspecting table schema"`,
`"vo_adql_query": "Running ADQL on remote TAP service"`,
`"vo_cone_search": "Running VO cone search"`.

Prompt bullet: `When no built-in tool covers an archive/dataset, use the VO
chain: \`vo_find_services\` -> \`vo_list_tables\` -> \`vo_describe_table\`
-> \`vo_adql_query\` (SELECT-only). Always inspect the schema before
writing ADQL, and quote table names that contain '/' or '.' in double
quotes.`

## Unit tests (offline — inject all three factories/fns)

1. registry_search: fake resources (one missing access_url, one duplicate
   URL, one full) → dedupe + skip-count warning + row schema; max_rows
   clamp; keywords as str coerced to list.
2. run_adql guards: "DROP TABLE x" rejected; "  -- comment\nselect top 5.."
   accepted (comment stripping); maxrec = min(max_rows, cap) passed to
   run_sync (assert via fake); truncated flag when count==maxrec.
3. run_adql normalization: fake result with bytes, numpy int64, masked
   value → str/int/None.
4. DALQueryError-ish exception (any Exception with server text) → success
   False, text preserved.
5. list_tables: 60k-name fake (generate 200) with keyword filter +
   truncation warning; describe_table case-insensitive match + 200-col cap.
6. cone_search radius clamp + normalization.

## Smoke expectations

- `registry_search("GLEAM", service_type="tap")` → ≥1 row with an access
  URL containing "vizier" or "casda" or "mwa" (verified live 2026-07-04:
  22 rows via Servicetype('tap').include_auxiliary_services(); the first
  resolves to https://tapvizier.cds.unistra.fr/TAPVizieR/tap).
- `run_adql("https://tapvizier.cds.unistra.fr/TAPVizieR/tap",
  'SELECT TOP 5 * FROM "VIII/100/gleamegc"')` → 5 rows.
- `run_adql(same, "DELETE FROM x")` → success False mentioning SELECT.
- `list_tables("https://tapvizier.cds.unistra.fr/TAPVizieR/tap",
  keyword="nvss", max_tables=20)` → ≥1, truncation warning acceptable.
Exit 0 only if all hold.

## Chat acceptance question

"Find me a VO service with HI 21-cm survey data and pull 5 example rows
from its main table." (Agent should chain all four tools.)

## Acceptance checklist

- [ ] CONVENTIONS definition-of-done ticked
- [ ] Unit tests green; smoke passes; wiring complete (5 tools)
- [ ] SELECT-only guard verified by test AND smoke

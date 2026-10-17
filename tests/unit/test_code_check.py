"""Run-once code check (core/code_check.py, ALMAbench round 5)."""
from __future__ import annotations

import json
import os

import pytest

from core import code_check as cc
from core.code_check import RunResult, check_answer_code, find_archive_script, run_script

@pytest.fixture(autouse=True)
def _check_on(monkeypatch):
    monkeypatch.setenv("QUASAR_CODE_CHECK", "1")


def test_off_by_default(monkeypatch):
    monkeypatch.delenv("QUASAR_CODE_CHECK", raising=False)
    assert check_answer_code(ANSWER, model="m", runner=_Runner()).status == "skipped"


SCRIPT = """import pyvo
service = pyvo.dal.TAPService("https://almascience.eso.org/tap")
result = service.search("SELECT TOP 5 obs_id FROM ivoa.obscore WHERE band_list = '6'")
print(result.to_table())"""

ANSWER = f"Here is a script.\n\n```python\n{SCRIPT}\n```\n\nIt lists the rows."


def _ok(out="obs_id\nuid://A001/X1/X2\n"):
    return RunResult(True, 0, out, "")


def _fail(err="Traceback (most recent call last):\n  File x\npyvo.dal.exceptions.DALQueryError: Unknown column 'bandlist'"):
    return RunResult(False, 1, "", err)


class _Runner:
    def __init__(self, *results):
        self.results = list(results)
        self.codes = []

    def __call__(self, code, timeout):
        self.codes.append(code)
        return self.results.pop(0)


def test_finds_archive_script_and_its_span():
    block = find_archive_script(ANSWER)
    assert block is not None
    assert ANSWER[block.start:block.end] == SCRIPT


@pytest.mark.parametrize("extra", [
    "from casatasks import tclean\n",
    "Alma.login('someone')\n",
    "pw = input('password')\n",
    "!pip install pyvo\n",
    "user = YOUR_USERNAME\n",
    "path = '<path to your file>'\n",
])
def test_skips_scripts_that_cannot_run_unattended(extra):
    assert find_archive_script(f"```python\n{extra}{SCRIPT}\n```") is None


def test_skips_non_archive_and_non_python_blocks():
    assert find_archive_script("```python\nimport numpy as np\nx = np.arange(3)\nprint(x)\n```") is None
    assert find_archive_script(f"```sql\n{SCRIPT}\n```") is None


def test_skips_scripts_with_modules_not_installed():
    code = "import pyvo\nimport surely_not_a_real_module_xyz\nprint(pyvo)\nprint(1)"
    assert find_archive_script(f"```python\n{code}\n```") is None


def test_comparison_operators_are_not_placeholders():
    code = SCRIPT + "\nn = len(result)\nif n < 3 and n > 1:\n    print('few')"
    assert find_archive_script(f"```python\n{code}\n```") is not None


def test_passing_script_gets_note_and_output_excerpt():
    out = check_answer_code(ANSWER, model="m", runner=_Runner(_ok()), today="2026-10-16")
    assert out.status == "passed"
    assert SCRIPT in out.text
    assert "**Code check:**" in out.text and "2026-10-16" in out.text
    assert "uid://A001/X1/X2" in out.text.split("**Code check:**")[1]
    assert out.note and out.text.endswith(out.note)


def test_failed_script_is_repaired_once_and_replaced():
    fixed = SCRIPT.replace("band_list = '6'", "band_list LIKE '%6%'")
    runner = _Runner(_fail(), _ok())
    calls = []

    def repair(prompt, model, max_tokens):
        calls.append((prompt, model))
        return f"```python\n{fixed}\n```"

    out = check_answer_code(ANSWER, question="list band 6 rows", model="arm-model", runner=runner,
                            repair_call=repair, today="2026-10-16")
    assert out.status == "repaired"
    assert fixed in out.text and SCRIPT not in out.text
    assert len(calls) == 1 and calls[0][1] == "arm-model"
    assert "Unknown column" in calls[0][0] and "list band 6 rows" in calls[0][0]
    assert runner.codes == [SCRIPT, fixed]
    assert "first version failed" in out.text


def test_repair_that_still_fails_keeps_original_and_says_so():
    runner = _Runner(_fail(), _fail("Traceback (most recent call last):\nValueError: still bad"))
    out = check_answer_code(ANSWER, model="m", runner=runner,
                            repair_call=lambda p, m, t: "```python\nimport pyvo\nprint(1)\nprint(2)\n```")
    assert out.status == "failed"
    assert SCRIPT in out.text
    assert "did not run either" in out.text
    assert "DALQueryError" in out.text


def test_no_model_means_no_repair_and_no_false_claim():
    out = check_answer_code(ANSWER, model="", runner=_Runner(_fail()))
    assert out.status == "failed"
    assert "did not run either" not in out.text


def _fence(code):
    return f"```python\n{code}\n```"


def test_timeout_gets_one_repair_and_honest_note():
    runner = _Runner(RunResult(False, None, "", "", timed_out=True), RunResult(False, None, "", "", timed_out=True))
    fixed = SCRIPT.replace("TOP 5", "TOP 1")
    out = check_answer_code(ANSWER, model="m", runner=runner, repair_call=lambda *a: _fence(fixed))
    assert out.status == "timed_out" and "not verified" in out.text and "did not finish either" in out.text
    assert SCRIPT in out.text


def test_placeholder_in_docstring_or_comment_does_not_skip():
    code = '"""Author: <your name>"""\n' + SCRIPT + "  # replace <your path> if needed"
    assert find_archive_script(_fence(code)) is not None


UIDS = [f"uid://A001/X1/X{i:x}" for i in range(10, 20)]
WRONG = "\n".join(f"uid://A002/X9/X{i:x}" for i in range(3))


def test_uid_mismatch_triggers_repair_and_keeps_the_closer_script():
    ref = json.dumps({"rows": [{"member_ous_uid": u} for u in UIDS]})
    fixed = SCRIPT.replace("TOP 5", "TOP 50")
    prompts = []
    runner = _Runner(_ok(WRONG), _ok("\n".join(UIDS)))
    out = check_answer_code(ANSWER, model="m", runner=runner, reference_text=ref,
                            repair_call=lambda p, m, t: prompts.append(p) or _fence(fixed))
    assert out.status == "repaired" and fixed in out.text
    assert UIDS[0] in prompts[0] and "differ" in prompts[0]
    assert "differ from the archive results" in out.text


def test_uid_mismatch_unresolved_is_reported():
    out = check_answer_code(ANSWER, model="m", runner=_Runner(_ok(WRONG)), reference_text=json.dumps(UIDS),
                            repair_call=lambda p, m, t: _fence(SCRIPT))
    assert out.status == "mismatch" and "compare the two" in out.text


def test_truncated_table_print_is_not_a_mismatch():
    shown = "\n".join(UIDS[:2]) + "\n...\nLength = 10 rows"
    out = check_answer_code(ANSWER, model="m", runner=_Runner(_ok(shown)), reference_text=json.dumps(UIDS))
    assert out.status == "passed"


def test_matching_uids_pass():
    out = check_answer_code(ANSWER, model="m", runner=_Runner(_ok("\n".join(UIDS))), reference_text=json.dumps(UIDS))
    assert out.status == "passed"


def test_empty_result_offers_repair_but_unchanged_reply_keeps_it():
    runner = _Runner(_ok("<Table length=0>\n"))
    out = check_answer_code(ANSWER, model="m", runner=runner, repair_call=lambda p, m, t: f"```python\n{SCRIPT}\n```")
    assert out.status == "empty"
    assert out.info.get("repair") == "unchanged"
    assert "empty result" in out.text


def test_disabled_flag_and_short_deadline_skip(monkeypatch):
    monkeypatch.setenv("QUASAR_CODE_CHECK", "0")
    assert check_answer_code(ANSWER, model="m", runner=_Runner()).text == ANSWER
    monkeypatch.setenv("QUASAR_CODE_CHECK", "1")
    import time

    out = check_answer_code(ANSWER, model="m", runner=_Runner(), deadline=time.monotonic() + 25)
    assert out.status == "skipped" and out.text == ANSWER


def test_repair_reply_with_skip_markers_is_rejected():
    runner = _Runner(_fail())
    out = check_answer_code(ANSWER, model="m", runner=runner,
                            repair_call=lambda p, m, t: f"```python\nimport subprocess\nAlma.login('x')\n{SCRIPT}\n```")
    assert out.status == "failed" and runner.codes == [SCRIPT]


# ---- the real child interpreter (no network needed: blocked hosts are refused before any lookup)

def test_sandbox_runs_plain_code():
    r = run_script("import numpy as np\nprint(np.arange(3).sum())", timeout=60)
    assert r.ok, r.stderr
    assert r.stdout.strip() == "3"


def test_sandbox_environment_has_no_secrets(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    r = run_script("import os\nprint(sorted(os.environ))", timeout=60)
    assert r.ok, r.stderr
    assert "OPENAI_API_KEY" not in r.stdout


def test_sandbox_blocks_other_hosts():
    r = run_script("import urllib.request\nurllib.request.urlopen('https://example.com', timeout=5)", timeout=60)
    assert not r.ok and "not allowed" in r.stderr


def test_sandbox_blocks_reads_and_writes_outside(tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("hidden", encoding="utf-8")
    r = run_script(f"print(open(r'{secret}').read())", timeout=60)
    assert not r.ok and "reading" in r.stderr and "hidden" not in r.stdout
    target = tmp_path / "out.txt"
    r = run_script(f"open(r'{target}', 'w').write('x')", timeout=60)
    assert not r.ok and not target.exists()
    r = run_script("open('local.csv', 'w').write('a,b')\nprint(open('local.csv').read())", timeout=60)
    assert r.ok and "a,b" in r.stdout


def test_sandbox_blocks_processes():
    r = run_script("import subprocess\nsubprocess.run(['python', '-c', 'print(1)'])", timeout=60)
    assert not r.ok and "subprocess.Popen" in r.stderr
    r = run_script("import os\nos.system('echo hi')", timeout=60)
    assert not r.ok and "os.system" in r.stderr


def test_sandbox_timeout():
    r = run_script("import time\ntime.sleep(30)", timeout=3)
    assert r.timed_out and not r.ok


def test_error_line_picks_the_exception():
    r = RunResult(False, 1, "", "Traceback (most recent call last):\n  File \"a.py\", line 1\nKeyError: 'x'\n")
    assert r.error_line() == "KeyError: 'x'"


def test_repair_call_error_does_not_claim_a_second_run():
    def boom(prompt, model, max_tokens):
        raise ValueError("no key")

    out = check_answer_code(ANSWER, model="m", runner=_Runner(_fail()), repair_call=boom)
    assert out.status == "failed"
    assert "did not run either" not in out.text
    assert "no key" in out.info["repair_error"]


def test_attribute_error_lists_real_attributes():
    r = run_script("import pyvo\nsvc = pyvo.dal.TAPService('https://almascience.eso.org/tap')\nsvc.no_such_method()", timeout=60)
    assert not r.ok
    assert "[code check] public attributes of TAPService:" in r.stderr and "run_sync" in r.stderr
    assert r.error_line().startswith("AttributeError")


def test_repair_prompt_names_installed_versions():
    from core.code_check import build_repair_prompt

    p = build_repair_prompt("q", SCRIPT, _fail(), "2026-10-16")
    assert "Installed: " in p and "pyvo " in p


def test_untagged_adql_block_is_not_treated_as_python():
    adql = "SELECT DISTINCT proposal_id FROM ivoa.obscore\nWHERE scientific_category = 'Sun'\n  AND science_observation = 'T'"
    assert find_archive_script(f"Query:\n\n```\n{adql}\n```\n") is None
    assert find_archive_script(f"```\n{SCRIPT}\n```") is not None  # untagged but real Python with imports


def test_output_excerpt_that_trips_the_safety_guard_is_left_out(monkeypatch):
    from services import content_safety

    monkeypatch.setattr(content_safety, "strict_web_filter_enabled", lambda: True)
    out = check_answer_code(ANSWER, model="m", runner=_Runner(_ok("name\nXXX\n")))
    assert out.status == "passed" and "**Code check:**" in out.text
    assert "XXX" not in out.text and not content_safety.looks_explicit_text(out.text)

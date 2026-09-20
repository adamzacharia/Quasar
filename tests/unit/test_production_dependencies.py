"""Static contract tests for the production dependency deployment."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_INPUT = ROOT / "requirements" / "production.in"
OPTIONAL_SPARCL_INPUT = ROOT / "requirements" / "optional-sparcl.in"
PRODUCTION_LOCK = (
    ROOT / "requirements" / "production-py312-linux-x86_64.txt"
)
ANSIBLE_PLAYBOOK = ROOT / "deploy" / "ansible" / "quasar.yml"
ANSIBLE_README = ROOT / "deploy" / "ansible" / "README.md"


def _active_lines(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _requirement_map(path: Path) -> dict[str, str]:
    requirements: dict[str, str] = {}
    for line in _active_lines(path):
        if line.startswith(("-", "--")):
            continue
        match = re.match(r"^([A-Za-z0-9_.-]+)(.*)$", line)
        assert match, f"Unparseable requirement in {path}: {line}"
        name = match.group(1).lower().replace("_", "-").replace(".", "-")
        requirements[name] = match.group(2).strip()
    return requirements


def _locked_requirement_blocks(text: str) -> list[list[str]]:
    blocks: list[list[str]] = []
    current: list[str] = []
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        is_top_level = raw_line == raw_line.lstrip()
        if is_top_level and stripped.startswith("--"):
            continue
        if is_top_level and re.match(r"^[A-Za-z0-9_.-]+==", stripped):
            if current:
                blocks.append(current)
            current = [stripped]
        elif is_top_level:
            raise AssertionError(f"Lock entry is not exact: {stripped}")
        elif current:
            current.append(stripped)
        else:
            raise AssertionError(f"Orphaned lock continuation: {stripped}")
    if current:
        blocks.append(current)
    return blocks


def test_locked_requirement_parser_retains_hashes_and_rejects_ranges() -> None:
    digest_a = "a" * 64
    digest_b = "b" * 64
    sample = (
        "--only-binary :all:\n"
        "alpha==1.0 \\\n"
        f"    --hash=sha256:{digest_a}\n"
        "beta==2.0 \\\n"
        f"    --hash=sha256:{digest_b}\n"
    )
    blocks = _locked_requirement_blocks(sample)
    assert blocks == [
        ["alpha==1.0 \\", f"--hash=sha256:{digest_a}"],
        ["beta==2.0 \\", f"--hash=sha256:{digest_b}"],
    ]

    try:
        _locked_requirement_blocks("ranged-package>=1.0\n")
    except AssertionError as exc:
        assert "not exact" in str(exc)
    else:
        raise AssertionError("A ranged top-level lock entry was accepted")


def test_default_requirements_delegate_to_production_input() -> None:
    assert _active_lines(ROOT / "requirements.txt") == [
        "-r requirements/production.in"
    ]


def test_production_and_optional_inputs_are_deliberately_separated() -> None:
    production = _requirement_map(PRODUCTION_INPUT)
    optional = _requirement_map(OPTIONAL_SPARCL_INPUT)

    assert "sparclclient" not in production
    assert production["pandas"] == "==2.3.3"
    assert production["lsdb"] == "==0.9.2"
    assert production["alminer"] == "==0.1.2"
    assert production["pyvo"] == ">=1.8"
    assert production["mcp"] == ">=1.8,<2"

    assert optional == {
        "sparclclient": '==1.3.0; python_version >= "3.13"'
    }


def test_production_lock_is_complete_exact_and_hashed() -> None:
    assert PRODUCTION_LOCK.exists(), (
        "Generate the production lock with the exact uv command documented in "
        "deploy/ansible/README.md"
    )
    text = PRODUCTION_LOCK.read_text(encoding="utf-8")
    blocks = _locked_requirement_blocks(text)
    assert blocks, "Production lock contains no exact requirements"

    locked: dict[str, str] = {}
    for block in blocks:
        requirement = block[0].removesuffix("\\").strip()
        match = re.match(
            r"^([A-Za-z0-9_.-]+)==([^;\\\s]+)(?:\s*;.*)?$", requirement
        )
        assert match, f"Lock entry is not exact: {requirement}"
        assert any(
            re.search(r"--hash=sha256:[0-9a-f]{64}(?:\s|\\|$)", line)
            for line in block[1:]
        ), f"Lock entry has no SHA-256 hash: {requirement}"
        name = match.group(1).lower().replace("_", "-").replace(".", "-")
        locked[name] = match.group(2)

    assert locked["openai"] == "2.41.1"
    assert locked["langchain"] == "0.3.30"
    assert locked["langchain-community"] == "0.3.31"
    assert locked["langchain-openai"] == "0.3.35"
    assert locked["qdrant-client"] == "1.18.0"
    assert locked["pandas"] == "2.3.3"
    assert locked["lsdb"] == "0.9.2"
    assert locked["alminer"] == "0.1.2"
    assert "mcp" in locked
    assert "fastapi" in locked
    assert "uvicorn" in locked
    assert "sparclclient" not in locked


def test_ansible_installs_and_validates_only_the_hashed_lock() -> None:
    playbook = ANSIBLE_PLAYBOOK.read_text(encoding="utf-8")

    assert (
        'quasar_production_lock: "{{ quasar_app_dir }}/requirements/'
        'production-py312-linux-x86_64.txt"' in playbook
    )
    assert 'requirements: "{{ quasar_production_lock }}"' in playbook
    assert 'requirements: "{{ quasar_app_dir }}/requirements.txt"' not in playbook
    assert "--require-hashes --only-binary=:all:" in playbook
    assert "py312-{{ quasar_lock_stat.stat.checksum }}" in playbook
    assert ".quasar-lock-ok" in playbook
    assert "Remove an interrupted unactivated release" in playbook
    assert "Activate validated venv symlink atomically" in playbook
    assert re.search(r"- mv\n\s+- -Tf", playbook)
    assert "Check locked dependency consistency" in playbook
    assert "from pyvo.registry import Freetext" in playbook
    assert 'importlib.util.find_spec("sparcl") is None' in playbook
    assert "from api.main import app" in playbook
    assert "assert agent is not None, get_agent_error()" in playbook
    assert "rescue:" in playbook
    assert "Restore previous validated venv symlink atomically" in playbook
    assert "Restore preserved legacy venv directory" in playbook
    assert (
        'url: "http://127.0.0.1:{{ quasar_backend_port }}/health"' in playbook
    )
    assert ".get('agent_loaded', false)" in playbook


def test_documented_compile_command_is_targeted_and_reproducible() -> None:
    readme = ANSIBLE_README.read_text(encoding="utf-8")
    for flag in (
        "--python-version 3.12",
        "--python-platform x86_64-manylinux_2_28",
        "--only-binary :all:",
        "--emit-build-options",
        "--generate-hashes",
        "--exclude-newer 2026-09-02T00:00:00Z",
        "--upgrade",
        "--no-cache",
        "--no-python-downloads",
        "--output-file requirements/production-py312-linux-x86_64.txt",
    ):
        assert flag in readme

"""Saved papers are per-user server state.

They used to live only in the browser's memory, so every reload (and so every
redeploy) emptied the Saved Papers panel. These tests pin the properties the
fix depends on: a bookmark survives a fresh service instance (a restart), users
never see each other's bookmarks, and DOI keys with "/" round-trip.
"""

import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (REPO, os.path.join(REPO, "ui-pro")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from services import saved_papers_service as sp_module  # noqa: E402
from services.saved_papers_service import SavedPaperError, SavedPapersService  # noqa: E402

PAPER = {"title": "A radio survey", "authors": "Doe, J.", "year": 2024, "bibcode": "2024ApJ...1A"}


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "saved.db")


def test_bookmark_survives_a_restart(db_path):
    SavedPapersService(db_path).save_paper("alice", "2024ApJ...1A", PAPER)
    # A new instance on the same database is what a redeploy looks like.
    papers = SavedPapersService(db_path).list_papers("alice")
    assert papers == [{**PAPER, "id": "2024ApJ...1A"}]


def test_users_are_isolated(db_path):
    svc = SavedPapersService(db_path)
    svc.save_paper("alice", "k1", PAPER)
    assert svc.list_papers("bob") == []
    svc.remove_paper("bob", "k1")  # bob cannot delete alice's bookmark
    assert len(svc.list_papers("alice")) == 1


def test_resave_is_idempotent_and_keeps_order(db_path):
    svc = SavedPapersService(db_path)
    svc.save_paper("alice", "first", {**PAPER, "title": "First"})
    svc.save_paper("alice", "second", {**PAPER, "title": "Second"})
    svc.save_paper("alice", "first", {**PAPER, "title": "First, updated"})
    papers = svc.list_papers("alice")
    assert [p["id"] for p in papers] == ["first", "second"]
    assert papers[0]["title"] == "First, updated"


def test_remove(db_path):
    svc = SavedPapersService(db_path)
    svc.save_paper("alice", "k1", PAPER)
    svc.remove_paper("alice", "k1")
    svc.remove_paper("alice", "k1")  # removing twice is not an error
    assert svc.list_papers("alice") == []


def test_rejects_bad_input(db_path, monkeypatch):
    svc = SavedPapersService(db_path)
    with pytest.raises(SavedPaperError):
        svc.save_paper("alice", "  ", PAPER)
    with pytest.raises(SavedPaperError):
        svc.save_paper("alice", "k", {"abstract": "x" * (sp_module.MAX_PAPER_JSON_BYTES + 1)})
    monkeypatch.setattr(sp_module, "MAX_SAVED_PAPERS_PER_USER", 1)
    svc.save_paper("alice", "a", PAPER)
    with pytest.raises(SavedPaperError):
        svc.save_paper("alice", "b", PAPER)
    svc.save_paper("alice", "a", PAPER)  # updating an existing one is still fine at the cap


@pytest.fixture
def api(db_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from api.deps import get_current_user
    from api.routers import saved_papers

    monkeypatch.setattr(saved_papers, "saved_papers_service", SavedPapersService(db_path))
    app = FastAPI()
    app.include_router(saved_papers.router)
    user = {"sub": "alice"}
    app.dependency_overrides[get_current_user] = lambda: user
    client = TestClient(app)
    client.user = user
    return client


def test_routes_round_trip_a_doi_key(api):
    key = "doi:10.3847/1538-4357/ab1234"
    assert api.put("/api/saved-papers", json={"key": key, "paper": PAPER}).status_code == 200
    assert [p["id"] for p in api.get("/api/saved-papers").json()["papers"]] == [key]

    api.user["sub"] = "bob"
    assert api.get("/api/saved-papers").json()["papers"] == []
    api.user["sub"] = "alice"

    assert api.delete("/api/saved-papers", params={"key": key}).status_code == 200
    assert api.get("/api/saved-papers").json()["papers"] == []


def test_routes_reject_bad_input(api):
    assert api.put("/api/saved-papers", json={"key": " ", "paper": PAPER}).status_code == 400
    assert api.put("/api/saved-papers", json={"key": "k", "paper": "nope"}).status_code == 422
    assert api.delete("/api/saved-papers").status_code == 422

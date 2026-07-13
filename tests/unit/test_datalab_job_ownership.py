from __future__ import annotations

import asyncio
import sys
import threading
from pathlib import Path

import pandas as pd
import pytest
from fastapi import HTTPException

from capabilities import datalab as datalab_capability
from capabilities.base import CallContext
from integrations.datalab_client import ANON_TOKEN
import integrations.datalab_client as datalab_client_module
import services.datalab_job_service as job_service_module
import services.datalab_result_store as result_store_module
from services.datalab_job_service import DatalabJobService
from services.datalab_result_store import DatalabResultStore


UI_PRO = Path(__file__).resolve().parents[2] / "ui-pro"
if str(UI_PRO) not in sys.path:
    sys.path.insert(0, str(UI_PRO))

from api.routers import datalab as datalab_router  # noqa: E402


def test_job_service_scopes_records_when_owner_is_provided():
    service = DatalabJobService(max_workers=1)
    release = threading.Event()
    alice_local = service.start(
        "scan",
        lambda cancel_check: release.wait(timeout=2),
        owner_id="alice",
    )
    service.register_external("alice-remote", owner_id="alice")
    service.register_external("bob-remote", owner_id="bob")

    assert {record["job_id"] for record in service.list_jobs(owner_id="alice")} == {
        alice_local,
        "alice-remote",
    }
    assert "_owner_id" not in service.status("alice-remote", owner_id="alice")
    with pytest.raises(KeyError, match="Unknown Data Lab job"):
        service.status("alice-remote", owner_id="bob")
    with pytest.raises(KeyError, match="Unknown Data Lab job"):
        service.cancel("alice-remote", owner_id="bob")
    with pytest.raises(KeyError, match="Unknown Data Lab job"):
        service.update_external("alice-remote", owner_id="bob", status="failed")
    assert service.status("alice-remote", owner_id="alice")["status"] == "submitted"

    # Existing capability callers omit owner_id and retain their pre-SX behavior.
    assert {record["job_id"] for record in service.list_jobs()} >= {
        alice_local,
        "alice-remote",
        "bob-remote",
    }
    release.set()


def test_job_routes_scope_list_status_and_cancel_as_unknown(monkeypatch):
    service = DatalabJobService(max_workers=1)
    service.register_external("shared-id", owner_id="alice", status="succeeded")
    service.register_external("bob-id", owner_id="bob", status="succeeded")
    monkeypatch.setattr(job_service_module, "_DEFAULT_JOB_SERVICE", service)

    listed = asyncio.run(datalab_router.list_datalab_jobs(current_user={"sub": "alice"}))
    assert [record["job_id"] for record in listed["jobs"]] == ["shared-id"]
    assert asyncio.run(
        datalab_router.get_datalab_job("shared-id", current_user={"sub": "alice"})
    )["job_id"] == "shared-id"

    with pytest.raises(HTTPException) as cross_status:
        asyncio.run(
            datalab_router.get_datalab_job("shared-id", current_user={"sub": "bob"})
        )
    with pytest.raises(HTTPException) as cross_cancel:
        asyncio.run(
            datalab_router.cancel_datalab_job("shared-id", current_user={"sub": "bob"})
        )

    monkeypatch.setattr(
        job_service_module,
        "_DEFAULT_JOB_SERVICE",
        DatalabJobService(max_workers=1),
    )
    with pytest.raises(HTTPException) as unknown_status:
        asyncio.run(
            datalab_router.get_datalab_job("shared-id", current_user={"sub": "bob"})
        )
    with pytest.raises(HTTPException) as unknown_cancel:
        asyncio.run(
            datalab_router.cancel_datalab_job("shared-id", current_user={"sub": "bob"})
        )

    assert (cross_status.value.status_code, cross_status.value.detail) == (
        unknown_status.value.status_code,
        unknown_status.value.detail,
    )
    assert (cross_cancel.value.status_code, cross_cancel.value.detail) == (
        unknown_cancel.value.status_code,
        unknown_cancel.value.detail,
    )
    assert cross_status.value.status_code == cross_cancel.value.status_code == 404


def test_job_capability_enforces_call_context_owner():
    service = DatalabJobService(max_workers=1)
    job_id = service.start("scan", lambda cancel_check: {}, owner_id="alice")
    inp = datalab_capability.JobIdInput(job_id=job_id)

    allowed = datalab_capability.JobStatus().run(
        inp,
        CallContext(
            services={"datalab_job_service": service},
            user_id="alice",
        ),
    ).to_native()
    denied = datalab_capability.JobStatus().run(
        inp,
        CallContext(
            services={"datalab_job_service": service},
            user_id="bob",
        ),
    ).to_native()

    assert allowed["success"] is True
    assert denied["success"] is False
    assert "Unknown Data Lab job" in denied["error"]


def test_external_cancel_updates_only_after_remote_success_and_surfaces_errors(monkeypatch):
    service = DatalabJobService(max_workers=1)
    service.register_external("remote-id", owner_id="alice")
    monkeypatch.setattr(job_service_module, "_DEFAULT_JOB_SERVICE", service)

    class FailingClient:
        token = "real-token"

        def abort(self, job_id):
            assert service.status(job_id, owner_id="alice")["status"] == "submitted"
            raise RuntimeError("remote abort failed")

    monkeypatch.setattr(datalab_client_module, "DatalabClient", FailingClient)
    with pytest.raises(HTTPException) as failed:
        asyncio.run(
            datalab_router.cancel_datalab_job("remote-id", current_user={"sub": "alice"})
        )
    assert failed.value.status_code == 502
    assert "remote abort failed" in failed.value.detail
    assert service.status("remote-id", owner_id="alice")["status"] == "submitted"

    aborted = []

    class SuccessfulClient:
        token = "real-token"

        def abort(self, job_id):
            assert service.status(job_id, owner_id="alice")["status"] == "submitted"
            aborted.append(job_id)

    monkeypatch.setattr(datalab_client_module, "DatalabClient", SuccessfulClient)
    record = asyncio.run(
        datalab_router.cancel_datalab_job("remote-id", current_user={"sub": "alice"})
    )
    assert aborted == ["remote-id"]
    assert record["status"] == "canceled"

    service.register_external("anon-id", owner_id="alice")

    class AnonymousClient:
        token = ANON_TOKEN

    monkeypatch.setattr(datalab_client_module, "DatalabClient", AnonymousClient)
    with pytest.raises(HTTPException) as unavailable:
        asyncio.run(
            datalab_router.cancel_datalab_job("anon-id", current_user={"sub": "alice"})
        )
    assert unavailable.value.status_code == 503
    assert service.status("anon-id", owner_id="alice")["status"] == "submitted"


def test_mytables_routes_isolate_users_and_return_strict_json(monkeypatch):
    store = DatalabResultStore(enable_disk_cache=False)
    alice_result = store.put(pd.DataFrame({"value": [1.0, float("nan"), float("inf")]}))
    bob_result = store.put(pd.DataFrame({"value": [2.0]}))
    monkeypatch.setattr(result_store_module, "_DEFAULT_STORE", store)

    asyncio.run(
        datalab_router.save_my_table(
            datalab_router.SaveMyTableRequest(result_id=alice_result, name="shared"),
            current_user={"sub": "alice"},
        )
    )
    asyncio.run(
        datalab_router.save_my_table(
            datalab_router.SaveMyTableRequest(result_id=bob_result, name="shared"),
            current_user={"sub": "bob"},
        )
    )

    alice = asyncio.run(
        datalab_router.get_my_table("shared", current_user={"sub": "alice"})
    )
    bob = asyncio.run(
        datalab_router.get_my_table("shared", current_user={"sub": "bob"})
    )
    assert alice["rows"] == [{"value": 1.0}, {"value": None}, {"value": None}]
    assert bob["rows"] == [{"value": 2.0}]
    assert asyncio.run(
        datalab_router.list_my_tables(current_user={"sub": "alice"})
    )["count"] == 1

    asyncio.run(
        datalab_router.delete_my_table("shared", current_user={"sub": "alice"})
    )
    with pytest.raises(HTTPException) as missing:
        asyncio.run(
            datalab_router.get_my_table("shared", current_user={"sub": "alice"})
        )
    assert missing.value.status_code == 404
    assert asyncio.run(
        datalab_router.get_my_table("shared", current_user={"sub": "bob"})
    )["rows"] == [{"value": 2.0}]

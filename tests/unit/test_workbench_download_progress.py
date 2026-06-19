from services.cube_workbench import CubeWorkbenchService


def test_prepare_job_records_size_speed_eta_and_transfer_percent(monkeypatch):
    service = object.__new__(CubeWorkbenchService)
    updates = []
    clock = iter([100.0, 102.0, 104.0])
    monkeypatch.setattr(
        "services.cube_workbench.time.monotonic",
        lambda: next(clock),
    )

    monkeypatch.setattr(
        service,
        "_job_cancel_requested",
        lambda **kwargs: False,
    )
    monkeypatch.setattr(
        service,
        "get_session",
        lambda **kwargs: {
            "source_url": "https://example.test/cube.fits",
            "filename": "cube.fits",
        },
    )
    monkeypatch.setattr(
        service,
        "_update_job_state",
        lambda **kwargs: updates.append(kwargs),
    )

    def fake_prepare_product(**kwargs):
        progress = kwargs["progress_callback"]
        progress(2 * 1024 * 1024, 8 * 1024 * 1024)
        progress(8 * 1024 * 1024, 8 * 1024 * 1024)
        return {"status": "cached"}

    monkeypatch.setattr(service, "prepare_product", fake_prepare_product)

    result = service._execute_job_operation(
        session_id="session-1",
        user_id="user-1",
        job_id="job-1",
        operation="prepare",
        payload={},
    )

    transfer_updates = [
        update
        for update in updates
        if (update.get("metrics") or {}).get("bytes_done")
    ]
    first_metrics = transfer_updates[0]["metrics"]
    final_metrics = transfer_updates[-1]["metrics"]

    assert result["status"] == "cached"
    assert first_metrics["filename"] == "cube.fits"
    assert first_metrics["transfer_kind"] == "download"
    assert first_metrics["bytes_total"] == 8 * 1024 * 1024
    assert first_metrics["speed_bps"] > 0
    assert first_metrics["eta_seconds"] > 0
    assert first_metrics["transfer_percent"] == 25.0
    assert final_metrics["transfer_percent"] == 100.0
    assert final_metrics["eta_seconds"] == 0.0

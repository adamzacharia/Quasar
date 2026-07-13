from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pandas as pd
import pytest

from services.datalab_result_store import DatalabResultStore


def _put(store: DatalabResultStore, value: int) -> str:
    return store.put(
        pd.DataFrame({"value": [value]}),
        {"provenance": {"catalog": "test", "rowcount": 1}},
    )


def test_user_namespaces_isolate_same_table_name_and_delete():
    store = DatalabResultStore(enable_disk_cache=False)
    alice_result = _put(store, 1)
    bob_result = _put(store, 2)

    store.save_result(alice_result, "shared", user_id="alice@example.test")
    store.save_result(bob_result, "shared", user_id="bob@example.test")

    assert store.list_my_tables() == []
    assert [entry["name"] for entry in store.list_my_tables(user_id="alice@example.test")] == ["shared"]
    assert [entry["name"] for entry in store.list_my_tables(user_id="bob@example.test")] == ["shared"]
    assert store.load_my_table("shared", user_id="alice@example.test").dataframe.iloc[0, 0] == 1
    assert store.load_my_table("shared", user_id="bob@example.test").dataframe.iloc[0, 0] == 2

    store.delete_my_table("shared", user_id="alice@example.test")
    with pytest.raises(KeyError):
        store.load_my_table("shared", user_id="alice@example.test")
    assert store.load_my_table("shared", user_id="bob@example.test").dataframe.iloc[0, 0] == 2


def test_omitted_user_id_preserves_legacy_default_keys():
    store = DatalabResultStore(enable_disk_cache=False)
    result_id = _put(store, 3)

    store.save_result(result_id, "legacy")

    assert "dlt_index_v1" in store._memory
    assert "dlt_legacy" in store._memory
    assert store.load_my_table("legacy").dataframe.iloc[0, 0] == 3


def test_concurrent_memory_saves_do_not_lose_index_entries():
    store = DatalabResultStore(enable_disk_cache=False)
    count = 20
    result_ids = [_put(store, value) for value in range(count)]
    barrier = Barrier(count)

    def save(value: int) -> None:
        barrier.wait()
        store.save_result(result_ids[value], f"table_{value}", user_id="one-user")

    with ThreadPoolExecutor(max_workers=count) as executor:
        list(executor.map(save, range(count)))

    assert {entry["name"] for entry in store.list_my_tables(user_id="one-user")} == {
        f"table_{value}" for value in range(count)
    }


def test_diskcache_transaction_prevents_cross_store_lost_updates(tmp_path):
    cache_dir = tmp_path / "results"
    first = DatalabResultStore(cache_dir=cache_dir)
    second = DatalabResultStore(cache_dir=cache_dir)
    first_result = _put(first, 10)
    second_result = _put(second, 20)
    barrier = Barrier(2)

    def save(store: DatalabResultStore, result_id: str, name: str) -> None:
        barrier.wait()
        store.save_result(result_id, name, user_id="same-user")

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(save, first, first_result, "first"),
            executor.submit(save, second, second_result, "second"),
        ]
        for future in futures:
            future.result()

    restarted = DatalabResultStore(cache_dir=cache_dir)
    assert {entry["name"] for entry in restarted.list_my_tables(user_id="same-user")} == {
        "first",
        "second",
    }

    restarted.delete_my_table("first", user_id="same-user")
    after_delete = DatalabResultStore(cache_dir=cache_dir)
    assert [
        entry["name"] for entry in after_delete.list_my_tables(user_id="same-user")
    ] == ["second"]
    with pytest.raises(KeyError):
        after_delete.load_my_table("first", user_id="same-user")

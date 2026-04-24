"""Per-backend autotune cache tests (B5).

Verifies:
    * cache key includes backend so quack / cute_sm80 entries don't collide
    * per-backend JSON files are written / read correctly
    * pre-B5 legacy single-file cache is migrated and backed up

See ``docs/internal/research/ns_backend_dispatch_plan.md`` §3 (B5).
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)

import pytest
import torch

from dmuon.optim import syrk_dispatch


@pytest.fixture
def tmp_cache_dir(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setenv("DMUON_CACHE_DIR", tmp)
        # Clear any in-memory state the module may have accumulated from
        # previous tests in this process.
        monkeypatch.setattr(syrk_dispatch, "_syrk_autotune_cache", {})
        yield Path(tmp)


# ---------------------------------------------------------------------------
# Cache file path shape
# ---------------------------------------------------------------------------
def test_cache_path_includes_backend(tmp_cache_dir):
    p_cute = syrk_dispatch._get_autotune_cache_path("cute_sm80")
    p_cublas = syrk_dispatch._get_autotune_cache_path("cublas")
    p_quack = syrk_dispatch._get_autotune_cache_path("quack")
    # All three under the tmp dir, with the backend name in the filename
    for p, tag in [(p_cute, "cute_sm80"), (p_cublas, "cublas"), (p_quack, "quack")]:
        assert p.parent == tmp_cache_dir
        assert tag in p.name
    # Distinct files
    assert p_cute != p_cublas
    assert p_cute != p_quack
    assert p_cublas != p_quack


# ---------------------------------------------------------------------------
# Save / load round-trip per backend
# ---------------------------------------------------------------------------
def test_save_load_roundtrip_isolates_backends(tmp_cache_dir):
    # Seed cache with two entries on different backends, same (M,K,dtype,...)
    key_cute = (128, 64, 0, torch.bfloat16, False, "cute_sm80")
    key_cublas = (128, 64, 0, torch.bfloat16, False, "cublas")
    syrk_dispatch._syrk_autotune_cache[key_cute] = (128, 32, 4)
    syrk_dispatch._syrk_autotune_cache[key_cublas] = None

    syrk_dispatch._save_autotune_cache("cute_sm80")
    syrk_dispatch._save_autotune_cache("cublas")

    cute_file = syrk_dispatch._get_autotune_cache_path("cute_sm80")
    cublas_file = syrk_dispatch._get_autotune_cache_path("cublas")
    assert cute_file.exists()
    assert cublas_file.exists()

    # Each file holds only its backend's entry
    with open(cute_file) as f:
        cute_data = json.load(f)
    with open(cublas_file) as f:
        cublas_data = json.load(f)
    assert len(cute_data) == 1
    assert len(cublas_data) == 1
    # The JSON rows don't include backend (that's the filename) — so both
    # data blobs have the same shape-row key; the distinction is the file.
    assert list(cute_data.values())[0] == [128, 32, 4]
    assert list(cublas_data.values())[0] is None

    # Reload into a fresh in-memory cache and verify both entries come
    # back with the correct backend tag
    syrk_dispatch._syrk_autotune_cache.clear()
    syrk_dispatch._load_autotune_cache("cute_sm80")
    syrk_dispatch._load_autotune_cache("cublas")
    assert syrk_dispatch._syrk_autotune_cache[key_cute] == (128, 32, 4)
    assert syrk_dispatch._syrk_autotune_cache[key_cublas] is None


def test_save_only_writes_requested_backend(tmp_cache_dir):
    """Mutating cute_sm80 and saving cublas must NOT touch cute_sm80 file."""
    key_cute = (64, 32, 0, torch.bfloat16, False, "cute_sm80")
    key_cublas = (64, 32, 0, torch.bfloat16, False, "cublas")
    syrk_dispatch._syrk_autotune_cache[key_cute] = (64, 16, 2)
    syrk_dispatch._syrk_autotune_cache[key_cublas] = None

    # Save only cublas
    syrk_dispatch._save_autotune_cache("cublas")

    cute_file = syrk_dispatch._get_autotune_cache_path("cute_sm80")
    cublas_file = syrk_dispatch._get_autotune_cache_path("cublas")
    assert not cute_file.exists()
    assert cublas_file.exists()


# ---------------------------------------------------------------------------
# Legacy cache migration
# ---------------------------------------------------------------------------
def test_legacy_cache_migrates_to_cute_sm80_and_backs_up(tmp_cache_dir):
    """Pre-B5 single-file caches get copied to the cute_sm80 path and the
    original is renamed with a .bak_preB5 suffix."""
    legacy = syrk_dispatch._get_legacy_cache_path()
    # Seed a legacy-format JSON (pre-B5 shape)
    legacy_row = json.dumps([128, 64, 0, "bf16", False])
    legacy.write_text(json.dumps({legacy_row: [128, 32, 4]}))

    syrk_dispatch._migrate_legacy_cache_if_present()

    new_path = syrk_dispatch._get_autotune_cache_path("cute_sm80")
    assert new_path.exists()
    assert not legacy.exists()
    backup = legacy.with_suffix(legacy.suffix + ".bak_preB5")
    assert backup.exists()

    # Reload from the new file, check entry lands in cute_sm80 backend
    syrk_dispatch._syrk_autotune_cache.clear()
    syrk_dispatch._load_autotune_cache("cute_sm80")
    key = (128, 64, 0, torch.bfloat16, False, "cute_sm80")
    assert syrk_dispatch._syrk_autotune_cache[key] == (128, 32, 4)


def test_legacy_migration_noop_when_already_migrated(tmp_cache_dir):
    """If the cute_sm80 file exists already, the legacy file is left
    untouched so manual edits survive.  The expectation is that an admin
    who cares enough to hand-edit the new file also knows what they're
    doing with the legacy copy."""
    # Create both files; both should remain afterwards
    legacy = syrk_dispatch._get_legacy_cache_path()
    legacy.write_text("{}")
    new_path = syrk_dispatch._get_autotune_cache_path("cute_sm80")
    new_path.write_text("{}")

    syrk_dispatch._migrate_legacy_cache_if_present()

    # Legacy still present, not renamed to bak
    assert legacy.exists()
    backup = legacy.with_suffix(legacy.suffix + ".bak_preB5")
    assert not backup.exists()

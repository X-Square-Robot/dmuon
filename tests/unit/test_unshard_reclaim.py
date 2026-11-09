"""Real cross-stream consumers must survive early allocator reclamation."""

import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch


CASES = [
    ("ordinary", False),
    ("matrix", False),
    ("sharded_adamw", False),
    ("sharded_adamw", True),
    ("sharded_muon", False),
    ("sharded_muon", True),
]


def _exercise(kind, separate, backend):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from dmuon._backends.fsdp2.group import DedicatedParamGroup
    from dmuon._backends.fsdp2.matrix_group import MatrixParamGroup

    torch.cuda.set_device(0)
    if backend == "native":
        assert torch.cuda.memory.get_allocator_backend() == "native"
        assert torch.cuda.memory._snapshot()["allocator_settings"]["expandable_segments"]
    else:
        assert torch.cuda.memory.get_allocator_backend() == "cudaMallocAsync"
    broadcast, compute = torch.cuda.Stream(), torch.cuda.Stream()
    sharded = torch.cuda.Stream() if separate else broadcast
    allocation = sharded if kind.startswith("sharded") else broadcast
    # One full 20 MiB VMM page; the output also occupies complete pages.
    # Trimming cannot accidentally synchronize compute via an unrelated page.
    count = 20 * 2**20 // 2
    with torch.cuda.stream(allocation):
        source = torch.full((count,), 0.5, dtype=torch.bfloat16, device="cuda")
    with torch.cuda.stream(compute):
        target = torch.empty(count, dtype=torch.float32, device="cuda")
        compute.wait_stream(allocation)
        target.copy_(source)  # Warm the conversion kernel before the delayed read.
    torch.cuda.synchronize()

    group = DedicatedParamGroup.__new__(DedicatedParamGroup)
    group.params = []
    group._broadcast_event = group._sharded_adamw_unshard_event = None
    group._matrix_params = None
    group._packed_buf_by_owner = {}
    group._device_module = torch.cuda
    group.comm_ctx = SimpleNamespace(
        broadcast_stream=broadcast,
        sharded_adamw_unshard_stream=sharded,
        sharded_adamw_unshard_separate_stream_enabled=separate,
    )
    group._is_unsharded = True
    if kind == "ordinary":
        group._packed_buf_by_owner[(0, 0)] = source
    elif kind == "matrix":

        class Parent:
            def install_placeholder(self):
                pass

        matrices = MatrixParamGroup.__new__(MatrixParamGroup)
        matrices.buffers = {Parent(): source}
        group._matrix_params = matrices
    else:
        parameter = SimpleNamespace(
            reshard=lambda: None,
            uses_sharded_adamw=lambda: kind == "sharded_adamw",
            uses_sharded_muon_forward=lambda: kind == "sharded_muon",
        )
        setattr(
            parameter,
            "_sharded_adamw_full_padded"
            if kind == "sharded_adamw"
            else "_sharded_muon_full_padded",
            source,
        )
        group.params = [parameter]

    with torch.cuda.stream(compute):
        torch.cuda._sleep(200_000_000)
        target.copy_(source)
        group.reshard()
    torch.cuda.empty_cache()
    compute.synchronize()
    assert source.untyped_storage().nbytes() == 0
    assert not group._is_unsharded
    torch.testing.assert_close(target, torch.full_like(target, 0.5), rtol=0, atol=0)


@pytest.mark.parametrize("kind,separate", CASES)
@pytest.mark.parametrize("backend", ["native", "cudaMallocAsync"])
def test_unshard_storage_survives_compute_until_reclaim(kind, separate, backend):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if backend == "cudaMallocAsync" and tuple(map(int, torch.version.cuda.split(".")[:2])) < (
        11,
        4,
    ):
        pytest.skip("requires CUDA 11.4+ async allocator")
    env = dict(os.environ)
    env.pop("PYTORCH_ALLOC_CONF", None)
    env.pop("PYTORCH_NO_CUDA_MEMORY_CACHING", None)
    env["PYTORCH_CUDA_ALLOC_CONF"] = (
        "expandable_segments:True" if backend == "native" else "backend:cudaMallocAsync"
    )
    env["CUDA_LAUNCH_BLOCKING"] = "0"
    # An unsafe release can poison a CUDA context. Keep every case isolated.
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--child",
            kind,
            str(int(separate)),
            backend,
        ],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    assert sys.argv[1] == "--child"
    _exercise(sys.argv[2], bool(int(sys.argv[3])), sys.argv[4])

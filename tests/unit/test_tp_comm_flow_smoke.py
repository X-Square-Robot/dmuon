"""Smoke test for the TP comm flow — validates §2.3 lifecycle data flow.

**Pre-T2 validation**: this script exercises the full 6-stage TP lifecycle
using only raw gloo collectives on a (dp, tp) mesh.  It confirms:

1. DP process groups are per-TP-coord (sliced sub-mesh gives separate
   groups per TP rank — verified by doing DP reduce within one TP coord
   without affecting another TP coord's grads).
2. After DP reduce, every rank (s*, t) — one per TP coord — holds
   a meaningful TP-local grad shard.
3. TP gather within the TP group at the DP owner collects all T shards
   at TP owner t*.
4. After owner's stand-in "NS" (scale-by-3), TP scatter distributes
   slices back.
5. DP broadcast fans the updated shard out to DP peers.

Uses torch.multiprocessing.spawn with gloo (4 ranks, 2×2 DP×TP) so we
exercise real collectives.  fake_pg would only validate call structure,
not numerical values.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)

import torch
import torch.distributed as dist
import torch.multiprocessing as tmp
from torch.distributed import init_device_mesh


def _worker(rank: int, world_size: int, tmp_dir: str) -> None:
    """One worker process. Writes a per-rank result file under tmp_dir.

    Any exception is captured and written to ``rank<r>.err`` so the parent
    can surface it.
    """
    out_path = os.path.join(tmp_dir, f"rank{rank}.ok")
    err_path = os.path.join(tmp_dir, f"rank{rank}.err")

    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29517"
        dist.init_process_group(
            backend="gloo", rank=rank, world_size=world_size, init_method="env://"
        )

        mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("dp", "tp"))
        dp_group = mesh["dp"].get_group()
        tp_group = mesh["tp"].get_group()

        my_dp = mesh["dp"].get_local_rank()
        my_tp = mesh["tp"].get_local_rank()

        full_shape = (4, 4)
        tp_size = mesh["tp"].size()
        local_shape = (full_shape[0] // tp_size, full_shape[1])

        # Per-rank marker: 10*dp + tp → after DP AVG within a TP coord,
        # expect 5 + tp.
        local_grad = torch.full(local_shape, float(10 * my_dp + my_tp))

        # Stage ② DP reduce (AVG). ``dst`` in torch.distributed.reduce is a
        # global rank; the sub-mesh group at this rank may not contain
        # global 0, so convert the in-group rank 0 to the global rank.
        dp_owner_global = dist.get_global_rank(dp_group, 0)
        dist.reduce(
            local_grad, dst=dp_owner_global, op=dist.ReduceOp.AVG, group=dp_group
        )
        is_dp_owner = my_dp == 0

        checks: dict = {"rank": rank, "my_dp": my_dp, "my_tp": my_tp}

        if is_dp_owner:
            expected = 5.0 + my_tp
            checks["dp_reduced"] = bool(
                torch.allclose(local_grad, torch.full_like(local_grad, expected))
            )

        # Stage ③ TP gather (only DP owners participate — the full flow only
        # cares about DP owner coords within the TP group)
        tp_owner_global = dist.get_global_rank(tp_group, 0) if is_dp_owner else None
        full_grad = None
        if is_dp_owner:
            if my_tp == 0:
                buf = [torch.empty_like(local_grad) for _ in range(tp_size)]
                dist.gather(
                    local_grad, gather_list=buf, dst=tp_owner_global, group=tp_group
                )
                full_grad = torch.cat(buf, dim=0)
                expected_full = torch.tensor(
                    [[5.0] * 4] * 2 + [[6.0] * 4] * 2
                )
                checks["tp_gathered_full_grad"] = bool(
                    torch.allclose(full_grad, expected_full)
                )
            else:
                dist.gather(
                    local_grad, gather_list=None, dst=tp_owner_global, group=tp_group
                )

        # Stage ④ NS stand-in: multiply by 3
        delta_full = full_grad * 3.0 if (is_dp_owner and my_tp == 0) else None

        # Stage ⑤ TP scatter
        if is_dp_owner:
            delta_local = torch.empty_like(local_grad)
            if my_tp == 0:
                splits = [s.contiguous() for s in delta_full.split(local_shape[0], dim=0)]
                dist.scatter(
                    delta_local, scatter_list=splits, src=tp_owner_global,
                    group=tp_group,
                )
            else:
                dist.scatter(
                    delta_local, scatter_list=None, src=tp_owner_global,
                    group=tp_group,
                )
            expected_local = torch.full(local_shape, (5.0 + my_tp) * 3.0)
            checks["tp_scattered_delta"] = bool(
                torch.allclose(delta_local, expected_local)
            )

        # Stage ⑥ DP broadcast to non-owner DP peers
        if not is_dp_owner:
            delta_local = torch.empty_like(local_grad)
        dist.broadcast(delta_local, src=dp_owner_global, group=dp_group)
        expected_local = torch.full(local_shape, (5.0 + my_tp) * 3.0)
        checks["dp_broadcast_delta"] = bool(
            torch.allclose(delta_local, expected_local)
        )

        with open(out_path, "w") as f:
            import json
            json.dump(checks, f)
    except Exception as e:
        import traceback
        with open(err_path, "w") as f:
            f.write(f"rank {rank}: {e}\n{traceback.format_exc()}")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def test_tp_comm_flow_4rank_gloo(tmp_path):
    """torch.multiprocessing.spawn: 4 ranks = DP(2) × TP(2).  Validates
    the full 6-stage TP lifecycle numerically.
    """
    world_size = 4
    tmp_dir = str(tmp_path)
    tmp.spawn(
        _worker,
        args=(world_size, tmp_dir),
        nprocs=world_size,
        join=True,
    )

    errs: list[str] = []
    results: list[dict] = []
    for rank in range(world_size):
        err = os.path.join(tmp_dir, f"rank{rank}.err")
        ok = os.path.join(tmp_dir, f"rank{rank}.ok")
        if os.path.exists(err):
            with open(err) as f:
                errs.append(f.read())
        elif os.path.exists(ok):
            import json
            with open(ok) as f:
                results.append(json.load(f))
    assert not errs, "worker errors:\n" + "\n".join(errs)
    assert len(results) == world_size, f"missing results: got {len(results)}/{world_size}"

    dp_owners = [r for r in results if r["my_dp"] == 0]
    assert len(dp_owners) == 2
    for r in dp_owners:
        assert r["dp_reduced"], f"DP reduce wrong at rank {r['rank']}"
        assert r["tp_scattered_delta"], f"TP scatter wrong at rank {r['rank']}"

    tp_owner = next(r for r in results if r["my_dp"] == 0 and r["my_tp"] == 0)
    assert tp_owner["tp_gathered_full_grad"], "TP gather reconstruction wrong"

    for r in results:
        assert r["dp_broadcast_delta"], f"DP broadcast wrong at rank {r['rank']}"

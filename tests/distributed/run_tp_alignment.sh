#!/usr/bin/env bash
# Loss parity matrix for TP topologies.
#
# This wraps test_tp_alignment.py instead of duplicating setup code.  It runs
# deterministic trajectories for each topology/baseline and checks loss plus
# global owned-data digest gaps.
#
# Usage:
#   bash tests/distributed/run_tp_alignment.sh
#   DMUON_ALIGN_MATRIX=llm_full DMUON_ALIGN_OUT=docs/internal/report/tp_llm_loss_matrix \
#       bash tests/distributed/run_tp_alignment.sh

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
SCRIPT="${HERE}/test_tp_alignment.py"
OUT="${DMUON_ALIGN_OUT:-/tmp/dmuon_align_$$}"
STEPS="${DMUON_ALIGN_STEPS:-3}"
MATRIX="${DMUON_ALIGN_MATRIX:-tiny_full}"
rm -rf "$OUT"
mkdir -p "$OUT"

port_base="${DMUON_ALIGN_PORT_BASE:-33100}"

run_case() {
    local key="$1"
    local topology="$2"
    local mode="$3"
    local owner="$4"
    local world="$5"
    local model="${6:-tiny}"
    local tp_scope="${7:-mlp}"
    local port="$port_base"
    port_base=$((port_base + 1))
    echo "=== ${key}: model=${model} topology=${topology} owner=${owner} mode=${mode} scope=${tp_scope} world=${world} port=${port} ==="
    DMUON_ALIGN_TOPOLOGY="$topology" \
    DMUON_ALIGN_MODE="$mode" \
    DMUON_ALIGN_OWNER="$owner" \
    DMUON_ALIGN_MODEL="$model" \
    DMUON_ALIGN_TP_SCOPE="$tp_scope" \
    DMUON_ALIGN_RUN="$key" \
    DMUON_ALIGN_OUT="$OUT" \
    DMUON_ALIGN_STEPS="$STEPS" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    torchrun --nproc_per_node="$world" --master_port="$port" "$SCRIPT"
}

run_expected_fail() {
    local key="$1"
    local topology="$2"
    local world="$3"
    local model="$4"
    local tp_scope="$5"
    local pattern="$6"
    local port="$port_base"
    local log="${OUT}/${key}.expected_failure.log"
    port_base=$((port_base + 1))
    echo "=== ${key}: EXPECT FAIL model=${model} topology=${topology} scope=${tp_scope} world=${world} port=${port} ==="
    set +e
    DMUON_ALIGN_TOPOLOGY="$topology" \
    DMUON_ALIGN_MODE="sync" \
    DMUON_ALIGN_OWNER="lpt" \
    DMUON_ALIGN_MODEL="$model" \
    DMUON_ALIGN_TP_SCOPE="$tp_scope" \
    DMUON_ALIGN_RUN="$key" \
    DMUON_ALIGN_OUT="$OUT" \
    DMUON_ALIGN_STEPS="$STEPS" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    torchrun --nproc_per_node="$world" --master_port="$port" "$SCRIPT" >"$log" 2>&1
    local status=$?
    set -e
    if [[ "$status" -eq 0 ]]; then
        echo "${key}: expected failure but command succeeded" >&2
        exit 1
    fi
    if ! grep -q "$pattern" "$log"; then
        echo "${key}: failure log did not contain expected pattern: ${pattern}" >&2
        tail -n 80 "$log" >&2
        exit 1
    fi
    echo "PASS expected failure: ${key} (${pattern})"
}

run_tiny_full_matrix() {
    # Pure TP topologies.
    run_case tp2_lpt tp2 sync lpt 2 tiny mlp
    run_case tp2_async tp2 async lpt 2 tiny mlp
    run_case tp2_rank0 tp2 sync rank0 2 tiny mlp

    run_case tp4_lpt tp4 sync lpt 4 tiny mlp
    run_case tp4_async tp4 async lpt 4 tiny mlp
    run_case tp4_rank0 tp4 sync rank0 4 tiny mlp

    # FSDP shard/DP x TP topologies.
    run_case dp_tp2_lpt dp_tp2 sync lpt 4 tiny mlp
    run_case dp_tp2_async dp_tp2 async lpt 4 tiny mlp
    run_case dp_tp2_rank0 dp_tp2 sync rank0 4 tiny mlp

    run_case dp_tp4_lpt dp_tp4 sync lpt 8 tiny mlp
    run_case dp_tp4_async dp_tp4 async lpt 8 tiny mlp
    run_case dp_tp4_rank0 dp_tp4 sync rank0 8 tiny mlp

    # 3D HSDP x TP.  With 8 GPUs this covers true HSDP only for TP=2
    # (R=2, G=2, T=2).  HSDP x TP=4 with both R>1 and G>1 requires 16 GPUs.
    run_case hsdp_tp2_lpt hsdp_tp2 sync lpt 8 tiny mlp
    run_case hsdp_tp2_async hsdp_tp2 async lpt 8 tiny mlp
    run_case hsdp_tp2_rank0 hsdp_tp2 sync rank0 8 tiny mlp

    # TP-size-1 must behave like pure DP.
    run_case tp1 tp1 sync lpt 4 tiny mlp
    run_case dp_only dp_only sync lpt 4 tiny mlp
}

run_llm_full_matrix() {
    # Llama has kv_heads=4 in this test shape, so full attention TP2 and TP4
    # are both valid.  This covers the full topology matrix available on 8 GPUs.
    run_case llama_tp2_lpt tp2 sync lpt 2 llama full
    run_case llama_tp2_async tp2 async lpt 2 llama full
    run_case llama_tp2_async_drain tp2 async_drain lpt 2 llama full
    run_case llama_tp2_rank0 tp2 sync rank0 2 llama full

    run_case llama_tp4_lpt tp4 sync lpt 4 llama full
    run_case llama_tp4_async tp4 async lpt 4 llama full
    run_case llama_tp4_async_drain tp4 async_drain lpt 4 llama full
    run_case llama_tp4_rank0 tp4 sync rank0 4 llama full

    run_case llama_dp_tp2_lpt dp_tp2 sync lpt 4 llama full
    run_case llama_dp_tp2_async dp_tp2 async lpt 4 llama full
    run_case llama_dp_tp2_async_drain dp_tp2 async_drain lpt 4 llama full
    run_case llama_dp_tp2_rank0 dp_tp2 sync rank0 4 llama full

    run_case llama_dp_tp4_lpt dp_tp4 sync lpt 8 llama full
    run_case llama_dp_tp4_async dp_tp4 async lpt 8 llama full
    run_case llama_dp_tp4_async_drain dp_tp4 async_drain lpt 8 llama full
    run_case llama_dp_tp4_rank0 dp_tp4 sync rank0 8 llama full

    run_case llama_hsdp_tp2_lpt hsdp_tp2 sync lpt 8 llama full
    run_case llama_hsdp_tp2_async hsdp_tp2 async lpt 8 llama full
    run_case llama_hsdp_tp2_async_drain hsdp_tp2 async_drain lpt 8 llama full
    run_case llama_hsdp_tp2_rank0 hsdp_tp2 sync rank0 8 llama full

    run_case llama_tp1 tp1 sync lpt 4 llama full
    run_case llama_dp_only dp_only sync lpt 4 llama full

    # Qwen has kv_heads=2 in this test shape.  Full TP2 is valid, while full
    # TP4 is intentionally rejected to lock the GQA divisibility contract.
    run_case qwen_tp2_lpt tp2 sync lpt 2 qwen full
    run_case qwen_tp2_async tp2 async lpt 2 qwen full
    run_case qwen_tp2_async_drain tp2 async_drain lpt 2 qwen full
    run_case qwen_tp2_rank0 tp2 sync rank0 2 qwen full

    run_case qwen_dp_tp2_lpt dp_tp2 sync lpt 4 qwen full
    run_case qwen_dp_tp2_async dp_tp2 async lpt 4 qwen full
    run_case qwen_dp_tp2_async_drain dp_tp2 async_drain lpt 4 qwen full
    run_case qwen_dp_tp2_rank0 dp_tp2 sync rank0 4 qwen full

    run_case qwen_hsdp_tp2_lpt hsdp_tp2 sync lpt 8 qwen full
    run_case qwen_hsdp_tp2_async hsdp_tp2 async lpt 8 qwen full
    run_case qwen_hsdp_tp2_async_drain hsdp_tp2 async_drain lpt 8 qwen full
    run_case qwen_hsdp_tp2_rank0 hsdp_tp2 sync rank0 8 qwen full

    run_expected_fail qwen_tp4_full_invalid tp4 4 qwen full \
        "kv_heads=2 must be divisible by tp_size=4"
}

case "$MATRIX" in
    tiny_full)
        run_tiny_full_matrix
        ;;
    llm_full)
        run_llm_full_matrix
        ;;
    *)
        echo "Unknown DMUON_ALIGN_MATRIX=${MATRIX}; expected tiny_full or llm_full" >&2
        exit 2
        ;;
esac

echo
echo "=== TP loss parity matrix ==="
DMUON_ALIGN_MATRIX="$MATRIX" python - "$OUT" <<'PY'
import glob
import json
import os
import sys

out = sys.argv[1]
matrix = os.environ.get("DMUON_ALIGN_MATRIX", "tiny_full")
strict = float(os.environ.get("DMUON_ALIGN_STRICT_TOL", "1e-6"))
rank0_tol = float(os.environ.get("DMUON_ALIGN_RANK0_TOL", "1e-2"))
rank0_rel_tol = float(os.environ.get("DMUON_ALIGN_RANK0_REL_TOL", "1e-2"))
async_rel_tol = float(os.environ.get("DMUON_ALIGN_ASYNC_REL_TOL", "1e-3"))
async_digest_rel_tol = float(os.environ.get("DMUON_ALIGN_ASYNC_DIGEST_REL_TOL", "5e-2"))
tp1_tol = float(os.environ.get("DMUON_ALIGN_TP1_TOL", "2e-2"))
cross_rel_tol = float(os.environ.get("DMUON_ALIGN_CROSS_REL_TOL", "5e-2"))
cross_iter0_tol = float(os.environ.get("DMUON_ALIGN_CROSS_ITER0_TOL", "1e-4"))

data = {}
for fp in sorted(glob.glob(os.path.join(out, "*.json"))):
    with open(fp) as f:
        d = json.load(f)
    data[d["run_id"]] = d

def max_gap(a, b, field):
    xs = data[a][field]
    ys = data[b][field]
    return max(abs(float(x) - float(y)) for x, y in zip(xs, ys))

def rel_gap(a, b, field):
    xs = [float(x) for x in data[a][field]]
    ys = [float(y) for y in data[b][field]]
    gap = max(abs(x - y) for x, y in zip(xs, ys))
    denom = max(max(abs(x) for x in xs), max(abs(y) for y in ys), 1e-8)
    return gap, gap / denom

def assert_gap(name, a, b, field, tol):
    gap = max_gap(a, b, field)
    status = "PASS" if gap <= tol else "FAIL"
    print(f"{name:<42} {field:<14} gap={gap:.6e} tol={tol:.1e} {status}")
    if gap > tol:
        raise SystemExit(
            f"{name} {field} gap {gap:.6e} exceeds tolerance {tol:.1e}"
        )

def assert_rel_gap(name, a, b, field, tol):
    gap, rel = rel_gap(a, b, field)
    status = "PASS" if rel <= tol else "FAIL"
    print(
        f"{name:<42} {field:<14} rel={rel:.6e} abs={gap:.6e} "
        f"tol={tol:.1e} {status}"
    )
    if rel > tol:
        raise SystemExit(
            f"{name} {field} relative gap {rel:.6e} exceeds tolerance {tol:.1e}"
        )

def report_rel_gap(name, a, b, field):
    gap, rel = rel_gap(a, b, field)
    print(f"{name:<42} {field:<14} rel={rel:.6e} abs={gap:.6e} INFO")

def tp_size_for(topology):
    if topology in ("tp4", "dp_tp4"):
        return 4
    if topology in ("tp2", "dp_tp", "dp_tp2", "hsdp_tp", "hsdp_tp2"):
        return 2
    return 1

def assert_owner_coverage():
    for key, d in sorted(data.items()):
        tp_size = tp_size_for(d["topology"])
        coverage = [int(x) for x in d["owner_coverage"]]
        if tp_size == 1:
            if int(d["tp_param_count"]) != 0:
                raise SystemExit(
                    f"{key}: expected no TP params for {d['topology']}, "
                    f"got {d['tp_param_count']}"
                )
            continue
        if d["owner_mode"] == "rank0":
            if coverage != [0]:
                raise SystemExit(f"{key}: rank0 baseline coverage={coverage}")
        elif len(coverage) != tp_size:
            raise SystemExit(
                f"{key}: LPT should cover all {tp_size} TP ranks, "
                f"got coverage={coverage}"
            )

print(
    f"{'case':<24} {'model':<6} {'scope':<5} {'topology':<9} "
    f"{'owner':<6} {'mode':<6} {'coverage':<14} losses"
)
print("-" * 130)
for key in sorted(data):
    d = data[key]
    losses = ", ".join(f"{float(x):+.10f}" for x in d["losses"])
    print(
        f"{key:<24} {d.get('model', 'tiny'):<6} {d.get('tp_scope', 'mlp'):<5} "
        f"{d['topology']:<9} {d['owner_mode']:<6} "
        f"{d['mode']:<6} {str(d['owner_coverage']):<14} {losses}"
    )

print()
print("=== Owner coverage checks ===")
assert_owner_coverage()
print("PASS: owner coverage matches topology expectations")

print()
print("=== Gap checks ===")

def maybe_check_family(prefixes):
    for prefix in prefixes:
        lpt = f"{prefix}_lpt"
        async_key = f"{prefix}_async"
        async_drain_key = f"{prefix}_async_drain"
        rank0 = f"{prefix}_rank0"
        if lpt in data and async_key in data:
            if prefix.startswith(("llama_", "qwen_")):
                assert_rel_gap(
                    f"{prefix} sync vs async bounded",
                    lpt,
                    async_key,
                    "losses",
                    async_rel_tol,
                )
                assert_rel_gap(
                    f"{prefix} sync vs async digest bounded",
                    lpt,
                    async_key,
                    "weight_digest",
                    async_digest_rel_tol,
                )
            else:
                assert_gap(
                    f"{prefix} sync vs async",
                    lpt,
                    async_key,
                    "losses",
                    strict,
                )
                assert_gap(
                    f"{prefix} sync vs async",
                    lpt,
                    async_key,
                    "weight_digest",
                    strict,
                )
        if lpt in data and async_drain_key in data:
            if prefix.startswith(("llama_", "qwen_")):
                assert_rel_gap(
                    f"{prefix} sync vs async_drain bounded",
                    lpt,
                    async_drain_key,
                    "losses",
                    async_rel_tol,
                )
                assert_rel_gap(
                    f"{prefix} sync vs async_drain digest bounded",
                    lpt,
                    async_drain_key,
                    "weight_digest",
                    async_digest_rel_tol,
                )
            else:
                assert_gap(
                    f"{prefix} sync vs async_drain",
                    lpt,
                    async_drain_key,
                    "losses",
                    strict,
                )
                assert_gap(
                    f"{prefix} sync vs async_drain",
                    lpt,
                    async_drain_key,
                    "weight_digest",
                    strict,
                )
        if lpt in data and rank0 in data:
            if prefix.startswith(("llama_", "qwen_")):
                assert_rel_gap(
                    f"{prefix} LPT vs rank0 bounded",
                    lpt,
                    rank0,
                    "losses",
                    rank0_rel_tol,
                )
            else:
                assert_gap(
                    f"{prefix} LPT vs rank0 bounded",
                    lpt,
                    rank0,
                    "losses",
                    rank0_tol,
                )
            report_rel_gap(
                f"{prefix} LPT vs rank0 digest", lpt, rank0, "weight_digest"
            )

maybe_check_family([
    "tp2", "tp4", "dp_tp2", "dp_tp4", "hsdp_tp2",
    "llama_tp2", "llama_tp4", "llama_dp_tp2", "llama_dp_tp4",
    "llama_hsdp_tp2",
    "qwen_tp2", "qwen_dp_tp2", "qwen_hsdp_tp2",
])

if "tp1" in data and "dp_only" in data:
    iter0_gap = abs(float(data["tp1"]["losses"][0]) - float(data["dp_only"]["losses"][0]))
    print(
        f"{'TP1 vs DP-only iter0':<42} {'losses':<14} "
        f"gap={iter0_gap:.6e} tol={strict:.1e} "
        f"{'PASS' if iter0_gap <= strict else 'FAIL'}"
    )
    if iter0_gap > strict:
        raise SystemExit("TP1 vs DP-only iter0 mismatch")
    assert_gap("TP1 vs DP-only bounded", "tp1", "dp_only", "losses", tp1_tol)

if "llama_tp1" in data and "llama_dp_only" in data:
    iter0_gap = abs(
        float(data["llama_tp1"]["losses"][0])
        - float(data["llama_dp_only"]["losses"][0])
    )
    print(
        f"{'Llama TP1 vs DP-only iter0':<42} {'losses':<14} "
        f"gap={iter0_gap:.6e} tol={strict:.1e} "
        f"{'PASS' if iter0_gap <= strict else 'FAIL'}"
    )
    if iter0_gap > strict:
        raise SystemExit("Llama TP1 vs DP-only iter0 mismatch")
    assert_gap(
        "Llama TP1 vs DP-only bounded",
        "llama_tp1",
        "llama_dp_only",
        "losses",
        tp1_tol,
    )

print()
print("=== Cross-topology loss gaps ===")
if matrix == "llm_full":
    cross_pairs = [
        ("Llama TP2 vs TP4", "llama_tp2_lpt", "llama_tp4_lpt"),
        ("Llama TP2 vs DPxTP2", "llama_tp2_lpt", "llama_dp_tp2_lpt"),
        ("Llama TP4 vs DPxTP4", "llama_tp4_lpt", "llama_dp_tp4_lpt"),
        ("Llama TP2 vs HSDPxTP2", "llama_tp2_lpt", "llama_hsdp_tp2_lpt"),
        ("Llama DP-only vs TP2", "llama_dp_only", "llama_tp2_lpt"),
        ("Llama DP-only vs TP4", "llama_dp_only", "llama_tp4_lpt"),
        ("Qwen TP2 vs DPxTP2", "qwen_tp2_lpt", "qwen_dp_tp2_lpt"),
        ("Qwen TP2 vs HSDPxTP2", "qwen_tp2_lpt", "qwen_hsdp_tp2_lpt"),
    ]
    for name, a, b in cross_pairs:
        if a not in data or b not in data:
            continue
        iter0_gap = abs(float(data[a]["losses"][0]) - float(data[b]["losses"][0]))
        status = "PASS" if iter0_gap <= cross_iter0_tol else "FAIL"
        print(
            f"{name + ' iter0':<42} {'losses':<14} "
            f"gap={iter0_gap:.6e} tol={cross_iter0_tol:.1e} {status}"
        )
        if iter0_gap > cross_iter0_tol:
            raise SystemExit(f"{name} iter0 gap {iter0_gap:.6e} too large")
        assert_rel_gap(name, a, b, "losses", cross_rel_tol)
else:
    topology_keys = [
        "dp_only",
        "tp1",
        "tp2_lpt",
        "tp4_lpt",
        "dp_tp2_lpt",
        "dp_tp4_lpt",
        "hsdp_tp2_lpt",
    ]
    print(f"{'pair':<30} max_loss_gap")
    print("-" * 48)
    max_pair = ("", "", -1.0)
    for i, a in enumerate(topology_keys):
        if a not in data:
            continue
        for b in topology_keys[i + 1:]:
            if b not in data:
                continue
            gap = max_gap(a, b, "losses")
            print(f"{a + ' vs ' + b:<30} {gap:.6e}")
            if gap > max_pair[2]:
                max_pair = (a, b, gap)
    print(
        f"max_cross_topology_gap = {max_pair[2]:.6e} "
        f"({max_pair[0]} vs {max_pair[1]})"
    )

print()
print(f"PASS: wrote artifacts to {out}")
PY

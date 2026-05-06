#!/usr/bin/env bash
# Benchmark DMuon TP on random-init HF Llama/Qwen model shapes.
#
# Defaults:
#   model: llama3b
#   matrix: core (TP2/TP4 LPT sync/async + rank0 sync)
#   TP scope: full projection TP unless DMUON_TP_LLM_PARALLELIZE overrides it
#   output: docs/internal/report/tp_llm_benchmark

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
SCRIPT="${HERE}/bench_tp_llm.py"
OUT="${DMUON_TP_LLM_DIR:-docs/internal/report/tp_llm_benchmark}"
PORT_BASE="${DMUON_TP_LLM_PORT_BASE:-30200}"
MODEL="${DMUON_TP_LLM_MODEL:-llama3b}"
MATRIX="${DMUON_TP_LLM_MATRIX:-core}"

mkdir -p "$OUT"

run_case() {
    local key="$1"
    local topology="$2"
    local world="$3"
    local owner="$4"
    local async="$5"
    local port="$PORT_BASE"
    PORT_BASE=$((PORT_BASE + 1))
    local outfile="${OUT}/${MODEL}_${key}.json"
    echo "=== ${MODEL}_${key}: topology=${topology} owner=${owner} async=${async} world=${world} port=${port} ==="
    DMUON_TP_LLM_OWNER="$owner" \
    DMUON_TP_LLM_ASYNC="$async" \
    DMUON_TP_LLM_OUT="$outfile" \
    torchrun --nproc_per_node="$world" --master_port="$port" "$SCRIPT" "$MODEL" "$topology"
}

run_core_matrix() {
    run_case tp2_lpt_sync tp2 2 lpt 0
    run_case tp2_lpt_async tp2 2 lpt 1
    run_case tp2_rank0_sync tp2 2 rank0 0

    run_case tp4_lpt_sync tp4 4 lpt 0
    run_case tp4_lpt_async tp4 4 lpt 1
    run_case tp4_rank0_sync tp4 4 rank0 0
}

run_full_matrix() {
    run_core_matrix

    run_case dp_tp2_lpt_sync dp_tp2 4 lpt 0
    run_case dp_tp2_lpt_async dp_tp2 4 lpt 1
    run_case dp_tp2_rank0_sync dp_tp2 4 rank0 0

    run_case dp_tp4_lpt_sync dp_tp4 8 lpt 0
    run_case dp_tp4_lpt_async dp_tp4 8 lpt 1
    run_case dp_tp4_rank0_sync dp_tp4 8 rank0 0

    run_case hsdp_tp2_lpt_sync hsdp_tp2 8 lpt 0
    run_case hsdp_tp2_lpt_async hsdp_tp2 8 lpt 1
    run_case hsdp_tp2_rank0_sync hsdp_tp2 8 rank0 0
}

case "$MATRIX" in
    core)
        run_core_matrix
        ;;
    full)
        run_full_matrix
        ;;
    *)
        echo "Unknown DMUON_TP_LLM_MATRIX=${MATRIX}; expected core or full" >&2
        exit 2
        ;;
esac

echo
echo "=== TP LLM benchmark summary ==="
python - "$OUT" "$MODEL" <<'PY'
import glob
import json
import os
import sys

out, model = sys.argv[1], sys.argv[2]
rows = []
data = {}
for fp in sorted(glob.glob(os.path.join(out, f"{model}_*.json"))):
    with open(fp) as f:
        d = json.load(f)
    key = os.path.splitext(os.path.basename(fp))[0].removeprefix(f"{model}_")
    data[key] = d
    s = d["summary"]
    data_factor = s.get("data_parallel_factor")
    if data_factor is None:
        topology = d["topology"]
        tp_size = 4 if topology.endswith("tp4") else 2
        data_factor = d["world_size"] // tp_size
    global_tokens_per_s = s.get(
        "global_tokens_per_s_p50",
        s["tokens_per_s_p50"] * data_factor,
    )
    rows.append((
        key,
        d["topology"],
        d["owner"],
        "async" if d["replicate_async"] else "sync",
        d["world_size"],
        data_factor,
        s["step_ms_p50"],
        s["step_ms_p90"],
        s["tokens_per_s_p50"],
        global_tokens_per_s,
        s["peak_memory_allocated_gb_max_rank"],
        d["tp_profile"]["owner_coverage"],
    ))

print(
    f"{'case':<24} {'topology':<9} {'owner':<6} {'mode':<6} {'world':>5} {'data':>4} "
    f"{'p50_ms':>10} {'p90_ms':>10} {'local tok/s':>12} {'global tok/s':>13} {'memGB':>8} coverage"
)
print("-" * 150)
for row in rows:
    key, topo, owner, mode, world, data_factor, p50, p90, tok_s, global_tok_s, mem, cov = row
    print(
        f"{key:<24} {topo:<9} {owner:<6} {mode:<6} {world:>5} {data_factor:>4} "
        f"{p50:>10.3f} {p90:>10.3f} {tok_s:>12.1f} {global_tok_s:>13.1f} {mem:>8.2f} {cov}"
    )

print()
print("=== Speedups ===")
topologies = sorted({d["topology"] for d in data.values()})
for topo in topologies:
    sync_key = f"{topo}_lpt_sync"
    async_key = f"{topo}_lpt_async"
    rank0_key = f"{topo}_rank0_sync"
    if sync_key in data and async_key in data:
        sync_ms = data[sync_key]["summary"]["step_ms_p50"]
        async_ms = data[async_key]["summary"]["step_ms_p50"]
        print(f"{topo:<9} async_speedup_vs_sync = {sync_ms / async_ms:.3f}x")
    if sync_key in data and rank0_key in data:
        sync_ms = data[sync_key]["summary"]["step_ms_p50"]
        rank0_ms = data[rank0_key]["summary"]["step_ms_p50"]
        print(f"{topo:<9} lpt_speedup_vs_rank0 = {rank0_ms / sync_ms:.3f}x")
PY

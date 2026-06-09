#!/usr/bin/env bash
# Communication-order smoke for TP post-step dependencies.
#
# This runner intentionally uses test_tp_comm_order.py instead of the loss
# matrix. It splits the optimizer step into internal phases and asserts the
# lifecycle directly: reduce -> TP gather -> Muon full update -> TP scatter ->
# optional HSDP replicate broadcast -> explicit drain.

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
SCRIPT="${HERE}/test_tp_comm_order.py"
OUT="${DMUON_COMM_OUT_DIR:-.pytest_artifacts/tp_comm_order_smoke}"
PORT_BASE="${DMUON_COMM_PORT_BASE:-31600}"
STEPS="${DMUON_COMM_STEPS:-2}"
MODEL="${DMUON_COMM_MODEL:-tiny}"
VISIBLE_4="${DMUON_COMM_VISIBLE_4:-0,1,2,3}"
VISIBLE_8="${DMUON_COMM_VISIBLE_8:-0,1,2,3,4,5,6,7}"

rm -rf "$OUT"
mkdir -p "$OUT"

run_case() {
    local key="$1"
    local visible="$2"
    local nproc="$3"
    local topology="$4"
    local mode="$5"
    local port="$PORT_BASE"
    PORT_BASE=$((PORT_BASE + 1))

    echo "=== ${key}: topology=${topology} mode=${mode} model=${MODEL} world=${nproc} port=${port} ==="
    CUDA_VISIBLE_DEVICES="$visible" \
    DMUON_COMM_TOPOLOGY="$topology" \
    DMUON_COMM_MODE="$mode" \
    DMUON_COMM_MODEL="$MODEL" \
    DMUON_COMM_STEPS="$STEPS" \
    DMUON_COMM_OUT="$OUT/${key}.json" \
    torchrun --nproc_per_node="$nproc" --master_port="$port" "$SCRIPT"
}

run_case tp4_sync "$VISIBLE_4" 4 tp4 sync
run_case tp4_async "$VISIBLE_4" 4 tp4 async
run_case dp_tp2_sync "$VISIBLE_4" 4 dp_tp2 sync
run_case dp_tp2_async "$VISIBLE_4" 4 dp_tp2 async
run_case dp_tp4_sync "$VISIBLE_8" 8 dp_tp4 sync
run_case dp_tp4_async "$VISIBLE_8" 8 dp_tp4 async
run_case hsdp_tp2_sync "$VISIBLE_8" 8 hsdp_tp2 sync
run_case hsdp_tp2_async "$VISIBLE_8" 8 hsdp_tp2 async

echo
echo "PASS: wrote communication-order artifacts to ${OUT}"

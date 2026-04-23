#!/usr/bin/env bash
# Phase A / B driver — runs the alignment instrument 4-6 times and prints
# the noise-floor table.  Default runs sync×2 + async×2 (Phase A).  Pass
# 'drain' as $1 to also run async_drain×2 (Phase B1).
#
# Usage:
#   bash run_tp_alignment.sh           # Phase A only
#   bash run_tp_alignment.sh drain     # Phase A + B1

HERE="$(cd "$(dirname "$0")" && pwd)"
SCRIPT="${HERE}/test_tp_alignment.py"
OUT="/tmp/dmuon_align_$$"
rm -rf "$OUT"; mkdir -p "$OUT"

modes=(sync async)
if [[ "$1" == "drain" ]]; then
    modes+=(async_drain)
fi

port_base=33100
for mode in "${modes[@]}"; do
    for run in 1 2; do
        port=$((port_base++))
        echo "=== $mode run=$run (port=$port) ==="
        DMUON_ALIGN_MODE="$mode" \
        DMUON_ALIGN_RUN="$run" \
        DMUON_ALIGN_OUT="$OUT" \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        torchrun --nproc_per_node=8 --master_port="$port" "$SCRIPT" 2>&1 \
          | grep -E "loss|wrote|SKIP" || true
    done
done

echo
echo "=== Alignment table (rank 0 loss per iter) ==="
python - <<PY
import glob, json, os
out = "$OUT"
data = {}
for fp in sorted(glob.glob(os.path.join(out, "*.json"))):
    with open(fp) as f:
        d = json.load(f)
    data[(d["mode"], d["run_id"])] = d["losses"]

# Print table
keys = sorted(data.keys())
print(f"{'key':<24} | " + " | ".join(f"iter{i}" for i in range(3)))
print("-" * 24 + "-+-" + "-+-".join("-" * 18 for _ in range(3)))
for k in keys:
    losses = data[k]
    print(f"{k[0]}_r{k[1]:<20} | " + " | ".join(f"{l:>18.12f}" for l in losses))

# Gaps
def diff(a, b):
    return max(abs(x - y) for x, y in zip(a, b))

print()
print("=== Noise floor / gaps ===")
if ("sync", "1") in data and ("sync", "2") in data:
    print(f"sync_self          = {diff(data[('sync','1')], data[('sync','2')]):.3e}")
if ("async", "1") in data and ("async", "2") in data:
    print(f"async_self         = {diff(data[('async','1')], data[('async','2')]):.3e}")
if ("sync", "1") in data and ("async", "1") in data:
    print(f"sync_vs_async_gap  = {diff(data[('sync','1')], data[('async','1')]):.3e}")
if ("sync", "1") in data and ("async_drain", "1") in data:
    print(f"sync_vs_drain_gap  = {diff(data[('sync','1')], data[('async_drain','1')]):.3e}")
if ("async", "1") in data and ("async_drain", "1") in data:
    print(f"async_vs_drain_gap = {diff(data[('async','1')], data[('async_drain','1')]):.3e}")
PY

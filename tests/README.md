# Tests

## Directory Structure

```
tests/
├── unit/                           # Single GPU (python)
│   ├── test_partition.py           # Partition algorithm correctness
│   ├── test_syrk.py               # SYRK kernel vs cuBLAS
│   └── test_newton_schulz.py      # Newton-Schulz algorithm
├── distributed/                    # Multi GPU (torchrun)
│   ├── test_multiprocessing.py    # DedicatedParamGroup communication
│   ├── test_reduce_regression.py  # Gradient reduce regression (bug fix)
│   ├── test_muon_step.py          # Muon optimizer step correctness
│   ├── test_e2e_dp.py             # End-to-end DP training
│   ├── test_e2e_tp_dp.py          # End-to-end TP+DP training
│   └── test_correctness.py        # DMuon vs DDP+Muon loss comparison
```

## Running Tests

### Unit tests (single GPU)

```bash
python tests/unit/test_partition.py
python tests/unit/test_syrk.py
python tests/unit/test_newton_schulz.py
```

### Distributed tests (multi GPU)

```bash
torchrun --nproc_per_node=4 tests/distributed/test_multiprocessing.py
torchrun --nproc_per_node=4 tests/distributed/test_reduce_regression.py
torchrun --nproc_per_node=4 tests/distributed/test_muon_step.py
torchrun --nproc_per_node=4 tests/distributed/test_e2e_dp.py
torchrun --nproc_per_node=8 tests/distributed/test_e2e_tp_dp.py   # needs 8 GPUs (DP=4, TP=2)
torchrun --nproc_per_node=4 tests/distributed/test_correctness.py
```

### Run a specific test

Most test files support running individual tests:

```bash
python tests/unit/test_syrk.py diag_add
torchrun --nproc_per_node=4 tests/distributed/test_reduce_regression.py single
```

### Ascend tests (single NPU)

Use a matched PyTorch, `torch_npu`, and CANN environment. The native suite uses
one visible NPU and a world-size-one HCCL group; it does not load the transfer
shim. It checks matrix dtypes/shapes, clipping result types, independent AdamW
step counters, gradient accumulation, asynchronous publication, stream lifetime,
and checkpoint restoration through the public package APIs.

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
DMUON_BUILD_FAST_CLIP=0 pip install -e . --no-build-isolation
pytest tests/unit/test_device_runtime.py tests/unit/test_import_isolation.py \
    tests/unit/test_nested_input_cast.py tests/unit/test_grad_clip_result_contract.py
ASCEND_RT_VISIBLE_DEVICES=0 python tests/native/ascend_snapshot.py \
    --pathway fsdp2 --output /path/to/new-fsdp2.pt
ASCEND_RT_VISIBLE_DEVICES=0 python tests/native/ascend_snapshot.py \
    --pathway ddp --output /path/to/new-ddp.pt
```

For a backend migration, run the same snapshot utility in a separate process
against a reference with the same optimizer dtype policy using `--package-root`.
If that package requires
a legacy compatibility bootstrap, pass its existing file with
`--legacy-bootstrap`; keep it out of the new package process. Run the two
processes sequentially on the same NPU, and compare corresponding snapshots:

```bash
DMUON_NPU_REFERENCE=/path/to/old-fsdp2.pt \
DMUON_NPU_SNAPSHOT=/path/to/new-fsdp2.pt \
pytest tests/native/test_ascend_snapshot.py
```

The comparison requires exact tensor values, dtypes, shapes, result fields and
optimizer state. Use the corresponding DDP snapshots for the DDP check.

Exact comparisons require matching Muon momentum policies. Historical snapshots
with BF16 momentum are not an exact reference for a package using FP32 momentum.

### Ascend distributed restart tests (four NPUs)

The [contribution guide](../docs/contributing.md#native-ascend-distributed-tests-requires-4-npus)
defines the mandatory native gate. Run it before a PR that changes NPU
distributed behavior:

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
    tests/distributed/test_hsdp_correctness.py --device npu --steps 4 \
    --resume-after 2 --output-dir /path/to/new-hsdp-proof
```

It uses the upstream model with identical initialization and actual inputs
across a 1D shard-only mesh and a 2x2 HSDP mesh, checks every rank's full state
without tolerance, and verifies continuation from one canonical rank-0 disk
checkpoint. This is separate from the varied-input regression below.

The tiny HCCL regression covers a one-dimensional four-rank mesh and a 2x2
hybrid mesh. Each topology tests GAS4 with synchronized/default gradients and
`no_sync` with explicit FP32 gradients, followed by an exact disk restart.
It performs 28 global optimizer calls in total. The checkpoint retains each
rank's exported state to preserve replica-local collective rounding:

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
torchrun --standalone --nproc_per_node=4 tests/native/ascend_distributed.py \
    --package-root . --topology all --output-dir /path/to/new-restart-proof
```

The proof compares live momentum, AdamW moments, master parameters, model
parameters, scheduler, RNG, and data cursor within each execution path.
Synchronized and `no_sync` paths are not asserted to share a numerical trajectory.

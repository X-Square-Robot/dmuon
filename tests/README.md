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

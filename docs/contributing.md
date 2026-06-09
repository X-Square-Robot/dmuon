# Contributing

!!! tip "TL;DR"
    Issues, bug fixes, and new adapter integrations are welcome.  Please open a
    GitHub issue first for major changes so we can discuss scope before you
    invest time writing code.  All distributed changes must pass the
    bit-identical correctness test (`test_hsdp_correctness.py`).

---

## Dev setup

```bash
# Clone and install in editable mode
git clone https://github.com/StarrickLiu/dmuon
cd dmuon

# Create environment (conda or venv)
conda create -n dmuon-dev python=3.11 && conda activate dmuon-dev

# Install PyTorch (adjust CUDA version as needed)
pip install "torch>=2.6" --index-url https://download.pytorch.org/whl/cu121

# Install DMuon and dev dependencies
pip install -e ".[dev]"

```

The `[dev]` extra installs test, lint, packaging, and documentation tools,
including `pytest`, `ruff`, `build`, `mkdocs-material`, and
`mkdocstrings[python]`. Release checks that need `twine` can install
`pip install -e ".[release]"`.

---

## Running tests

### Unit tests

```bash
python -m pytest tests/unit/ -v
```

### Distributed tests (requires 4 GPUs)

```bash
# Bit-identical correctness — run before every PR with distributed changes
torchrun --nproc_per_node=4 tests/distributed/test_hsdp_correctness.py
```

Distributed tests take ~2 minutes per run.  Write a small isolated unit test
first to verify your logic before using the distributed harness as a debug loop
(see the project memory note on smoke tests).

### Single-test smoke check pattern

```python
# Minimal isolated test — write this BEFORE modifying distributed logic
import torch
# ... 10-line setup without torchrun ...
```

---

## Coding style

- **Formatter:** `ruff format` (configured in `pyproject.toml`)
- **Linter:** `ruff check` (configured in `pyproject.toml`)
- **Type hints:** required on all public functions and class methods
- **Docstrings:** NumPy style (configured in mkdocstrings); concise —
  one line summary, Args, Returns
- **No emojis** in code, docstrings, or comments

Run style checks:

```bash
ruff format dmuon/ examples/
ruff check dmuon/ examples/
```

Run both before sending changes for review.

---

## PR checklist

Before opening a pull request:

- [ ] Rebase on `main` (`git fetch origin && git rebase origin/main`)
- [ ] Unit tests pass (`pytest tests/unit/`)
- [ ] For distributed changes: `test_hsdp_correctness.py` passes
- [ ] Public API changes have updated docstrings
- [ ] Docs updated if behavior or API surface changed
- [ ] **Bit-identical correctness maintained** — no change to loss values for
  the same random seed across 1D shard-only, HSDP, and checkpoint-resume runs

For new features, open an issue first to align on scope.

---

## Architecture orientation

Read [Design / Architecture](design/architecture.md) before touching the
core ownership machinery (`dmuon/api.py`, `dmuon/_core/`,
`dmuon/_backends/`).  The key invariants are:

- `DedicatedParam._owned_data` is the single source of truth for the parameter
  value on the owner rank
- Hook registration order (pre-forward → post-forward → pre-backward →
  post-backward) must match FSDP2's hook ordering
- Every async collective must have a matching drain path (either in the
  next forward's `_pre_forward_wait` or in `wait_all_replicate_broadcasts`)

---

## Release process

DMuon follows semantic versioning: `MAJOR.MINOR.PATCH`.

- **Patch:** bug fixes, docstring/docs updates, no API change
- **Minor:** new public API symbols, new backends, new guide pages
- **Major:** breaking API changes (rare; require discussion)

Documentation is deployed automatically to GitHub Pages on every push to
`main` via GitHub Actions.

---

## Community

- **Bug reports and feature requests:** [GitHub Issues](https://github.com/StarrickLiu/dmuon/issues)
- **Design discussions:** [GitHub Discussions](https://github.com/StarrickLiu/dmuon/discussions)
- No Discord at this time

Please search existing issues before opening a new one.

---

## See also

- [Design / Architecture](design/architecture.md)
- [FAQ](faq/index.md)
- [Troubleshooting](troubleshooting.md)

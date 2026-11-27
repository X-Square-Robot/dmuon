"""Per-shape kernel handles and tile selection for the 950DT."""

from __future__ import annotations

import contextlib
import threading
from collections import OrderedDict
from collections.abc import Mapping

_cache_lock = threading.Lock()
_kernel_cache: "OrderedDict[tuple, object]" = OrderedDict()
_KERNEL_CACHE_MAX = 256
_kernel_context = threading.local()
# Cross-thread scope depth: any thread inside a capture/pinned scope makes
# every dispatch take the full resolution path.  The counter is a hint --
# the full path re-checks the thread-local state -- but keeping it at zero
# in steady state preserves the original per-call cost of cached_kernel,
# which the optimizer step (host-enqueue saturated) pays for every wrapper
# call (measured 126 ns -> 1120 ns without this guard, -1% end to end).
_scope_depth = 0


class KernelHandles(Mapping):
    """An immutable-after-warmup, bounded owner of compiled kernel handles."""

    def __init__(self, max_entries):
        if type(max_entries) is not int or not 1 <= max_entries <= 4096:
            raise ValueError("max_entries must be an integer in [1,4096]")
        self._handles = {}
        self.max_entries = max_entries
        self._sealed = False

    def __getitem__(self, key):
        return self._handles[key]

    def __iter__(self):
        return iter(self._handles)

    def __len__(self):
        return len(self._handles)

    def _add(self, key, kernel):
        if self._sealed:
            raise RuntimeError("Cannot modify a sealed kernel handle set")
        if key not in self._handles and len(self._handles) >= self.max_entries:
            raise RuntimeError("Kernel warmup exceeded its explicit handle budget")
        self._handles[key] = kernel


def _remember_kernel(key, kernel):
    for handles in getattr(_kernel_context, "collectors", ()):
        handles._add(key, kernel)
    return kernel


@contextlib.contextmanager
def capture_kernel_handles(*, max_entries=256):
    """Collect every cached/built kernel used by this thread during warmup.

    Nested collectors all see the calls.  Retain the returned bounded mapping
    alongside its graph; dropping that graph entry releases these references.
    This is host bookkeeping and does not initiate NPUGraph capture.
    """
    global _scope_depth
    handles = KernelHandles(max_entries)
    previous = getattr(_kernel_context, "collectors", ())
    _kernel_context.collectors = (*previous, handles)
    _scope_depth += 1
    try:
        yield handles
    finally:
        _kernel_context.collectors = previous
        _scope_depth -= 1
        handles._sealed = True


@contextlib.contextmanager
def use_pinned_kernel_handles(handles, *, strict=True):
    """Resolve wrappers from retained handles, independent of the global LRU.

    Strict misses raise before a wrapper can compile.  Nested scopes form a
    union, with the innermost scope taking precedence for a duplicate key.
    Scopes are thread-local and never modify another thread's dispatch.
    """
    if not isinstance(handles, KernelHandles) or not handles._sealed:
        raise TypeError("Use a completed capture_kernel_handles context")
    if type(strict) is not bool:
        raise TypeError("strict must be a bool")
    global _scope_depth
    previous = getattr(_kernel_context, "pinned", ())
    _kernel_context.pinned = (*previous, (handles, strict))
    _scope_depth += 1
    try:
        yield handles
    finally:
        _kernel_context.pinned = previous
        _scope_depth -= 1


def kernel_cache():
    """LRU cache of compiled tilelang kernels keyed by shape signature."""
    return _kernel_cache


def cached_kernel(key):
    # Retained (pinned) handle sets resolve first, so a captured plan never
    # depends on global LRU state; strict scopes reject unknown keys instead
    # of compiling inside a capture.  The scope depth guard keeps the
    # no-scope steady state at the original cost of this function.
    if _scope_depth:
        pinned = getattr(_kernel_context, "pinned", ())
        for handles, _ in reversed(pinned):
            kernel = handles._handles.get(key)
            if kernel is not None:
                return _remember_kernel(key, kernel)
        if any(strict for _, strict in pinned):
            raise RuntimeError(f"Kernel was not collected before capture: {key!r}")
        k = _kernel_cache.get(key)
        if k is not None:
            return _remember_kernel(key, k)
        with _cache_lock:
            k = _kernel_cache.get(key)
        return _remember_kernel(key, k) if k is not None else None
    k = _kernel_cache.get(key)
    if k is not None:
        return k
    with _cache_lock:
        return _kernel_cache.get(key)


def store_kernel(key, kernel):
    with _cache_lock:
        _kernel_cache[key] = kernel
        _kernel_cache.move_to_end(key)
        while len(_kernel_cache) > _KERNEL_CACHE_MAX:
            _kernel_cache.popitem(last=False)
    if _scope_depth:
        _remember_kernel(key, kernel)
    return kernel


# ---------------------------------------------------------------------------
_NUM_CORES = 32  # persistent blocks; refined during autotune


def _fit_pow2(v: int, lo: int, hi: int) -> int:
    t = hi
    while t > lo and v % t != 0:
        t //= 2
    return t


def fit_tiles(m: int, n: int, k: int, dtype: str = "float16") -> dict:
    """Tile config whose tiles exactly divide (m, n, k).

    The ascend copy lowering rejects symbolic tail extents, so divisibility
    is a hard requirement; callers must zero-pad dims to 16-multiples first
    (mainstream training shapes are 256-multiples and skip padding).

    Tile caps keep the Z3 loop scheduler's shared-scope (UB+L1) budget
    satisfied: 64x64 output tiles with fp32 acc_ub/c_ub/o_ub and 2-stage
    pipelining stay well inside the ~216 KiB UB budget; exceeding it makes
    the scheduler fail or (worse) emit a broken schedule.
    """
    # Cube kernels carry no UB epilogue buffers (fp32 GM direct store), so
    # the budget is L1-only: (A+B fp16 tiles, double-buffered) <= ~192 KiB.
    # The 128x128x64 probe measured 182 TFLOPS fp16 on 950DT.
    tm = _fit_pow2(m, 16, 128)
    tn = _fit_pow2(n, 16, 128)
    tk = _fit_pow2(k, 16, 256)
    while (tm + tn) * tk * 2 * 2 > 192 * 1024 and (tk > 16 or tm > 16 or tn > 16):
        if tk > 64:
            tk //= 2
        elif tn > 16:
            tn //= 2
        elif tm > 16:
            tm //= 2
        elif tk > 16:
            tk //= 2
    return {
        "tile_m": tm,
        "tile_n": tn,
        "tile_k": tk,
        "num_stages": 2,
        "num_blocks": min(_NUM_CORES, max(1, (m // tm) * (n // tn))),
    }

"""Per-shape CUDA-graph replay for the Newton-Schulz hot path.

Motivation (2026-08-06): one NS invocation issues ~25-30 kernels through
python (syrk dispatch, addmm, elementwise), and a rank steps ~50 muon
params -> >1000 launches with python between them, which is the bulk of
DMuon's CPU-serial section.  Shapes repeat heavily (11-20 unique shapes
per rank), so we capture NS once per input shape and replay.

Design constraints this honors:
  * ``_reduced_grad`` is a fresh ``clone()`` every step -> graphs never
    reference it; callers copy into a persistent ``stage_in`` buffer.
  * The update is written to a persistent ``update_out`` buffer that the
    caller must consume (``owned.add_``) before the next replay of the
    SAME shape -- true in ``_step_muon_params``'s sequential loop.
  * lr / wd / momentum never enter the graph (they stay in the caller's
    eager pre/post steps), so schedule changes need no re-capture.
  * All graphs share one memory pool; capture happens lazily on the
    second call per shape (the first call runs eagerly and doubles as
    the warmup that triggers autotune / cuBLAS workspace / quack JIT --
    capturing those would either fail or bake cold paths).

Failure policy: capture errors log loudly once and permanently fall back
to eager for that shape (correctness first; the win is additive).
"""

from __future__ import annotations

import logging
import os
from typing import Callable, Optional

import torch
from torch import Tensor

_logger = logging.getLogger(__name__)


def enabled() -> bool:
    """Default ON (set DMUON_NS_CUDA_GRAPH=0 to opt out).

    Validated bitwise-identical to eager across shapes/configs; capture
    failures degrade loudly per shape. Cost of default-on: a one-time
    ~1s capture at the first steps, and the assumption that NS input
    shapes are stable across steps (true for standard training; dynamic
    shape churn would keep re-capturing -- opt out in that case).
    """
    return os.environ.get("DMUON_NS_CUDA_GRAPH", "1") != "0"


class _ShapeGraph:
    __slots__ = ("stage_in", "update_out", "graph", "warmed", "dead")

    def __init__(self):
        self.stage_in: Optional[Tensor] = None
        self.update_out: Optional[Tensor] = None
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.warmed = False
        self.dead = False  # capture failed -> permanent eager fallback


class NSGraphCache:
    """One instance per NewtonSchulz backend object."""

    def __init__(self):
        self._graphs: dict[tuple, _ShapeGraph] = {}
        self._pool = None  # shared mempool across all shape graphs

    def run(self, ns_fn: Callable[[Tensor], Tensor], ns_input: Tensor) -> Tensor:
        """Return NS(ns_input), via graph replay when possible.

        ``ns_fn`` must be a pure function of its tensor argument (steps /
        coefficients closed over).  The returned tensor is either a fresh
        eager result or the shape's persistent ``update_out`` buffer --
        consume it before the next call with the same shape.
        """
        key = (tuple(ns_input.shape), ns_input.dtype, ns_input.device.index)
        sg = self._graphs.get(key)
        if sg is None:
            sg = self._graphs[key] = _ShapeGraph()

        if sg.dead:
            return ns_fn(ns_input)

        if sg.graph is not None:
            sg.stage_in.copy_(ns_input)
            sg.graph.replay()
            return sg.update_out

        if not sg.warmed:
            # First call per shape: eager (runs autotune / workspace init /
            # JIT that must not be captured).
            sg.warmed = True
            return ns_fn(ns_input)

        # Second call: capture.
        try:
            sg.stage_in = ns_input.clone()
            if self._pool is None:
                self._pool = torch.cuda.graph_pool_handle()
            graph = torch.cuda.CUDAGraph()
            # Side-stream warmup run per torch capture protocol.
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                ns_fn(sg.stage_in)
            torch.cuda.current_stream().wait_stream(side)
            with torch.cuda.graph(graph, pool=self._pool):
                sg.update_out = ns_fn(sg.stage_in)
            sg.graph = graph
            _logger.info(
                "[NSGraph] captured shape=%s dtype=%s", key[0], ns_input.dtype
            )
            sg.stage_in.copy_(ns_input)
            graph.replay()
            return sg.update_out
        except Exception as exc:  # noqa: BLE001 -- loud degrade, never crash step
            sg.dead = True
            sg.graph = None
            sg.stage_in = None
            sg.update_out = None
            _logger.warning(
                "[NSGraph] capture FAILED for shape=%s (%s: %s); "
                "permanent eager fallback for this shape",
                key[0],
                type(exc).__name__,
                exc,
            )
            return ns_fn(ns_input)

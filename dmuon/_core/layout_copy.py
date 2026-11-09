"""Batched, bounded-CTA copies between native and optimizer matrix layouts.

Maps are metadata only. The torch reference uses strided views for affine
regions and bounded index tiles for the general case, never a packed parameter
temporary. CUDA consumes the same maps in one launch on the caller's stream.
Version-2 extensions specialize 16-bit copies and BF16 widening, using packed
shared-memory tiles for even transposes. Other casts, accumulation and odd
transpose dimensions retain the general path. Version-3 extensions additionally
vectorize contiguous FP32-to-BF16/FP16 copy-in. The default budget is 96 CTAs,
selected using full decoder training on a 132-SM device; each launch is also
capped by the device SM count and available tiles. The environment override
remains useful for other devices and workloads.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property, lru_cache
from importlib import import_module
import math
import os

import torch


def _strides(shape):
    return tuple(math.prod(shape[i + 1 :]) for i in range(len(shape)))


@dataclass(frozen=True)
class IndexTransform:
    sizes: tuple[int, ...]
    strides: tuple[int, ...]
    offset: int = 0

    def apply(self, index):
        rest, result = index, self.offset
        for size, stride in zip(reversed(self.sizes), reversed(self.strides)):
            result = result + (rest % size) * stride
            rest = rest // size
        return result


@dataclass(frozen=True)
class MatrixMap:
    shape: tuple[int, int]
    offset: int
    start: int = 0
    transforms: tuple[IndexTransform, ...] = ()
    # Factorized logical shape and native strides, when a zero-copy view exists.
    region: IndexTransform | None = None

    @property
    def numel(self):
        return math.prod(self.shape)

    @property
    def contiguous_offset(self):
        r = self.region
        if r is not None and all(
            n == 1 or s == c for n, s, c in zip(r.sizes, r.strides, _strides(r.sizes))
        ):
            return r.offset
        return None

    def indices(self, index):
        if self.region is not None:
            return self.region.apply(index)
        index = index + self.start
        for transform in self.transforms:
            index = transform.apply(index)
        return index + self.offset

    def native_view(self, tensor):
        if self.region is None or not tensor.is_contiguous():
            return None
        r = self.region
        return tensor.as_strided(r.sizes, r.strides, tensor.storage_offset() + r.offset)


def _matrix_region(region, numel, index):
    """Select a matrix-sized logical interval without folding strided axes."""
    if region is None:
        return None
    suffix = 1
    for axis in reversed(range(region.ndim)):
        size = region.shape[axis]
        if numel % suffix == 0 and size % (numel // suffix) == 0:
            count = numel // suffix
            shape = (*region.shape[:axis], size // count, count, *region.shape[axis + 1 :])
            region = region.view(shape)
            split = axis + 1
            offset = region.storage_offset()
            for dim, stride in zip(reversed(shape[:split]), reversed(region.stride()[:split])):
                offset += (index % dim) * stride
                index //= dim
            return IndexTransform(tuple(shape[split:]), tuple(region.stride()[split:]), offset)
        suffix *= size
        if suffix > numel:
            break
    return None


def compile_matrix_maps(layout):
    """Lower the resolved operation tree into independent matrix address maps."""
    if not layout.roots:
        return tuple(
            MatrixMap(
                m.shape, m.offset, region=IndexTransform(m.shape, _strides(m.shape), m.offset)
            )
            for m in layout.matrices
        )
    result = []
    native = torch.empty(layout.source_shape, device="meta").view(-1)

    def walk(node, path, region, root_offset):
        if node.kind == "leaf":
            numel = math.prod(node.shape[-2:])
            for index in range(math.prod(node.shape[:-2])):
                result.append(
                    MatrixMap(
                        tuple(node.shape[-2:]),
                        root_offset,
                        index * numel,
                        tuple(reversed(path)),
                        _matrix_region(region, numel, index),
                    )
                )
        elif node.kind == "reshape":
            if region is not None:
                try:
                    child_region = region.view(node.dimensions)
                except RuntimeError:
                    # Retain the factorized region across trailing reshapes.
                    child_region = region
            else:
                child_region = None
            walk(node.children[0], path, child_region, root_offset)
        elif node.kind == "permute":
            p = node.dimensions
            transform = IndexTransform(
                tuple(node.shape[d] for d in p), tuple(_strides(node.shape)[d] for d in p)
            )
            child_region = (
                region.permute(p)
                if region is not None and tuple(region.shape) == node.shape
                else None
            )
            walk(node.children[0], (*path, transform), child_region, root_offset)
        else:
            dim, *sizes = node.dimensions
            start = 0
            for child, size in zip(node.children, sizes):
                transform = IndexTransform(
                    child.shape, _strides(node.shape), start * _strides(node.shape)[dim]
                )
                child_region = (
                    region.narrow(dim, start, size)
                    if region is not None and tuple(region.shape) == node.shape
                    else None
                )
                walk(child, (*path, transform), child_region, root_offset)
                start += size

    for root in layout.roots:
        region = native.narrow(0, root.offset, math.prod(root.shape)).view(root.shape)
        walk(root, (), region, root.offset)
    return tuple(result)


@dataclass(frozen=True)
class CopyTask:
    source: torch.Tensor
    destination: torch.Tensor
    mapping: MatrixMap | None = None
    scatter: bool = False
    add: bool = False

    @property
    def numel(self):
        return self.mapping.numel if self.mapping is not None else self.source.numel()

    @cached_property
    def address_map(self):
        m = self.mapping
        if m is None:
            shape = tuple(self.source.shape)
            return MatrixMap((1, self.numel), 0, region=IndexTransform(shape, _strides(shape)))
        native = self.destination if self.scatter else self.source
        if native.is_contiguous():
            return m
        # Map the native tensor's logical flat address to its actual strides.
        transform = IndexTransform(tuple(native.shape), tuple(native.stride()))
        if m.region is not None:
            before = (m.region,)
            return MatrixMap(m.shape, 0, transforms=(*before, transform))
        # Root offsets must be added before interpreting the native shape.
        shift = IndexTransform((math.prod(native.shape),), (1,), m.offset)
        return MatrixMap(m.shape, 0, m.start, (*m.transforms, shift, transform))

    def check(self):
        s, d = self.source, self.destination
        if s.device != d.device:
            raise ValueError("layout copy requires source and destination on the same device")
        if not s.is_floating_point() or not d.is_floating_point():
            raise ValueError("layout copy requires floating-point tensors")
        canonical = s if self.scatter else d
        if not canonical.is_contiguous():
            raise ValueError("The optimizer-side copy tensor must be contiguous")
        if self.mapping is None:
            if not s.is_contiguous() or not d.is_contiguous() or s.numel() != d.numel():
                raise ValueError("Unmapped copies require equal, contiguous tensor sizes")
        elif canonical.numel() != self.numel:
            raise ValueError("Matrix copy destination/source shape differs from its map")


_DTYPES = {torch.float32: 0, torch.float16: 1, torch.bfloat16: 2}
_TILE = 4096
_TRANSPOSE_TILE = 64


def _copy_kind(mapping):
    region = mapping.region
    if mapping.contiguous_offset is not None:
        return 2
    if (
        region is not None
        and len(region.sizes) == 2
        and region.strides[0] == 1
        and region.strides[1] >= region.sizes[0]
        and mapping.shape == region.sizes
    ):
        return 1
    if region is not None and region.strides[-1] == 1 and region.sizes[-1] >= 32:
        return 3
    return 0


def _copy_dispatch(tasks, kinds, device, api_version):
    """Versioned acceleration; old installed extensions keep the old tile plan.

    Fast dispatch bits: 256=FP32 narrowing (2=BF16, otherwise FP16),
    128=raw16, 1=scatter, 2=BF16->FP32, 4=mixed,
    8=contains transpose, 16=256x256 tile (otherwise 128x128).
    """
    transpose = 1 in kinds
    fallback = (int(transpose), _TRANSPOSE_TILE)
    if api_version < 2 or any(t.add or t.scatter != tasks[0].scatter for t in tasks):
        return fallback
    if (
        api_version >= 3
        and all(kind == 2 for kind in kinds)
        and all(
            t.source.dtype == torch.float32 and t.destination.dtype == tasks[0].destination.dtype
            for t in tasks
        )
        and tasks[0].destination.dtype in (torch.float16, torch.bfloat16)
    ):
        return 256 | (2 if tasks[0].destination.dtype == torch.bfloat16 else 0), 0
    copy16 = all(
        t.source.dtype == t.destination.dtype and t.source.dtype in (torch.float16, torch.bfloat16)
        for t in tasks
    )
    widen = all(
        t.source.dtype == torch.bfloat16 and t.destination.dtype == torch.float32 for t in tasks
    )
    if not copy16 and not widen:
        return fallback
    maps = [t.address_map for t, kind in zip(tasks, kinds) if kind == 1]
    if any(
        any(n % 2 for n in m.shape) or m.region.offset % 2 or m.region.strides[1] % 2 for m in maps
    ):
        return fallback
    mixed = not all(kind == 1 for kind in kinds)
    large = False
    if transpose and not mixed and not widen and all(min(m.shape) >= 256 for m in maps):
        properties = torch.cuda.get_device_properties(device)
        large = getattr(properties, "shared_memory_per_block_optin", 0) >= 128 * 129 * 8
    dispatch = (
        128
        | int(tasks[0].scatter)
        | (2 if widen else 0)
        | (4 if mixed else 0)
        | (8 if transpose else 0)
        | (16 if large else 0)
    )
    return dispatch, 256 if large else 128


@lru_cache(maxsize=1)
def _extension():
    try:
        return import_module("dmuon._layout_copy_cuda")
    except ImportError:
        return None


def _options(device=None, dtypes=()):
    backend = os.environ.get("DMUON_COPY_BACKEND", "auto")
    if backend not in ("auto", "cuda", "torch"):
        raise ValueError("DMUON_COPY_BACKEND must be auto, cuda or torch")
    budget = int(os.environ.get("DMUON_COPY_MAX_CTAS", "96"))
    if budget <= 0:
        raise ValueError("DMUON_COPY_MAX_CTAS must be positive")
    if backend == "cuda" and device is not None:
        if device.type != "cuda" or any(d not in _DTYPES for d in dtypes) or _extension() is None:
            raise RuntimeError(
                "CUDA layout copy unavailable: CUDA device, fp32/fp16/bf16 and extension required"
            )
    return backend, budget


def _storage_view(tensor):
    span = 1 + sum((n - 1) * s for n, s in zip(tensor.shape, tensor.stride()))
    return tensor.as_strided((span,), (1,))


def _reference_copy(task):
    s, d, m = task.source, task.destination, task.mapping
    if m is None:
        source, target = s.view(-1), d.view(-1)
    else:
        native = d if task.scatter else s
        region = m.native_view(native)
        if region is not None:
            canonical = (s if task.scatter else d).view(region.shape)
            source, target = (canonical, region) if task.scatter else (region, canonical)
        else:
            # General fallback is bounded by one tile, never a full parameter.
            src, dst = _storage_view(s), _storage_view(d)
            for start in range(0, task.numel, _TILE):
                j = torch.arange(start, min(start + _TILE, task.numel), device=s.device)
                address = task.address_map.indices(j)
                si, di = (j, address) if task.scatter else (address, j)
                value = src[si].to(d.dtype)
                dst[di] = dst[di] + value if task.add else value
            return
    if task.add:
        # Match cast-then-add semantics, including low-precision rounding.
        target.add_(source.to(target.dtype))
    else:
        target.copy_(source)


class CopyExecutor:
    """Per-group plan cache and async metadata lifetime; no global synchronizes."""

    def __init__(self, *, device=None, dtypes=()):
        self._device = torch.device(device) if device is not None else None
        self._dtypes = tuple(d for d in dtypes if d is not None)
        # Validate on every group rank, including ranks with no local owner
        # copy-in tasks. Strict-mode failures must precede its collectives.
        _options(self._device, self._dtypes)
        self._plans = {}
        self._pending = []
        self.last_timing = None
        self.last_stats = {"backend": "none", "tasks": 0, "launches": 0, "ctas": 0}

    @torch.no_grad()
    def run(self, tasks, *, validate=True):
        """Dispatch independent tasks; runtime-owned buffers may skip validation.

        validate=False is reserved for group allocations whose shapes/devices
        were established by the compiled layout and immutable policy.
        """
        self._pending = [p for p in self._pending if not p[0].query()]
        self.last_timing = None
        backend, budget = _options(self._device, self._dtypes)
        tasks = tuple(tasks)
        if validate:
            for task in tasks:
                task.check()
        tasks = tuple(t for t in tasks if t.numel)
        if not tasks:
            self.last_stats = {"backend": "none", "tasks": 0, "launches": 0, "ctas": 0}
            return
        device = tasks[0].source.device
        if validate and any(t.source.device != device for t in tasks):
            raise ValueError("A copy batch must use one device")
        eligible = device.type == "cuda" and all(
            t.source.dtype in _DTYPES and t.destination.dtype in _DTYPES for t in tasks
        )
        ext = _extension() if backend != "torch" and eligible else None
        if backend == "cuda" and ext is None:
            raise RuntimeError(
                "CUDA layout copy requested but its optional extension/device is unavailable"
            )
        if ext is None:
            plain = [t for t in tasks if t.mapping is None and not t.add]
            if plain:
                torch._foreach_copy_(
                    [t.destination.view(-1) for t in plain], [t.source.view(-1) for t in plain]
                )
            for task in tasks:
                if task.mapping is not None or task.add:
                    _reference_copy(task)
            self.last_stats = {
                "backend": "torch",
                "tasks": len(tasks),
                "launches": None,
                "ctas": None,
            }
            return
        self._run_cuda(ext, tasks, device, budget)

    def _run_cuda(self, ext, tasks, device, budget):
        api_version = getattr(ext, "api_version", 0)
        key = (
            device,
            api_version,
            tuple(
                (t.numel, t.source.dtype, t.destination.dtype, t.address_map, t.scatter, t.add)
                for t in tasks
            ),
        )
        stream = torch.cuda.current_stream(device)
        plan = self._plans.get(key)
        if plan is None:
            descriptors, program, tile_end = [], [], 0
            kinds = [_copy_kind(t.address_map) for t in tasks]
            dispatch, transpose_tile = _copy_dispatch(tasks, kinds, device, api_version)
            for t, kind in zip(tasks, kinds):
                m, r = t.address_map, t.address_map.region
                transpose = kind == 1
                tiles = (
                    ((m.shape[0] + transpose_tile - 1) // transpose_tile)
                    * ((m.shape[1] + transpose_tile - 1) // transpose_tile)
                    if kind == 1
                    else (t.numel + _TILE - 1) // _TILE
                )
                tile_end += tiles
                op_begin = len(program)
                ops = (r,) if r is not None else m.transforms
                for op in ops:
                    program.extend((len(op.sizes), op.offset, *op.sizes, *op.strides))
                # No pointer is cached here: resize/lease can change addresses.
                descriptors.append(
                    (
                        t.numel,
                        _DTYPES[t.source.dtype],
                        _DTYPES[t.destination.dtype],
                        int(t.add),
                        2 if t.scatter else 1,
                        m.offset if r is None else 0,
                        m.start if r is None else 0,
                        op_begin,
                        len(ops),
                        tile_end,
                        kind,
                        m.shape[0],
                        m.shape[1],
                        r.offset if r is not None else 0,
                        r.strides[1] if transpose else 0,
                        r.sizes[-1] if r is not None and r.sizes and r.strides[-1] == 1 else 0,
                    )
                )
            desc_host = torch.tensor(descriptors, dtype=torch.int64, device="cpu", pin_memory=True)
            prog_host = torch.tensor(program, dtype=torch.int64, device="cpu", pin_memory=True)
            desc = desc_host.to(device, non_blocking=True)
            prog = prog_host.to(device, non_blocking=True)
            ready = stream.record_event()
            plan = (desc, prog, tile_end, ready, desc_host, prog_host, dispatch)
            self._plans[key] = plan
        desc, prog, tile_count, ready, _desc_host, _prog_host, dispatch = plan
        stream.wait_event(ready)
        desc.record_stream(stream)
        prog.record_stream(stream)
        # Pinned host storage stays alive until its async upload and launch end.
        host = torch.tensor(
            [(task.source.data_ptr(), task.destination.data_ptr()) for task in tasks],
            dtype=torch.int64,
            device="cpu",
            pin_memory=True,
        )
        for task in tasks:
            task.source.record_stream(stream)
            task.destination.record_stream(stream)
        pointers = host.to(device, non_blocking=True)
        ctas = min(
            budget, tile_count, torch.cuda.get_device_properties(device).multi_processor_count
        )
        start = None
        if os.environ.get("DMUON_COPY_PROFILE", "0") == "1":
            start = torch.cuda.Event(enable_timing=True)
            start.record(stream)
        ext.copy(desc, prog, pointers, tile_count, ctas, dispatch)
        done = torch.cuda.Event(enable_timing=start is not None)
        done.record(stream)
        self.last_timing = (start, done) if start is not None else None
        # record_stream protects the large tensor allocations. Retaining the
        # task objects here would keep full native gradients alive until the
        # group's next use, defeating rolling backward buffer reclamation.
        self._pending.append((done, host, pointers, plan))
        fast = bool(dispatch & 128)
        tiled = False if dispatch & 256 else bool(dispatch & 8) if fast else bool(dispatch)
        self.last_stats = {
            "backend": "cuda",
            "tasks": len(tasks),
            "launches": 1,
            "ctas": ctas,
            "tiled": tiled,
            "path": "fp32-bf16"
            if dispatch & 256 and dispatch & 2
            else "fp32-fp16"
            if dispatch & 256
            else "packed16-widen"
            if dispatch & 128 and dispatch & 2
            else "packed16"
            if dispatch & 128
            else "general",
            "threads": 512 if dispatch & (128 | 256) else 256,
            "transpose_tile": (256 if dispatch & 16 else 128)
            if fast and tiled
            else 64
            if tiled
            else 0,
        }

"""Declarative optimizer matrix boundaries for an unchanged model parameter.

After all layout operations, the final two dimensions of each branch describe
one NS matrix; leading dimensions index independent updates. Layouts describe
optimizer semantics and never register model-side modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
import hashlib
import json
import math
import weakref

import torch
from torch import nn


@dataclass(frozen=True)
class MatrixSlice:
    """A matrix's interval in the packed optimizer buffer, in elements.

    This is also its source interval only for a contiguous compiled layout.
    """

    index: int
    offset: int
    shape: tuple[int, int]

    @property
    def numel(self) -> int:
        return math.prod(self.shape)


@dataclass(frozen=True)
class _LayoutNode:
    """Resolved, invertible operations; metadata size is independent of numel."""

    shape: tuple[int, ...]
    kind: str
    dimensions: tuple[int, ...] = ()
    children: tuple[_LayoutNode, ...] = ()
    offset: int = 0

    def state(self):
        return (
            self.shape,
            self.kind,
            self.dimensions,
            tuple(child.state() for child in self.children),
            self.offset,
        )

    def pack(self, source, destination):
        if self.kind == "leaf":
            destination.narrow(0, self.offset, source.numel()).view(self.shape).copy_(source)
        elif self.kind == "reshape":
            child = self.children[0]
            if child.kind == "leaf":
                # Reshape preserves logical element order. Copy directly into
                # a matching view of the packed destination, even when source
                # strides would make Tensor.reshape allocate a full temporary.
                destination.narrow(0, child.offset, source.numel()).view(self.shape).copy_(source)
            else:
                child.pack(source.reshape(self.dimensions), destination)
        elif self.kind == "permute":
            self.children[0].pack(source.permute(self.dimensions), destination)
        else:  # split; compile guarantees complete, disjoint coverage
            dim, *sizes = self.dimensions
            for child, part in zip(self.children, source.split(sizes, dim=dim)):
                child.pack(part, destination)

    def unpack(self, packed):
        if self.kind == "leaf":
            return packed.narrow(0, self.offset, math.prod(self.shape)).view(self.shape)
        if self.kind == "reshape":
            return self.children[0].unpack(packed).reshape(self.shape)
        if self.kind == "permute":
            inverse = tuple(self.dimensions.index(axis) for axis in range(len(self.shape)))
            return self.children[0].unpack(packed).permute(inverse)
        return torch.cat([child.unpack(packed) for child in self.children], dim=self.dimensions[0])

    def unpack_into(self, packed, destination):
        """Scatter into strided views whenever possible, without concatenation."""
        if self.kind == "leaf":
            destination.copy_(self.unpack(packed))
        elif self.kind == "permute":
            self.children[0].unpack_into(packed, destination.permute(self.dimensions))
        elif self.kind == "split":
            dim, *sizes = self.dimensions
            for child, part in zip(self.children, destination.split(sizes, dim=dim)):
                child.unpack_into(packed, part)
        else:  # reshape
            try:
                view = destination.view(self.dimensions)
            except RuntimeError:
                # The inverse can often view the packed data directly even
                # when the destination's strides cannot be folded together.
                destination.copy_(self.unpack(packed))
            else:
                self.children[0].unpack_into(packed, view)


@dataclass(frozen=True)
class CompiledMatrixLayout:
    source_shape: tuple[int, ...]
    matrices: tuple[MatrixSlice, ...]
    # Each root covers a contiguous source interval. Empty means the legacy
    # direct-slice path, whose buffer order is already the original order.
    roots: tuple[_LayoutNode, ...] = ()

    @property
    def is_contiguous(self) -> bool:
        return not self.roots or all(
            mapping.contiguous_offset == matrix.offset
            for mapping, matrix in zip(self.copy_maps, self.matrices)
        )

    @cached_property
    def copy_maps(self):
        from ._core.layout_copy import compile_matrix_maps

        return compile_matrix_maps(self)

    @property
    def signature(self) -> str:
        payload = (self.source_shape, tuple((m.offset, m.shape) for m in self.matrices))
        if self.roots:
            payload += (tuple(root.state() for root in self.roots),)
        return hashlib.sha256(json.dumps(payload).encode()).hexdigest()[:16]

    def state(self) -> dict:
        result = {
            "version": 2 if self.roots else 1,
            "source_shape": self.source_shape,
            "matrices": tuple((m.offset, m.shape) for m in self.matrices),
        }
        if self.roots:
            result["mapping"] = tuple(root.state() for root in self.roots)
        return result

    def pack(self, source: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        """Collect the original tensor into contiguous optimizer matrix order."""
        if tuple(source.shape) != self.source_shape:
            raise ValueError("Layout pack source shape differs from the compiled parameter")
        if out is None and self.is_contiguous:
            return source.reshape(-1)
        if out is None:
            out = source.new_empty(math.prod(self.source_shape))
        self._check_buffer(out)
        if self.is_contiguous:
            out.copy_(source.reshape(-1))
        else:
            flat = source.reshape(-1)
            for root in self.roots:
                part = flat.narrow(0, root.offset, math.prod(root.shape)).view(root.shape)
                root.pack(part, out)
        return out

    def unpack(self, packed: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        """Restore original parameter order; also used for model checkpoints."""
        self._check_buffer(packed)
        if out is None and self.is_contiguous:
            return packed.view(self.source_shape)
        if out is None:
            out = packed.new_empty(self.source_shape)
        if tuple(out.shape) != self.source_shape or not out.is_contiguous():
            raise ValueError("Layout unpack destination must be contiguous with the source shape")
        if self.is_contiguous:
            out.copy_(packed.view(self.source_shape))
        else:
            flat = out.view(-1)
            for root in self.roots:
                destination = flat.narrow(0, root.offset, math.prod(root.shape)).view(root.shape)
                root.unpack_into(packed, destination)
        return out

    def _check_buffer(self, buffer):
        if (
            buffer.ndim != 1
            or buffer.numel() != math.prod(self.source_shape)
            or not buffer.is_contiguous()
        ):
            raise ValueError(
                "Layout packed buffer must be contiguous, flat and cover the parameter"
            )


def _resolve_shape(dimensions, numel):
    dims = list(dimensions)
    if (
        not dims
        or any(not isinstance(d, int) or isinstance(d, bool) for d in dims)
        or dims.count(-1) > 1
        or any(d == 0 or d < -1 for d in dims)
    ):
        raise ValueError(
            "reshape/view needs positive integer dimensions and at most one inferred -1"
        )
    if -1 in dims:
        known = math.prod(d for d in dims if d != -1)
        if numel % known:
            raise ValueError("reshape/view cannot infer an integral dimension")
        dims[dims.index(-1)] = numel // known
    if math.prod(dims) != numel:
        raise ValueError(
            "matrix_layout reshape/view must exactly cover the original parameter or branch"
        )
    return tuple(dims)


@dataclass(frozen=True)
class MatrixLayout:
    """Pure configuration, compiled against a selected original parameter.

    ``whole()`` uses the existing rows-by-flattened-columns NS interpretation.
    ``view(E, 2, I, H)`` gives 2E matrices of shape (I, H).
    ``rows(q, k, v)`` partitions rows of each trailing 2D matrix into Q/K/V.
    Chained ``permute``, ``reshape`` and ``split`` run before matrix boundaries
    are determined. After ``split``, operations apply to each branch; branches
    are enumerated in split order, then by their leading dimensions.
    Unlike Tensor.view, these declarations can require copies at runtime.
    No layout is inferred from parameter names or model classes.
    """

    kind: str
    dimensions: tuple[int, ...] = ()
    operations: tuple[tuple[str, tuple[int, ...]], ...] = ()

    @classmethod
    def whole(cls) -> MatrixLayout:
        return cls("whole")

    @classmethod
    def view(cls, *shape: int) -> MatrixLayout:
        return cls("view", tuple(shape))

    @classmethod
    def rows(cls, *sizes: int) -> MatrixLayout:
        return cls("rows", tuple(sizes))

    def permute(self, *dims: int) -> MatrixLayout:
        """Reorder axes in every current branch, before defining NS matrices."""
        return self._append("permute", dims)

    def reshape(self, *shape: int) -> MatrixLayout:
        """Reshape each branch independently, allowing one inferred dimension."""
        return self._append("reshape", shape)

    def split(self, sizes: tuple[int, ...], *, dim: int = 0) -> MatrixLayout:
        """Partition an axis completely; later operations apply to each branch."""
        return self._append("split", (dim, *sizes))

    def _append(self, kind, dimensions):
        return MatrixLayout(
            self.kind, self.dimensions, self.operations + ((kind, tuple(dimensions)),)
        )

    def compile(self, shape: tuple[int, ...]) -> CompiledMatrixLayout:
        source = tuple(int(d) for d in shape)
        if len(source) < 2 or any(d <= 0 for d in source):
            raise ValueError(
                "matrix_layout requires a parameter with at least two positive dimensions"
            )
        total = math.prod(source)
        if any(not isinstance(d, int) or isinstance(d, bool) for d in self.dimensions):
            raise ValueError("matrix_layout dimensions must be integers")
        if self.kind == "whole":
            if self.dimensions:
                raise ValueError("whole() does not accept dimensions")
            shapes = [(source[0], math.prod(source[1:]))]
        elif self.kind == "view":
            dims = _resolve_shape(self.dimensions, total)
            if len(dims) < 2 and not self.operations:
                raise ValueError("view() needs at least two final dimensions")
            shapes = [dims] if self.operations else [tuple(dims[-2:])] * math.prod(dims[:-2])
        elif self.kind == "rows":
            if not self.dimensions or any(d <= 0 for d in self.dimensions):
                raise ValueError("rows() needs positive partition sizes")
            if sum(self.dimensions) != source[-2]:
                raise ValueError("rows() must exactly cover the parameter's row axis")
            shapes = [
                (r, source[-1]) for _ in range(math.prod(source[:-2])) for r in self.dimensions
            ]
        else:
            raise ValueError(f"Unknown matrix_layout kind: {self.kind!r}")
        if self.operations:
            # rows() retains its existing expert-then-part enumeration.
            return self._compile_operations(source, shapes)
        offset = 0
        matrices = []
        for index, matrix_shape in enumerate(shapes):
            matrix = MatrixSlice(index, offset, matrix_shape)
            matrices.append(matrix)
            offset += matrix.numel
        return CompiledMatrixLayout(source, tuple(matrices))

    def _compile_operations(self, source, root_shapes):
        matrices = []
        packed_offset = 0

        def build(shape, operations):
            nonlocal packed_offset
            if not operations:
                if len(shape) < 2:
                    raise ValueError("matrix_layout needs at least two final dimensions per branch")
                start = packed_offset
                for _ in range(math.prod(shape[:-2])):
                    spec = MatrixSlice(len(matrices), packed_offset, tuple(shape[-2:]))
                    matrices.append(spec)
                    packed_offset += spec.numel
                return _LayoutNode(tuple(shape), "leaf", offset=start)
            (kind, dimensions), *remaining = operations
            if any(not isinstance(d, int) or isinstance(d, bool) for d in dimensions):
                raise ValueError(f"matrix_layout {kind} requires integer dimensions")
            if kind == "reshape":
                resolved = _resolve_shape(dimensions, math.prod(shape))
                children = (build(resolved, remaining),)
            elif kind == "permute":
                rank = len(shape)
                resolved = tuple(d + rank if d < 0 else d for d in dimensions)
                if sorted(resolved) != list(range(rank)):
                    raise ValueError("matrix_layout permute must contain every axis exactly once")
                children = (build(tuple(shape[d] for d in resolved), remaining),)
            elif kind == "split":
                dim, *sizes = dimensions
                dim = dim + len(shape) if dim < 0 else dim
                if not 0 <= dim < len(shape) or not sizes or any(s <= 0 for s in sizes):
                    raise ValueError("matrix_layout split needs a valid axis and positive sizes")
                if sum(sizes) != shape[dim]:
                    raise ValueError("matrix_layout split must exactly cover its axis")
                resolved = (dim, *sizes)
                children = tuple(
                    build((*shape[:dim], size, *shape[dim + 1 :]), remaining) for size in sizes
                )
            else:
                raise ValueError(f"Unknown matrix_layout operation: {kind!r}")
            return _LayoutNode(tuple(shape), kind, tuple(resolved), children)

        roots = []
        source_offset = 0
        for shape in root_shapes:
            root = build(tuple(shape), self.operations)
            # The root offset addresses the source; leaf offsets address packed
            # storage. Every root has an operation, so these cannot collide.
            roots.append(
                _LayoutNode(root.shape, root.kind, root.dimensions, root.children, source_offset)
            )
            source_offset += math.prod(shape)
        return CompiledMatrixLayout(source, tuple(matrices), tuple(roots))


@dataclass(frozen=True, eq=False, init=False)
class LogicalMatrix:
    """Planner identity; contains metadata, not an additional Parameter."""

    _source_ref: object = field(repr=False)
    _dtype: object = field(repr=False)
    _device: object = field(repr=False)
    fqn: str
    layout: CompiledMatrixLayout
    matrix: MatrixSlice

    def __init__(
        self, source: nn.Parameter, fqn: str, layout: CompiledMatrixLayout, matrix: MatrixSlice
    ):
        # Returning a plan must not keep the original full expert storage alive.
        for name, value in {
            "_source_ref": weakref.ref(source),
            "_dtype": source.dtype,
            "_device": source.device,
            "fqn": fqn,
            "layout": layout,
            "matrix": matrix,
        }.items():
            object.__setattr__(self, name, value)

    @property
    def source(self):
        return self._source_ref()

    @property
    def shape(self):
        return self.matrix.shape

    @property
    def ndim(self):
        return 2

    @property
    def dtype(self):
        return self._dtype

    @property
    def device(self):
        return self._device

    def numel(self):
        return self.matrix.numel


def matrix_state_key(fqn: str, layout: CompiledMatrixLayout, matrix: MatrixSlice) -> str:
    return f"{fqn}::matrix[{matrix.index}]@{layout.signature}"

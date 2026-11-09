"""Runtime for logical matrices belonging to unchanged original parameters."""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager, ExitStack
import math
import weakref

import torch
from torch import nn
import torch.distributed as dist

from dmuon._core.internal_utils import alloc_storage, free_storage, unsafe_setattr_param
from dmuon._core.device import get_device_module
from dmuon._core.layout_copy import CopyExecutor, CopyTask
from dmuon.matrix_layout import matrix_state_key
from .param import DedicatedParam


@contextmanager
def _storage_write(buffer):
    with torch.no_grad(), torch.autograd._unsafe_preserve_version_counter(buffer):
        yield


class PartitionedParameter:
    """One original model attribute, with several independent optimizer units."""

    def __init__(self, source, fqn, layout, aliases, device):
        self.fqn = fqn
        self.layout = layout
        self.aliases = aliases
        self.source_ref = weakref.ref(source)
        self.units = []
        self._placeholder = nn.Parameter(
            torch.empty(0, dtype=source.dtype, device=device), requires_grad=source.requires_grad
        )
        # Presence marks a DMuon-managed parameter; there is no single owner.
        self._placeholder._dedicated_owner_rank = None
        self._unsharded_param = None

    def bind_buffer(self, buffer):
        self._unsharded_param = nn.Parameter(
            buffer.view(self.layout.source_shape), requires_grad=self._placeholder.requires_grad
        )
        self._unsharded_param._dedicated_owner_rank = None

    def install_placeholder(self):
        for module, name in self.aliases:
            unsafe_setattr_param(module, name, self._placeholder)

    def publish(self):
        for module, name in self.aliases:
            unsafe_setattr_param(module, name, self._unsharded_param)

    def parameter_refs(self):
        refs = [self._placeholder]
        if self._unsharded_param is not None:
            refs.append(self._unsharded_param)
        source = self.source_ref()
        if source is not None:
            refs.append(source)
        return refs


class DedicatedMatrix(DedicatedParam):
    """A 2D update unit reusing native owner storage, reductions and Muon math.

    Only PartitionedParameter publishes/restores the original model attribute.
    """

    def __init__(self, source, parent, spec, **kwargs):
        module, name = parent.aliases[0]
        super().__init__(
            param=nn.Parameter(source, requires_grad=parent._placeholder.requires_grad),
            module=module,
            param_name=name,
            _install_placeholder=False,
            **kwargs,
        )
        self.matrix_parent = parent
        self.matrix_spec = spec
        self._param_fqn = parent.fqn
        self._checkpoint_key = matrix_state_key(parent.fqn, parent.layout, spec)
        self._layout_grad = None
        self._layout_grad_owned = False
        self.module, self.param_name = parent.aliases[0]
        parent.units.append(self)

    def finish_unshard(self):
        self._is_unsharded = True

    def reshard(self):
        self._is_unsharded = False


class MatrixParamGroup:
    """Native slots with batched weight copies and final-buffer gradients."""

    def __init__(self, units, *, dtype, device):
        by_parent = defaultdict(list)
        for unit in units:
            parent = getattr(unit, "matrix_parent", None)
            if parent is not None:
                by_parent[parent].append(unit)
        self.entries = {}
        self.buffers = {}
        self.packed_buffers = {}
        self.copy_in = {}
        self.broadcast_runs = []
        self.copy_executor = CopyExecutor(
            device=device,
            dtypes=(dtype, *(p._grad_dtype or dtype for p in units)),
        )
        for parent, entries in by_parent.items():
            entries.sort(key=lambda unit: unit.matrix_spec.index)
            if [p.matrix_spec.index for p in entries] != list(range(len(parent.layout.matrices))):
                raise ValueError(
                    "All matrices of an original parameter must share a hook/policy group"
                )
            self.entries[parent] = entries
            buffer = torch.empty(math.prod(parent.layout.source_shape), dtype=dtype, device=device)
            self.buffers[parent] = buffer
            parent.bind_buffer(buffer)
            # A leading-axis permutation can reorder whole NS matrices without
            # breaking any matrix's native contiguity. Bind/broadcast those
            # intervals in place too; no native-layout reconstruction is needed.
            direct = all(m.contiguous_offset is not None for m in parent.layout.copy_maps)
            packed = buffer if direct else torch.empty_like(buffer)
            self.packed_buffers[parent] = packed
            by_owner = defaultdict(lambda: ([], []))
            runs = []
            for unit, mapping in zip(entries, parent.layout.copy_maps):
                spec = unit.matrix_spec
                offset = mapping.contiguous_offset if direct else spec.offset
                unit.bind_to_packed_buffer(packed, offset)
                destinations, sources = by_owner[unit.owner_rank]
                destinations.append(packed[offset : offset + spec.numel])
                sources.append(unit)
                runs.append((offset, spec.numel, unit.owner_rank))
            self.copy_in[parent] = by_owner
            merged = []
            for offset, count, owner in sorted(runs):
                if merged and merged[-1][2] == owner and merged[-1][0] + merged[-1][1] == offset:
                    off, previous, _ = merged[-1]
                    merged[-1] = (off, previous + count, owner)
                else:
                    merged.append((offset, count, owner))
            self.broadcast_runs.extend((packed, *run) for run in merged)
            parent.install_placeholder()
            free_storage(buffer)
            if packed is not buffer:
                free_storage(packed)

    @property
    def payload_bytes(self):
        return sum(buffer.numel() * buffer.element_size() for buffer in self.buffers.values())

    def allocate(self):
        for parent, buffer in self.buffers.items():
            with _storage_write(buffer):
                alloc_storage(buffer)
            packed = self.packed_buffers[parent]
            with _storage_write(packed):
                if packed is not buffer:
                    alloc_storage(packed)

    def copy_in_tasks(self, local_shard_rank):
        return [
            CopyTask(unit._owned_data, destination)
            for by_owner in self.copy_in.values()
            for owner, (destinations, units) in by_owner.items()
            if owner[0] == local_shard_rank
            for unit, destination in zip(units, destinations)
        ]

    def broadcast(self, dp_group, global_owner_ranks, device):
        if self.broadcast_runs:
            with dist._coalescing_manager(group=dp_group, device=device):
                for buffer, off, count, owner in self.broadcast_runs:
                    dist.broadcast(
                        buffer[off : off + count], src=global_owner_ranks[owner], group=dp_group
                    )

    def unpack(self):
        """One group-level batch, after all broadcasts and before the ready event."""
        tasks = []
        with ExitStack() as stack:
            for parent, buffer in self.buffers.items():
                packed = self.packed_buffers[parent]
                if packed is buffer:
                    continue
                stack.enter_context(_storage_write(buffer))
                for unit, mapping in zip(self.entries[parent], parent.layout.copy_maps):
                    spec = unit.matrix_spec
                    source = packed.narrow(0, spec.offset, spec.numel).view(spec.shape)
                    tasks.append(CopyTask(source, buffer, mapping, scatter=True))
            self.copy_executor.run(tasks, validate=False)
        for parent, packed in self.packed_buffers.items():
            if packed is not self.buffers[parent]:
                with _storage_write(packed):
                    free_storage(packed)

    def publish(self):
        for parent in self.entries:
            parent.publish()

    def release(self):
        for parent, buffer in self.buffers.items():
            parent.install_placeholder()
            with _storage_write(buffer):
                free_storage(buffer)

    def split_gradients(self):
        """Borrow contiguous slices or copy directly to final reduction inputs.

        Called on reduce_stream after its compute-stream dependency. Source
        gradients are recorded there before clearing the model's .grad.
        """
        tasks = []
        for parent, units in self.entries.items():
            parameter = parent._unsharded_param
            if parameter.grad is None:
                continue
            gradient = parameter.grad
            if gradient.device.type != "cpu":
                gradient.record_stream(
                    get_device_module(gradient.device).current_stream(gradient.device)
                )
            for unit, mapping in zip(units, parent.layout.copy_maps):
                target, spec = unit._unsharded_param, unit.matrix_spec
                if not target.requires_grad:
                    continue
                dtype = unit._grad_dtype or gradient.dtype
                view = mapping.native_view(gradient)
                can_borrow = view is not None and view.is_contiguous() and view.dtype == dtype
                old = unit._layout_grad
                if old is None and can_borrow:
                    unit._layout_grad = view.view(spec.shape).detach()
                else:
                    if old is None:
                        unit._layout_grad = torch.empty(
                            spec.shape, dtype=dtype, device=gradient.device
                        )
                    tasks.append(
                        CopyTask(gradient, unit._layout_grad, mapping, add=old is not None)
                    )
                # These views own their retained source storage and are disjoint;
                # unlike arbitrary borrowed TP views, no_sync can transfer them.
                unit._layout_grad_owned = True
            parameter.grad = None
        self.copy_executor.run(tasks, validate=False)

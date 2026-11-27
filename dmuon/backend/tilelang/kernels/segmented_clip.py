"""Segmented clip kernels: generic pointer jobs, FP32 reductions, device stats.

Each pointer job processes one contiguous chunk.  Host metadata is grouped by
storage dtype so that pointer casts are static and do not branch per element.
The job table, rather than the tensor count, determines grid work distribution.
"""

import tilelang.ascend.language as T

BLOCKS = 64
TILE = 8192


def copy_storage(nbytes: int):
    assert nbytes % 4 == 0
    words = nbytes // 4
    blocks = min(BLOCKS, max(1, (words + 1023) // 1024))
    span = (words + blocks - 1) // blocks

    @T.prim_func
    def main(Addresses: T.Tensor((2,), "int64")):
        with T.Kernel(blocks) as bx:
            source = T.make_tensor_from_addr(Addresses[0], (words,), "uint32")
            dest = T.make_tensor_from_addr(Addresses[1], (words,), "uint32")
            with T.SimtVF(threads=256):
                for offset in T.Parallel(span):
                    i = bx * span + offset
                    if i < words:
                        dest[i] = source[i]

    return main


def norm_jobs(jobs, max_numel, chunk, dtype):
    blocks = min(BLOCKS, jobs)
    steps = (chunk + TILE - 1) // TILE
    # Preserve a typed pointer Bind through TVM simplification. This opaque
    # identity cast is address materialization only; all arithmetic is TileLang.
    cast_name = (
        "reinterpret_cast<__gm__ "
        + {"float32": "float", "float16": "half", "bfloat16": "bfloat16_t"}[dtype]
        + "*>"
    )

    @T.prim_func
    def main(Meta: T.Tensor((jobs, 3), "int64"), Partials: T.Tensor((jobs,), "float32")):
        with T.Kernel(blocks) as bx:
            x = T.alloc_shared((TILE,), dtype)
            sums = T.alloc_shared((steps,), "float32")
            T.annotate_buffer_versions({x: 2})
            for job in T.serial(bx, jobs, blocks):
                ptr = Meta[job, 0]
                begin = Meta[job, 1]
                length = T.cast(Meta[job, 2], "int32")
                data = T.make_tensor_from_addr(
                    T.call_extern("handle", cast_name, ptr), (max_numel,), dtype
                )
                for step in T.Pipelined(steps, num_stages=2):
                    offset = step * TILE
                    n = T.max(T.min(length - offset, TILE), 0)
                    T.copy(data[begin + offset : begin + offset + n], x[:n])
                    with T.SimtVF(threads=256):
                        red = T.alloc_reducer((1,), "float32", op="sum")
                        T.reducer_init(red)
                        for i in T.Parallel(TILE):
                            if i < n:
                                v = T.cast(x[i], "float32")
                                T.reducer_update(red[0], v * v)
                        out = T.alloc_fragment((1,), "float32")
                        T.finalize_reducer(red, out)
                        sums[step] = out[0]
                with T.SimtVF(threads=64):
                    red2 = T.alloc_reducer((1,), "float32", op="sum")
                    T.reducer_init(red2)
                    for s in T.Parallel(steps):
                        T.reducer_update(red2[0], sums[s])
                    out2 = T.alloc_fragment((1,), "float32")
                    T.finalize_reducer(red2, out2)
                    Partials[job] = out2[0]

    return main


def scale_jobs(jobs, max_numel, chunk, dtype, segments):
    blocks = min(BLOCKS, jobs)
    steps = (chunk + TILE - 1) // TILE
    # Preserve a typed pointer Bind through TVM simplification. This opaque
    # identity cast is address materialization only; all arithmetic is TileLang.
    cast_name = (
        "reinterpret_cast<__gm__ "
        + {"float32": "float", "float16": "half", "bfloat16": "bfloat16_t"}[dtype]
        + "*>"
    )

    @T.prim_func
    def main(
        Meta: T.Tensor((jobs, 3), "int64"),
        Segments: T.Tensor((jobs,), "int32"),
        Coefs: T.Tensor((segments,), "float32"),
    ):
        with T.Kernel(blocks) as bx:
            x = T.alloc_shared((TILE,), dtype)
            coefficient = T.alloc_shared((1,), "float32")
            T.annotate_buffer_versions({x: 2})
            for job in T.serial(bx, jobs, blocks):
                ptr = Meta[job, 0]
                begin = Meta[job, 1]
                length = T.cast(Meta[job, 2], "int32")
                segment = Segments[job]
                data = T.make_tensor_from_addr(
                    T.call_extern("handle", cast_name, ptr), (max_numel,), dtype
                )
                T.copy(Coefs[segment : segment + 1], coefficient)
                for step in T.Pipelined(steps, num_stages=2):
                    offset = step * TILE
                    n = T.max(T.min(length - offset, TILE), 0)
                    T.copy(data[begin + offset : begin + offset + n], x[:n])
                    with T.SimtVF(threads=256):
                        for i in T.Parallel(TILE):
                            if i < n:
                                x[i] = T.cast(T.cast(x[i], "float32") * coefficient[0], dtype)
                    T.copy(x[:n], data[begin + offset : begin + offset + n])

    return main


def combine_jobs(jobs: int, segments: int):
    width = min(2048, max(1, jobs))
    steps = max(1, (jobs + width - 1) // width)
    blocks = min(BLOCKS, segments)

    @T.prim_func
    def main(
        Partials: T.Tensor((max(1, jobs),), "float32"),
        Segments: T.Tensor((max(1, jobs),), "int32"),
        Reduce: T.Tensor((segments,), "int32"),
        LocalSq: T.Tensor((segments,), "float32"),
        ReducedSq: T.Tensor((segments,), "float32"),
    ):
        with T.Kernel(blocks) as bx:
            p = T.alloc_shared((width,), "float32")
            seg = T.alloc_shared((width,), "int32")
            sums = T.alloc_shared((steps,), "float32")
            for bucket in T.serial(bx, segments, blocks):
                for step in T.serial(steps):
                    begin = step * width
                    n = T.min(width, jobs - begin)
                    if jobs > 0:
                        T.copy(Partials[begin : begin + n], p[:n])
                        T.copy(Segments[begin : begin + n], seg[:n])
                    with T.SimtVF(threads=64):
                        red = T.alloc_reducer((1,), "float32", op="sum")
                        T.reducer_init(red)
                        for i in T.Parallel(width):
                            if i < n:
                                if seg[i] == bucket:
                                    T.reducer_update(red[0], p[i])
                        out = T.alloc_fragment((1,), "float32")
                        T.finalize_reducer(red, out)
                        sums[step] = out[0]
                with T.SimtVF(threads=64):
                    red2 = T.alloc_reducer((1,), "float32", op="sum")
                    T.reducer_init(red2)
                    for s in T.Parallel(steps):
                        T.reducer_update(red2[0], sums[s])
                    out2 = T.alloc_fragment((1,), "float32")
                    T.finalize_reducer(red2, out2)
                    LocalSq[bucket] = out2[0]
                    ReducedSq[bucket] = T.if_then_else(Reduce[bucket] != 0, out2[0], 0.0)

    return main


def finish_stats(segments: int, has_max: bool, has_reduce: bool):
    @T.prim_func
    def main(
        LocalSq: T.Tensor((segments,), "float32"),
        ReducedSq: T.Tensor((segments,), "float32"),
        Reduce: T.Tensor((segments,), "int32"),
        Norms: T.Tensor((segments,), "float32"),
        Coefs: T.Tensor((segments,), "float32"),
        Clipped: T.Tensor((segments,), "bool"),
        FoundInf: T.Tensor((segments,), "bool"),
        Total: T.Tensor((1,), "float32"),
        max_norm: T.float32,
    ):
        with T.Kernel(1):
            local = T.alloc_shared((segments,), "float32")
            reduced = T.alloc_shared((segments,), "float32")
            flags = T.alloc_shared((segments,), "int32")
            T.copy(LocalSq, local)
            T.copy(ReducedSq, reduced)
            T.copy(Reduce, flags)
            with T.SimtVF(threads=64):
                red = T.alloc_reducer((1,), "float32", op="sum")
                T.reducer_init(red)
                for i in T.Parallel(segments):
                    sq = T.if_then_else(has_reduce and flags[i] != 0, reduced[i], local[i])
                    norm = T.sqrt(sq)
                    # Float self-comparisons are simplified under fast-math.
                    # IEEE exponent bits preserve NaN/Inf classification.
                    norm_abs_bits = T.reinterpret(norm, "uint32") & T.uint32(0x7FFFFFFF)
                    finite = norm_abs_bits < T.uint32(0x7F800000)
                    coef = T.float32(1.0)
                    if has_max:
                        raw = max_norm / (norm + 1.0e-6)
                        # min instructions may suppress NaN: select explicitly.
                        raw_abs_bits = T.reinterpret(raw, "uint32") & T.uint32(0x7FFFFFFF)
                        coef = T.if_then_else(
                            raw_abs_bits > T.uint32(0x7F800000), raw, T.min(raw, 1.0)
                        )
                    Norms[i] = norm
                    Coefs[i] = coef
                    Clipped[i] = finite and has_max and norm > max_norm
                    FoundInf[i] = not finite
                    T.reducer_update(red[0], norm * norm)
                total = T.alloc_fragment((1,), "float32")
                T.finalize_reducer(red, total)
                Total[0] = T.sqrt(total[0])

    return main

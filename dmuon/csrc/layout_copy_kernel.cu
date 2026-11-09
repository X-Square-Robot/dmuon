#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/Half.h>
#include <c10/util/BFloat16.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <type_traits>
#include <unordered_set>

namespace {
constexpr int kFields = 16;
constexpr int kThreads = 256;
constexpr int kTranspose = 64;
constexpr int64_t kTile = 4096;

__device__ __forceinline__ int64_t coordinate(int64_t& index, int64_t size) {
  if ((size & (size - 1)) == 0) {
    const int64_t value = index & (size - 1);
    index >>= __ffsll(size) - 1;
    return value;
  }
  // Matrix-local indices usually fit 32 bits even when expert/global offsets
  // do not. Avoid software 64-bit division in that common case.
  if (index <= UINT32_MAX && size <= UINT32_MAX) {
    const uint32_t q = static_cast<uint32_t>(index) / static_cast<uint32_t>(size);
    const int64_t value = index - static_cast<int64_t>(q) * size;
    index = q;
    return value;
  }
  const int64_t value = index % size;
  index /= size;
  return value;
}

__device__ __forceinline__ float load(uint64_t ptr, int dtype, int64_t i) {
  if (dtype == 0) return reinterpret_cast<const float*>(ptr)[i];
  if (dtype == 1) return static_cast<float>(reinterpret_cast<const c10::Half*>(ptr)[i]);
  return static_cast<float>(reinterpret_cast<const c10::BFloat16*>(ptr)[i]);
}

__device__ __forceinline__ void store(uint64_t ptr, int dtype, int64_t i, float value, bool add) {
  // Cast before adding: preserve the reference's low-precision rounding order.
  if (add) {
    if (dtype == 1) value = static_cast<float>(static_cast<c10::Half>(value));
    if (dtype == 2) value = static_cast<float>(static_cast<c10::BFloat16>(value));
    value += load(ptr, dtype, i);
  }
  if (dtype == 0) reinterpret_cast<float*>(ptr)[i] = value;
  else if (dtype == 1) reinterpret_cast<c10::Half*>(ptr)[i] = static_cast<c10::Half>(value);
  else reinterpret_cast<c10::BFloat16*>(ptr)[i] = static_cast<c10::BFloat16>(value);
}

__device__ __forceinline__ void copy_chunk(uint64_t src, uint64_t dst, int sd, int dd,
                                            int64_t si, int64_t di, int count, bool add) {
  const int bytes = sd == 0 ? 4 : 2;
  const int width = sd == dd ? 16 / bytes : 4;
  if (!add && sd == dd && count == width &&
      ((src + si * bytes) | (dst + di * bytes)) % 16 == 0) {
    *reinterpret_cast<uint4*>(dst + di * bytes) =
        *reinterpret_cast<const uint4*>(src + si * bytes);
  } else {
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
      if (k < count) store(dst, dd, di + k, load(src, sd, si + k), add);
    }
  }
}

__device__ __forceinline__ int64_t address(int64_t index, const int64_t* task,
                                          const int64_t* program) {
  index += task[6];
  const int64_t* op = program + task[7];
  for (int count = 0; count < task[8]; ++count) {
    const int rank = static_cast<int>(op[0]);
    int64_t offset = op[1];
    for (int axis = rank - 1; axis > 0; --axis) {
      offset += coordinate(index, op[2 + axis]) * op[2 + rank + axis];
    }
    if (rank > 0) offset += index * op[2 + rank];
    index = offset;
    op += 2 + 2 * rank;
  }
  return index + task[5];
}

template <bool BFloat>
__device__ __forceinline__ uint32_t narrow_pair(float a, float b) {
  if constexpr (BFloat) {
#if __CUDA_ARCH__ >= 800
    const __nv_bfloat162_raw value = __floats2bfloat162_rn(a, b);
    return uint32_t(value.x) | (uint32_t(value.y) << 16);
#else
    return uint32_t(c10::BFloat16(a).x) | (uint32_t(c10::BFloat16(b).x) << 16);
#endif
  } else {
    const __half2_raw value = __floats2half2_rn(a, b);
    return uint32_t(value.x) | (uint32_t(value.y) << 16);
  }
}

// Contiguous FP32 master weights to the existing 16-bit communication buffer.
// Each lane loads eight floats and stores four packed pairs. Pointer alignment is
// checked at launch time because buffers can be reallocated between calls.
template <bool BFloat, int Threads = 512>
__global__ __launch_bounds__(Threads) void narrow_copy_kernel(
    const int64_t* __restrict__ descriptors, const int64_t* __restrict__ pointers,
    int tasks, int64_t total_tiles) {
  int64_t first = 0, cached_end = 0, count = 0;
  const float* source = nullptr;
  uint16_t* destination = nullptr;
  bool aligned_load = false, aligned_store = false, wide_store = false;
  for (int64_t tile = blockIdx.x; tile < total_tiles; tile += gridDim.x) {
    if (tile >= cached_end) {
      int lo = 0, hi = tasks;
      while (lo < hi) {
        const int mid = lo + (hi - lo) / 2;
        if (tile < descriptors[mid * kFields + 9]) hi = mid;
        else lo = mid + 1;
      }
      const int64_t* task = descriptors + lo * kFields;
      first = lo == 0 ? 0 : descriptors[(lo - 1) * kFields + 9];
      cached_end = task[9];
      count = task[0];
      const bool scatter = task[4] == 2;
      source = reinterpret_cast<const float*>(pointers[lo * 2]) + (scatter ? 0 : task[13]);
      destination = reinterpret_cast<uint16_t*>(pointers[lo * 2 + 1]) + (scatter ? task[13] : 0);
      aligned_load = (reinterpret_cast<uintptr_t>(source) & 15) == 0;
      aligned_store = (reinterpret_cast<uintptr_t>(destination) & 7) == 0;
      wide_store = (reinterpret_cast<uintptr_t>(destination) & 15) == 0;
    }
    const int64_t begin = (tile - first) * kTile, end = min(begin + kTile, count);
    for (int64_t j = begin + threadIdx.x * 8; j < end; j += Threads * 8) {
      if (j + 8 <= end) {
        float4 values[2];
        uint2 packed[2];
        // Independent loads expose more memory-level parallelism per warp.
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
          const float* at = source + j + i * 4;
          values[i] = aligned_load ? *reinterpret_cast<const float4*>(at)
                                   : make_float4(at[0], at[1], at[2], at[3]);
        }
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
          const auto value = values[i];
          packed[i] = make_uint2(narrow_pair<BFloat>(value.x, value.y),
                                 narrow_pair<BFloat>(value.z, value.w));
        }
        if (wide_store) {
          *reinterpret_cast<uint4*>(destination + j) =
              make_uint4(packed[0].x, packed[0].y, packed[1].x, packed[1].y);
        } else if (aligned_store) {
          *reinterpret_cast<uint2*>(destination + j) = packed[0];
          *reinterpret_cast<uint2*>(destination + j + 4) = packed[1];
        } else {
          #pragma unroll
          for (int i = 0; i < 2; ++i) {
            auto* at = destination + j + i * 4;
            at[0] = packed[i].x; at[1] = packed[i].x >> 16;
            at[2] = packed[i].y; at[3] = packed[i].y >> 16;
          }
        }
      } else {
        for (int k = 0; k < 8 && j+k < end; ++k) {
          if constexpr (BFloat) destination[j+k] = c10::BFloat16(source[j+k]).x;
          else destination[j+k] = c10::Half(source[j+k]).x;
        }
      }
    }
  }
}

// Raw 16-bit copies and exact BF16->FP32 widening. Other casts/add operations
// retain the general kernel below. No floating arithmetic is needed here.
template <bool Widen>
using Copy16Output = typename std::conditional<Widen, uint32_t, uint16_t>::type;

template <bool Widen>
__device__ __forceinline__ void copy16_run(const uint16_t* src,
    Copy16Output<Widen>* dst, int64_t si, int64_t di, int count) {
  if (count == 8 && ((reinterpret_cast<uintptr_t>(src + si) |
                      reinterpret_cast<uintptr_t>(dst + di)) & 15) == 0) {
    const uint4 v = *reinterpret_cast<const uint4*>(src + si);
    if constexpr (Widen) {
      *reinterpret_cast<uint4*>(dst + di) = make_uint4(
          (v.x & 0xffff) << 16, v.x & 0xffff0000,
          (v.y & 0xffff) << 16, v.y & 0xffff0000);
      *reinterpret_cast<uint4*>(dst + di + 4) = make_uint4(
          (v.z & 0xffff) << 16, v.z & 0xffff0000,
          (v.w & 0xffff) << 16, v.w & 0xffff0000);
    } else *reinterpret_cast<uint4*>(dst + di) = v;
  } else {
    #pragma unroll
    for (int i = 0; i < 8; ++i) if (i < count) {
      if constexpr (Widen) dst[di+i] = uint32_t(src[si+i]) << 16;
      else dst[di+i] = src[si+i];
    }
  }
}

template <bool Widen>
__device__ __forceinline__ void store16_pair(Copy16Output<Widen>* dst,
                                            int64_t at, uint32_t bits) {
  if constexpr (Widen) {
    const uint32_t a = (bits & 0xffff) << 16, b = bits & 0xffff0000;
    if ((reinterpret_cast<uintptr_t>(dst + at) & 7) == 0)
      *reinterpret_cast<uint64_t*>(dst + at) = uint64_t(a) | (uint64_t(b) << 32);
    else { dst[at] = a; dst[at+1] = b; }
  } else {
    if ((reinterpret_cast<uintptr_t>(dst + at) & 3) == 0)
      *reinterpret_cast<uint32_t*>(dst + at) = bits;
    else { dst[at] = bits; dst[at+1] = bits >> 16; }
  }
}

template <bool Scatter, bool Widen, bool Mixed, bool CanTranspose, int Packed>
__global__ __launch_bounds__(512) void layout16_kernel(
    const int64_t* __restrict__ descriptors, const int64_t* __restrict__ program,
    const int64_t* __restrict__ pointers, int tasks, int64_t total_tiles) {
  constexpr int Threads = 512, Side = Packed*2, RowWorkers = Threads/Packed;
  constexpr int Items = Packed/RowWorkers;
  extern __shared__ uint64_t packed_shared[];
  auto* shared = reinterpret_cast<uint64_t (*)[Packed+1]>(packed_shared);
  const int x = threadIdx.x%Packed, y = threadIdx.x/Packed;
  int64_t cached_end = 0, first = 0, in_rows = 0, in_cols = 0;
  int64_t in_ld = 0, out_ld = 0, col_tiles = 0;
  const int64_t* t = nullptr;
  const uint16_t* source = nullptr;
  Copy16Output<Widen>* destination = nullptr;
  const uint16_t* input = nullptr;
  Copy16Output<Widen>* output = nullptr;
  for (int64_t tile = blockIdx.x; tile < total_tiles; tile += gridDim.x) {
    if (tile >= cached_end) {
      int lo = 0, hi = tasks;
      while (lo < hi) {
        const int mid = lo+(hi-lo)/2;
        if (tile < descriptors[mid*kFields+9]) hi = mid;
        else lo = mid+1;
      }
      t = descriptors+lo*kFields;
      first = lo == 0 ? 0 : descriptors[(lo-1)*kFields+9];
      cached_end = t[9];
      source = reinterpret_cast<const uint16_t*>(pointers[lo*2]);
      destination = reinterpret_cast<Copy16Output<Widen>*>(pointers[lo*2+1]);
      if constexpr (CanTranspose) {
        in_rows = Scatter ? t[11] : t[12]; in_cols = Scatter ? t[12] : t[11];
        in_ld = Scatter ? t[12] : t[14]; out_ld = Scatter ? t[14] : t[12];
        col_tiles = (in_cols+Side-1)/Side;
        input = source+(Scatter ? 0 : t[13]);
        output = destination+(Scatter ? t[13] : 0);
      }
    }
    if constexpr (Mixed) {
      if (!CanTranspose || t[10] != 1) {
        const int64_t begin = (tile-first)*kTile, end = min(begin+kTile,t[0]);
        if (t[10] == 2) {
          for (int64_t j = begin+threadIdx.x*8; j < end; j += Threads*8)
            copy16_run<Widen>(source,destination,Scatter ? j : t[13]+j,
                Scatter ? t[13]+j : j,static_cast<int>(min(int64_t(8),end-j)));
        } else if (t[10] == 3) {
          const int64_t inner = t[15];
          int64_t first_row = begin, last_row = end-1;
          coordinate(first_row,inner); coordinate(last_row,inner);
          const int workers = static_cast<int>(min(last_row-first_row+1,int64_t(Threads/32)));
          const int lanes = 32*max(1,(Threads/32)/workers);
          const int row_worker = threadIdx.x/lanes, lane = threadIdx.x%lanes;
          for (int64_t row = first_row+row_worker; row <= last_row; row += Threads/lanes) {
            const int64_t base = row*inner, native = address(base,t,program);
            const int64_t c0 = max(int64_t(0),begin-base), c1 = min(inner,end-base);
            for (int64_t c = c0+lane*8; c < c1; c += lanes*8)
              copy16_run<Widen>(source,destination,Scatter ? base+c : native+c,
                  Scatter ? native+c : base+c,static_cast<int>(min(int64_t(8),c1-c)));
          }
        } else {
          for (int64_t j = begin+threadIdx.x; j < end; j += Threads) {
            const int64_t at = address(j,t,program), si = Scatter ? j : at, di = Scatter ? at : j;
            if constexpr (Widen) destination[di] = uint32_t(source[si]) << 16;
            else destination[di] = source[si];
          }
        }
        continue;
      }
    }
    if constexpr (CanTranspose) {
      int64_t row_tile = tile-first;
      const int64_t c0 = coordinate(row_tile,col_tiles)*Side, r0 = row_tile*Side;
      uint64_t values[Items];
      #pragma unroll
      for (int i = 0; i < Items; ++i) {
        const int64_t r = r0+2*(y+i*RowWorkers), c = c0+2*x;
        uint32_t a = 0, b = 0;
        if (r < in_rows && c < in_cols) {
          const int64_t at = r*in_ld+c;
          if ((reinterpret_cast<uintptr_t>(input) & 3) == 0) {
            a = *reinterpret_cast<const uint32_t*>(input+at);
            b = *reinterpret_cast<const uint32_t*>(input+at+in_ld);
          } else {
            a = uint32_t(input[at]) | (uint32_t(input[at+1]) << 16);
            b = uint32_t(input[at+in_ld]) | (uint32_t(input[at+in_ld+1]) << 16);
          }
        }
        values[i] = uint64_t(a) | (uint64_t(b) << 32);
      }
      #pragma unroll
      for (int i = 0; i < Items; ++i) shared[y+i*RowWorkers][x] = values[i];
      __syncthreads();
      #pragma unroll
      for (int i = 0; i < Items; ++i) values[i] = shared[x][y+i*RowWorkers];
      __syncthreads();
      #pragma unroll
      for (int i = 0; i < Items; ++i) {
        const int64_t r = r0+2*x, c = c0+2*(y+i*RowWorkers);
        if (r < in_rows && c < in_cols) {
          const uint64_t bits = values[i];
          const uint32_t a = uint32_t(bits & 0xffff) | uint32_t((bits >> 16) & 0xffff0000);
          const uint32_t b = uint32_t((bits >> 16) & 0xffff) | uint32_t((bits >> 32) & 0xffff0000);
          const int64_t at = c*out_ld+r;
          store16_pair<Widen>(output,at,a); store16_pair<Widen>(output,at+out_ld,b);
        }
      }
    }
  }
}

template <bool Scatter, bool Widen, bool Mixed, bool Transpose, int Packed>
void launch_layout16(torch::Tensor d, torch::Tensor program, torch::Tensor p,
                     int64_t tiles, int blocks, cudaStream_t stream) {
  constexpr int bytes = Transpose ? Packed*(Packed+1)*sizeof(uint64_t) : 0;
  if constexpr (bytes > 48*1024) {
    static thread_local std::unordered_set<int> configured;
    if (configured.find(d.get_device()) == configured.end()) {
      C10_CUDA_CHECK(cudaFuncSetAttribute(layout16_kernel<Scatter,Widen,Mixed,Transpose,Packed>,
          cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));
      configured.insert(d.get_device());
    }
  }
  layout16_kernel<Scatter,Widen,Mixed,Transpose,Packed><<<blocks,512,bytes,stream>>>(
      d.data_ptr<int64_t>(),program.data_ptr<int64_t>(),p.data_ptr<int64_t>(),d.size(0),tiles);
}

template <bool Scatter, bool Widen>
void choose_layout16(torch::Tensor d, torch::Tensor program, torch::Tensor p,
                     int64_t tiles, int blocks, int dispatch, cudaStream_t stream) {
  if (!(dispatch & 8)) launch_layout16<Scatter,Widen,true,false,64>(d,program,p,tiles,blocks,stream);
  else if (dispatch & 4) launch_layout16<Scatter,Widen,true,true,64>(d,program,p,tiles,blocks,stream);
  else if constexpr (!Widen) {
    if (dispatch & 16) launch_layout16<Scatter,Widen,false,true,128>(d,program,p,tiles,blocks,stream);
    else launch_layout16<Scatter,Widen,false,true,64>(d,program,p,tiles,blocks,stream);
  } else launch_layout16<Scatter,Widen,false,true,64>(d,program,p,tiles,blocks,stream);
}

template <bool CanTranspose>
__global__ void layout_copy_kernel(const int64_t* __restrict__ descriptors,
                                    const int64_t* __restrict__ program,
                                    const int64_t* __restrict__ pointers,
                                    int tasks, int64_t tiles) {
  // The non-transpose instantiation has no tile allocation or register tile.
  extern __shared__ uint32_t shared[];
  auto* transposed = reinterpret_cast<uint32_t (*)[kTranspose + 1]>(shared);
  for (int64_t tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
    int lo = 0, hi = tasks;
    while (lo < hi) {
      const int mid = lo + (hi - lo) / 2;
      if (tile < descriptors[mid * kFields + 9]) hi = mid;
      else lo = mid + 1;
    }
    const int id = lo;
    const auto* t = descriptors + id * kFields;
    const int64_t first = id == 0 ? 0 : descriptors[(id - 1) * kFields + 9];
    const int64_t local_tile = tile - first;
    const uint64_t src = static_cast<uint64_t>(pointers[id * 2]);
    const uint64_t dst = static_cast<uint64_t>(pointers[id * 2 + 1]);
    const int sd = static_cast<int>(t[1]), dd = static_cast<int>(t[2]);
    const bool add = t[3] != 0, scatter = t[4] == 2;
    if (CanTranspose && t[10] == 1) {
      const int64_t rows = t[11], cols = t[12], ld = t[14], offset = t[13];
      const int64_t col_tiles = (cols + kTranspose - 1) / kTranspose;
      int64_t row_tile = local_tile;
      const int64_t c0 = coordinate(row_tile, col_tiles) * kTranspose;
      const int64_t r0 = row_tile * kTranspose;
      const int x = threadIdx.x % kTranspose, y = threadIdx.x / kTranspose;
      constexpr int items = kTranspose / (kThreads / kTranspose);
      uint32_t values[items];
      #pragma unroll
      for (int i = 0; i < items; ++i) {
        const int v = y + i * (kThreads / kTranspose);
        const int64_t r = r0 + (scatter ? v : x);
        const int64_t c = c0 + (scatter ? x : v);
        values[i] = 0;
        if (r < rows && c < cols) {
          const int64_t at = scatter ? r * cols + c : offset + r + c * ld;
          values[i] = sd == dd && !add
              ? (sd == 0 ? reinterpret_cast<const uint32_t*>(src)[at]
                         : reinterpret_cast<const uint16_t*>(src)[at])
              : __float_as_uint(load(src, sd, at));
        }
      }
      #pragma unroll
      for (int i = 0; i < items; ++i) {
        const int v = y + i * (kThreads / kTranspose);
        if (scatter) transposed[v][x] = values[i];
        else transposed[x][v] = values[i];
      }
      __syncthreads();
      #pragma unroll
      for (int i = 0; i < items; ++i) {
        const int v = y + i * (kThreads / kTranspose);
        values[i] = scatter ? transposed[x][v] : transposed[v][x];
      }
      // Finish shared reads before reuse, without waiting on the following
      // global stores. Register tiles expose multiple memory requests per warp.
      __syncthreads();
      #pragma unroll
      for (int i = 0; i < items; ++i) {
        const int v = y + i * (kThreads / kTranspose);
        const int64_t r = r0 + (scatter ? x : v);
        const int64_t c = c0 + (scatter ? v : x);
        if (r < rows && c < cols) {
          const int64_t at = scatter ? offset + r + c * ld : r * cols + c;
          const uint32_t bits = values[i];
          if (sd == dd && !add) {
            if (dd == 0) reinterpret_cast<uint32_t*>(dst)[at] = bits;
            else reinterpret_cast<uint16_t*>(dst)[at] = static_cast<uint16_t>(bits);
          } else store(dst, dd, at, __uint_as_float(bits), add);
        }
      }
    } else if (t[10] == 3) {
      // Decode the factorized native address once per row, not once per
      // vector. This matters when only a few CTAs are available to hide
      // descriptor-load/address-calculation latency.
      const int64_t begin = local_tile * kTile, end = min(begin + kTile, t[0]);
      const int64_t inner = t[15];
      int64_t first = begin, last = end - 1;
      coordinate(first, inner);
      coordinate(last, inner);
      const int workers = static_cast<int>(min(last - first + 1, int64_t(kThreads / 32)));
      const int lanes = 32 * max(1, (kThreads / 32) / workers);
      const int row_worker = threadIdx.x / lanes, row_step = kThreads / lanes;
      const int lane = threadIdx.x % lanes;
      const int width = sd == dd ? (sd == 0 ? 4 : 8) : 4;
      for (int64_t row = first + row_worker; row <= last; row += row_step) {
        const int64_t base = row * inner;
        const int64_t native = address(base, t, program);
        const int64_t c0 = max(int64_t(0), begin - base), c1 = min(inner, end - base);
        for (int64_t c = c0 + lane * width; c < c1; c += lanes * width) {
          const int count = static_cast<int>(min(int64_t(width), c1 - c));
          copy_chunk(src, dst, sd, dd, scatter ? base + c : native + c,
                     scatter ? native + c : base + c, count, add);
        }
      }
    } else {
      const int64_t begin = local_tile * kTile;
      const int64_t end = min(begin + kTile, t[0]);
      const int vector = sd == 0 ? 4 : 8;
      const bool vector_copy = !add && sd == dd && ((src | dst) & 15) == 0 &&
                               t[15] > 0 && t[15] % vector == 0;
      if (vector_copy) {
        for (int64_t j = begin + threadIdx.x * vector; j < end;
             j += kThreads * vector) {
          const int64_t mapped = t[10] == 2 ? t[13] + j : address(j, t, program);
          const int64_t si = scatter ? j : mapped, di = scatter ? mapped : j;
          if (j + vector <= end && si % vector == 0 && di % vector == 0) {
            const int bytes = sd == 0 ? 4 : 2;
            *reinterpret_cast<uint4*>(dst + di * bytes) =
                *reinterpret_cast<const uint4*>(src + si * bytes);
          } else {
            for (int k = 0; k < vector && j + k < end; ++k) {
              const int64_t at = address(j + k, t, program);
              store(dst, dd, scatter ? at : j + k, load(src, sd, scatter ? j + k : at), false);
            }
          }
        }
      } else {
        // Four adjacent elements per lane amortize address decoding where the
        // native innermost dimension is contiguous (copy/cast row blocks).
        for (int64_t j = begin + threadIdx.x * 4; j < end; j += kThreads * 4) {
          const int64_t mapped = t[10] == 2 ? t[13] + j : address(j, t, program);
          const bool consecutive = t[15] > 0 &&
                                   ((t[15] & 3) == 0 || j % t[15] + 4 <= t[15]);
          #pragma unroll
          for (int k = 0; k < 4; ++k) {
            if (j + k < end) {
              const int64_t at = consecutive || t[10] == 2 ? mapped + k : address(j + k, t, program);
              store(dst, dd, scatter ? at : j + k, load(src, sd, scatter ? j + k : at), add);
            }
          }
        }
      }
    }
  }
}
}  // namespace

void layout_copy(torch::Tensor descriptors, torch::Tensor program, torch::Tensor pointers,
                 int64_t tiles, int64_t max_ctas, int64_t dispatch) {
  for (const auto& t : {descriptors, program, pointers}) {
    TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.scalar_type() == c10::kLong,
                "layout copy metadata must be contiguous int64 CUDA tensors");
    TORCH_CHECK(t.device() == descriptors.device(), "layout copy metadata device mismatch");
  }
  TORCH_CHECK(descriptors.dim() == 2 && descriptors.size(1) == kFields,
              "layout copy descriptor shape mismatch");
  TORCH_CHECK(pointers.dim() == 2 && pointers.size(1) == 2 &&
              pointers.size(0) == descriptors.size(0), "layout copy pointer shape mismatch");
  TORCH_CHECK(max_ctas > 0 && tiles >= 0, "Invalid layout copy CTA budget/tile count");
  if (tiles == 0 || descriptors.size(0) == 0) return;
  const c10::cuda::CUDAGuard guard(descriptors.device());
  const int64_t blocks = std::min({max_ctas, tiles,
      static_cast<int64_t>(at::cuda::getCurrentDeviceProperties()->multiProcessorCount)});
  const auto stream = at::cuda::getCurrentCUDAStream();
  if (dispatch & 256) {
    if (dispatch & 2) narrow_copy_kernel<true><<<blocks,512,0,stream>>>(
        descriptors.data_ptr<int64_t>(),pointers.data_ptr<int64_t>(),descriptors.size(0),tiles);
    else narrow_copy_kernel<false><<<blocks,512,0,stream>>>(
        descriptors.data_ptr<int64_t>(),pointers.data_ptr<int64_t>(),descriptors.size(0),tiles);
  } else if (dispatch & 128) {
    if (dispatch & 1) {
      if (dispatch & 2) choose_layout16<true,true>(descriptors,program,pointers,tiles,blocks,dispatch,stream);
      else choose_layout16<true,false>(descriptors,program,pointers,tiles,blocks,dispatch,stream);
    } else {
      if (dispatch & 2) choose_layout16<false,true>(descriptors,program,pointers,tiles,blocks,dispatch,stream);
      else choose_layout16<false,false>(descriptors,program,pointers,tiles,blocks,dispatch,stream);
    }
  } else if (dispatch) {
    layout_copy_kernel<true><<<static_cast<unsigned>(blocks), kThreads,
                              kTranspose * (kTranspose + 1) * sizeof(uint32_t), stream>>>(
        descriptors.data_ptr<int64_t>(), program.data_ptr<int64_t>(), pointers.data_ptr<int64_t>(),
        static_cast<int>(descriptors.size(0)), tiles);
  } else {
    layout_copy_kernel<false><<<static_cast<unsigned>(blocks), kThreads, 0, stream>>>(
        descriptors.data_ptr<int64_t>(), program.data_ptr<int64_t>(), pointers.data_ptr<int64_t>(),
        static_cast<int>(descriptors.size(0)), tiles);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.attr("api_version") = 3;
  m.def("copy", &layout_copy, "Batched layout copy/cast/add with bounded CTA count");
}

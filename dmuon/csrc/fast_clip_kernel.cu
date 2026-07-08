#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace {

constexpr int kFloat32 = 0;
constexpr int kFloat16 = 1;
constexpr int kBFloat16 = 2;

template <typename scalar_t>
__device__ inline float load_as_float(const int64_t ptr, const int64_t offset) {
    const auto* data = reinterpret_cast<const scalar_t*>(static_cast<uintptr_t>(ptr));
    return static_cast<float>(data[offset]);
}

template <typename scalar_t>
__device__ inline void scale_value(const int64_t ptr, const int64_t offset, const float coef) {
    auto* data = reinterpret_cast<scalar_t*>(static_cast<uintptr_t>(ptr));
    const float value = static_cast<float>(data[offset]) * coef;
    data[offset] = static_cast<scalar_t>(value);
}

__global__ void segmented_l2_norm_sq_kernel(
    const int64_t* __restrict__ ptrs,
    const int64_t* __restrict__ numels,
    const int32_t* __restrict__ dtypes,
    const int32_t* __restrict__ segments,
    const int32_t* __restrict__ job_tensor_ids,
    const int64_t* __restrict__ job_offsets,
    const int64_t chunk_size,
    float* __restrict__ local_sq) {
    extern __shared__ float smem[];

    const int job = blockIdx.x;
    const int tensor_id = job_tensor_ids[job];
    const int64_t tensor_numel = numels[tensor_id];
    const int64_t begin = job_offsets[job];
    const int64_t end = min(begin + chunk_size, tensor_numel);
    const int dtype = dtypes[tensor_id];

    float thread_sum = 0.0f;
    for (int64_t offset = begin + threadIdx.x; offset < end; offset += blockDim.x) {
        float value = 0.0f;
        if (dtype == kFloat32) {
            value = load_as_float<float>(ptrs[tensor_id], offset);
        } else if (dtype == kFloat16) {
            value = load_as_float<c10::Half>(ptrs[tensor_id], offset);
        } else {
            value = load_as_float<c10::BFloat16>(ptrs[tensor_id], offset);
        }
        thread_sum += value * value;
    }

    smem[threadIdx.x] = thread_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smem[threadIdx.x] += smem[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(local_sq + segments[tensor_id], smem[0]);
    }
}

__global__ void segmented_scale_kernel(
    const int64_t* __restrict__ ptrs,
    const int64_t* __restrict__ numels,
    const int32_t* __restrict__ dtypes,
    const int32_t* __restrict__ segments,
    const int32_t* __restrict__ job_tensor_ids,
    const int64_t* __restrict__ job_offsets,
    const int64_t chunk_size,
    const float* __restrict__ coefs) {
    const int job = blockIdx.x;
    const int tensor_id = job_tensor_ids[job];
    const int64_t tensor_numel = numels[tensor_id];
    const int64_t begin = job_offsets[job];
    const int64_t end = min(begin + chunk_size, tensor_numel);
    const int dtype = dtypes[tensor_id];
    const float coef = coefs[segments[tensor_id]];

    for (int64_t offset = begin + threadIdx.x; offset < end; offset += blockDim.x) {
        if (dtype == kFloat32) {
            scale_value<float>(ptrs[tensor_id], offset, coef);
        } else if (dtype == kFloat16) {
            scale_value<c10::Half>(ptrs[tensor_id], offset, coef);
        } else {
            scale_value<c10::BFloat16>(ptrs[tensor_id], offset, coef);
        }
    }
}

void check_cuda_int_tensor(const torch::Tensor& tensor, const c10::ScalarType dtype) {
    TORCH_CHECK(tensor.is_cuda(), "metadata tensor must be CUDA");
    TORCH_CHECK(tensor.is_contiguous(), "metadata tensor must be contiguous");
    TORCH_CHECK(tensor.scalar_type() == dtype, "metadata tensor has unexpected dtype");
}

}  // namespace

void segmented_l2_norm_sq(
    torch::Tensor ptrs,
    torch::Tensor numels,
    torch::Tensor dtypes,
    torch::Tensor segments,
    torch::Tensor job_tensor_ids,
    torch::Tensor job_offsets,
    torch::Tensor local_sq,
    int64_t chunk_size) {
    check_cuda_int_tensor(ptrs, c10::kLong);
    check_cuda_int_tensor(numels, c10::kLong);
    check_cuda_int_tensor(dtypes, c10::kInt);
    check_cuda_int_tensor(segments, c10::kInt);
    check_cuda_int_tensor(job_tensor_ids, c10::kInt);
    check_cuda_int_tensor(job_offsets, c10::kLong);
    TORCH_CHECK(local_sq.is_cuda(), "local_sq must be CUDA");
    TORCH_CHECK(local_sq.is_contiguous(), "local_sq must be contiguous");
    TORCH_CHECK(local_sq.scalar_type() == c10::kFloat, "local_sq must be float32");
    TORCH_CHECK(chunk_size > 0, "chunk_size must be positive");

    const auto jobs = job_tensor_ids.numel();
    if (jobs == 0) {
        return;
    }

    const c10::cuda::CUDAGuard device_guard(ptrs.device());
    constexpr int threads = 256;
    const dim3 blocks(static_cast<unsigned int>(jobs));
    const size_t smem = threads * sizeof(float);
    segmented_l2_norm_sq_kernel<<<blocks, threads, smem, at::cuda::getCurrentCUDAStream()>>>(
        ptrs.data_ptr<int64_t>(),
        numels.data_ptr<int64_t>(),
        dtypes.data_ptr<int32_t>(),
        segments.data_ptr<int32_t>(),
        job_tensor_ids.data_ptr<int32_t>(),
        job_offsets.data_ptr<int64_t>(),
        chunk_size,
        local_sq.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void segmented_scale(
    torch::Tensor ptrs,
    torch::Tensor numels,
    torch::Tensor dtypes,
    torch::Tensor segments,
    torch::Tensor job_tensor_ids,
    torch::Tensor job_offsets,
    torch::Tensor coefs,
    int64_t chunk_size) {
    check_cuda_int_tensor(ptrs, c10::kLong);
    check_cuda_int_tensor(numels, c10::kLong);
    check_cuda_int_tensor(dtypes, c10::kInt);
    check_cuda_int_tensor(segments, c10::kInt);
    check_cuda_int_tensor(job_tensor_ids, c10::kInt);
    check_cuda_int_tensor(job_offsets, c10::kLong);
    TORCH_CHECK(coefs.is_cuda(), "coefs must be CUDA");
    TORCH_CHECK(coefs.is_contiguous(), "coefs must be contiguous");
    TORCH_CHECK(coefs.scalar_type() == c10::kFloat, "coefs must be float32");
    TORCH_CHECK(chunk_size > 0, "chunk_size must be positive");

    const auto jobs = job_tensor_ids.numel();
    if (jobs == 0) {
        return;
    }

    const c10::cuda::CUDAGuard device_guard(ptrs.device());
    constexpr int threads = 256;
    const dim3 blocks(static_cast<unsigned int>(jobs));
    segmented_scale_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        ptrs.data_ptr<int64_t>(),
        numels.data_ptr<int64_t>(),
        dtypes.data_ptr<int32_t>(),
        segments.data_ptr<int32_t>(),
        job_tensor_ids.data_ptr<int32_t>(),
        job_offsets.data_ptr<int64_t>(),
        chunk_size,
        coefs.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("segmented_l2_norm_sq", &segmented_l2_norm_sq, "Segmented L2 norm squared");
    m.def("segmented_scale", &segmented_scale, "Segmented gradient scaling");
}

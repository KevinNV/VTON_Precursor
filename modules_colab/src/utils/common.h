#pragma once

#include <ATen/ATen.h>
#include <cmath>

/*
 * Functions to share code between CPU and GPU
 */

#ifdef __CUDACC__
// CUDA versions

#define HOST_DEVICE __host__ __device__
#define INLINE_HOST_DEVICE __host__ __device__ inline
#define FLOOR(x) floor(x)

#if __CUDA_ARCH__ >= 600
// Recent compute capabilities support atomicAdd for doubles
#define ACCUM(x, y) atomicAdd(&(x), (y))
#else
// Generic atomicAdd + workaround for double on older GPUs
template <typename data_t>
__device__ inline data_t atomic_add(data_t* address, data_t val) {
  return atomicAdd(address, val);
}

// Specialization for double using atomicCAS
template <>
__device__ inline double atomic_add(double* address, double val) {
  unsigned long long int* addr_as_ull = reinterpret_cast<unsigned long long int*>(address);
  unsigned long long int old = *addr_as_ull, assumed;

  do {
    assumed = old;
    old = atomicCAS(addr_as_ull, assumed,
                    __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);

  return __longlong_as_double(old);
}

#define ACCUM(x, y) atomic_add(&(x), (y))
#endif // __CUDA_ARCH__ >= 600

#else
// CPU versions

#define HOST_DEVICE
#define INLINE_HOST_DEVICE inline
#define FLOOR(x) std::floor(x)
#define ACCUM(x, y) (x) += (y)

#endif // __CUDACC__


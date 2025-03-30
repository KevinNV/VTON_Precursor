#pragma once

#include <ATen/ATen.h>

// Compatibility macro for AT_CHECK (used in older versions of PyTorch)
#ifndef AT_CHECK
#define AT_CHECK AT_ASSERT
#endif

// Modern CUDA/CPU and contiguity checks using latest APIs
#define CHECK_CUDA(x) AT_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CPU(x)  AT_CHECK(!(x).is_cuda(), #x " must be a CPU tensor")
#define CHECK_CONTIGUOUS(x) AT_CHECK((x).is_contiguous(), #x " must be contiguous")

// Composite checks for convenience
#define CHECK_CUDA_INPUT(x)  CHECK_CUDA(x);  CHECK_CONTIGUOUS(x)
#define CHECK_CPU_INPUT(x)   CHECK_CPU(x);   CHECK_CONTIGUOUS(x)


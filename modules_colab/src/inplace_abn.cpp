#include <torch/extension.h>
#include <vector>
#include "inplace_abn.h"

// ------------------------ Mean & Variance ------------------------ //

std::vector<at::Tensor> mean_var(at::Tensor x) {
  if (x.is_cuda()) {
    if (x.scalar_type() == at::ScalarType::Half) {
      return mean_var_cuda_h(x);
    } else {
      return mean_var_cuda(x);
    }
  } else {
    return mean_var_cpu(x);
  }
}

// ------------------------ Forward ------------------------ //

at::Tensor forward(at::Tensor x, at::Tensor mean, at::Tensor var,
                   at::Tensor weight, at::Tensor bias,
                   bool affine, float eps) {
  if (x.is_cuda()) {
    if (x.scalar_type() == at::ScalarType::Half) {
      return forward_cuda_h(x, mean, var, weight, bias, affine, eps);
    } else {
      return forward_cuda(x, mean, var, weight, bias, affine, eps);
    }
  } else {
    return forward_cpu(x, mean, var, weight, bias, affine, eps);
  }
}

// ------------------------ edz & eydz ------------------------ //

std::vector<at::Tensor> edz_eydz(at::Tensor z, at::Tensor dz,
                                 at::Tensor weight, at::Tensor bias,
                                 bool affine, float eps) {
  if (z.is_cuda()) {
    if (z.scalar_type() == at::ScalarType::Half) {
      return edz_eydz_cuda_h(z, dz, weight, bias, affine, eps);
    } else {
      return edz_eydz_cuda(z, dz, weight, bias, affine, eps);
    }
  } else {
    return edz_eydz_cpu(z, dz, weight, bias, affine, eps);
  }
}

// ------------------------ Backward ------------------------ //

at::Tensor backward(at::Tensor z, at::Tensor dz, at::Tensor var,
                    at::Tensor weight, at::Tensor bias,
                    at::Tensor edz, at::Tensor eydz,
                    bool affine, float eps) {
  if (z.is_cuda()) {
    if (z.scalar_type() == at::ScalarType::Half) {
      return backward_cuda_h(z, dz, var, weight, bias, edz, eydz, affine, eps);
    } else {
      return backward_cuda(z, dz, var, weight, bias, edz, eydz, affine, eps);
    }
  } else {
    return backward_cpu(z, dz, var, weight, bias, edz, eydz, affine, eps);
  }
}

// ------------------------ Activations ------------------------ //

void leaky_relu_forward(at::Tensor z, float slope) {
  at::leaky_relu_(z, slope);
}

void leaky_relu_backward(at::Tensor z, at::Tensor dz, float slope) {
  if (z.is_cuda()) {
    if (z.scalar_type() == at::ScalarType::Half) {
      leaky_relu_backward_cuda_h(z, dz, slope);
    } else {
      leaky_relu_backward_cuda(z, dz, slope);
    }
  } else {
    leaky_relu_backward_cpu(z, dz, slope);
  }
}

void elu_forward(at::Tensor z) {
  at::elu_(z);
}

void elu_backward(at::Tensor z, at::Tensor dz) {
  if (z.is_cuda()) {
    elu_backward_cuda(z, dz);
  } else {
    elu_backward_cpu(z, dz);
  }
}

// ------------------------ Pybind Module ------------------------ //

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mean_var", &mean_var, "Mean and variance computation");
  m.def("forward", &forward, "In-place forward computation");
  m.def("edz_eydz", &edz_eydz, "First part of backward computation");
  m.def("backward", &backward, "Second part of backward computation");
  m.def("leaky_relu_forward", &leaky_relu_forward, "Leaky ReLU forward computation");
  m.def("leaky_relu_backward", &leaky_relu_backward, "Leaky ReLU backward computation and inversion");
  m.def("elu_forward", &elu_forward, "ELU forward computation");
  m.def("elu_backward", &elu_backward, "ELU backward computation and inversion");
}


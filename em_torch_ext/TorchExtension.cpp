/**********************************************************************************************************************
 * Name: TorchExtension.cpp
 * Author: Matin Raayai Ardakani
 * Email: matinraayai@seas.harvard.edu
 * This file contains all the Python function interfaces for the package em_pre_cuda.
 * Based on the code from Pytorch's tutorials: https://github.com/pytorch/extension-cpp
 **********************************************************************************************************************/
#include "TorchExtension.h"


torch::Tensor median_filter(const torch::Tensor& tensor, const torch::Tensor& kernel) {
    CHECK_CUDA_TENSOR(tensor)
    CHECK_TENSOR_TYPE(tensor, torch::kF32)
    CHECK_CPU_TENSOR(kernel)
    auto f_copy = kernel;
    //TODO: Make this accept all types.
    auto fa = f_copy.accessor<long, 1>();
    int radX = static_cast<int>(fa[2]);
    int radY = static_cast<int>(fa[1]);
    int radZ = static_cast<int>(fa[0]);
    return cuda_median_3d(tensor, radX, radY, radZ);
}


torch::Tensor median_filter_v2(const torch::Tensor& tensor) {
    CHECK_CUDA_TENSOR(tensor)
    CHECK_TENSOR_TYPE(tensor, torch::kF32)
    //Check if imStack has a float ScalarType
    return cuda_median_3d(tensor);
}

torch::Tensor median_filter_v3(const torch::Tensor& tensor, const std::vector<int>& kernel) {
    CHECK_CUDA_TENSOR(tensor);
    return cuda_median_3d(tensor, kernel[2], kernel[1], kernel[0]);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("median_filter", &median_filter, "CUDA 3D median filter.");
    m.def("median_filter", &median_filter_v2, "CUDA 3D median filter v.2 returning only the middle slice.");
    m.def("median_filter", &median_filter_v3, "CUDA 3D median filter v.3.");
}
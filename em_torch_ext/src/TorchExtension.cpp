/**********************************************************************************************************************
 * Name: TorchExtension.cpp
 * Author: Matin Raayai Ardakani
 * Email: matinraayai@seas.harvard.edu
 * This file contains all the Python function interfaces for the package em_pre_cuda.
 * Based on the code from Pytorch's tutorials: https://github.com/pytorch/extension-cpp
 **********************************************************************************************************************/
#include "TorchExtension.h"


at::Tensor median_filter(const at::Tensor& tensor, const at::Tensor& kernel) {
    CHECK_CUDA_TENSOR(tensor)
    CHECK_TENSOR_TYPE(tensor, torch::kF32)
    CHECK_CPU_TENSOR(kernel)
    auto ka = kernel.accessor<long, 1>();
    int radX = static_cast<int>(ka[2]);
    int radY = static_cast<int>(ka[1]);
    int radZ = static_cast<int>(ka[0]);
    return cuda_median_3d(tensor, radX, radY, radZ);
}

at::Tensor median_filter_v2(const at::Tensor& tensor, const std::vector<int>& kernel) {
    CHECK_CUDA_TENSOR(tensor)
    CHECK_TENSOR_TYPE(tensor, torch::kF32)
    return cuda_median_3d(tensor, kernel[2], kernel[1], kernel[0]);
}

at::Tensor idm(const at::Tensor& tensor1,
               const at::Tensor& tensor2,
               int patch_size,
               int warp_size,
               int patch_step,
               int metric) {
    CHECK_CUDA_TENSOR(tensor1)
    CHECK_CUDA_TENSOR(tensor2)
    TORCH_CHECK(tensor1.sizes() == tensor2.sizes(), "The input images must have the same dimensions.")
    return cuda_idm(tensor1, tensor2, patch_size, warp_size, patch_step, metric);

}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("median_filter", &median_filter, "CUDA 3D median filter.");
    m.def("median_filter", &median_filter_v2, "CUDA 3D median filter.");
    m.def("idm", &idm, "Image deformation model.");
}
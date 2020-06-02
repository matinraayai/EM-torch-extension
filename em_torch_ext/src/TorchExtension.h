/**
 * Name: TorchExtension.h
 * Author: Matin Raayai Ardakani
 * Email: matinraayai@seas.harvard.edu
 * Based on the code from Pytorch's tutorials: https://github.com/pytorch/extension-cpp
 **********************************************************************************************************************/
#pragma once
#include "TorchExtensionKernel.h"
#include <vector>

//Torch Tensor checks
#define CHECK_TENSOR_IS_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor.")
#define CHECK_TENSOR_IS_CPU(x) TORCH_CHECK(!x.device().is_cuda(), #x " must be a CPU tensor.")
#define CHECK_TENSOR_IS_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous.")
#define CHECK_CUDA_TENSOR(x) CHECK_TENSOR_IS_CUDA(x); CHECK_TENSOR_IS_CONTIGUOUS(x)
#define CHECK_CPU_TENSOR(x) CHECK_TENSOR_IS_CPU(x); CHECK_TENSOR_IS_CONTIGUOUS(x)
#define CHECK_TENSOR_TYPE(x, type) TORCH_CHECK(x.scalar_type() == type, #x " must be a " #type " tensor.")

/*3D Median filter needed by traditional deflicker in em_pre.=========================================================*/

//TODO: Create a Hashmap that holds the documentation for each function.

at::Tensor median_filter(const at::Tensor& tensor, const at::Tensor& kernel);

at::Tensor median_filter_v2(const at::Tensor& tensor, const std::vector<int>& kernel);

/*Image deformation model distance needed in em_pre.==================================================================*/
at::Tensor idm(const at::Tensor& tensor1,
               const at::Tensor& tensor2,
               int patch_size,
               int warp_size,
               int patch_step,
               int metric);

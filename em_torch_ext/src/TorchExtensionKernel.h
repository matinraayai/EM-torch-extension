/**********************************************************************************************************************
 * Name: em_pre_cuda_kernel.h
 * Author: Matin Raayai Ardakani
 * Email: matinraayai@seas.harvard.edu
 * Header file for em_pre_cuda_kernel.h
 * Based on the code from Pytorch's tutorials: https://github.com/pytorch/extension-cpp
 **********************************************************************************************************************/
#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <torch/torch.h>
#define BLOCK_DIM_LEN 8
#define MAX_GPU_ARRAY_LEN 30

torch::Tensor cuda_median_3d(const torch::Tensor& tensor, int radX, int radY, int radZ);

torch::Tensor cuda_idm(const torch::Tensor& tensor1,
                       const torch::Tensor& tensor2,
                       int patch_size,
                       int warp_size,
                       int step,
                       int metric);
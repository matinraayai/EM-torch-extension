/**********************************************************************************************************************
 * Name: TorchExtensionKernel.cpp
 * Author: Matin Raayai Ardakani
 * Email: matinraayai@seas.harvard.edu
 * Where the CUDA magic happens for the em_pre_cuda Python package.
 * Based on the code from Pytorch's tutorials: https://github.com/pytorch/extension-cpp
 **********************************************************************************************************************/
#include "TorchExtensionKernel.h"

/*
 * A helper function used in each CUDA thread that returns the idx if if idx > minIdx and idx < maxIdx. If not,
 * it will "reflect" the returned index so that it falls between the minimum and maximum range.
 * This helps with applying a 3D median filter while "reflecing" the boundries.
 * @param idx the index
 * @param minIdx the lower bound of the 1D tensor
 * @param maxIdx the upper bound of the 1D tensor
 * @return the index of the element, safe to access the intended 1D tensor.
 */
__device__ __forceinline__ int clamp_mirror(int idx, int minIdx, int maxIdx)
{
    if(idx < minIdx) return clamp_mirror(minIdx + (minIdx - idx), minIdx, maxIdx);
    else if(idx > maxIdx) return clamp_mirror(maxIdx - (idx - maxIdx), minIdx, maxIdx);
    else return idx;
}

template<typename scalar_t>
__device__ __host__ scalar_t get_median_of_array(scalar_t* vector, int vSize)
{
    for (int i = 0; i < vSize; i++) {
    for (int j = i + 1; j < vSize; j++) {
        if (vector[i] > vector[j]) {
            scalar_t tmp = vector[i];
            vector[i] = vector[j];
            vector[j] = tmp;
        }
    }}
    return vector[vSize / 2];
}

template<typename scalar_t>
__global__
void __median_3d(scalar_t* __restrict__ input, scalar_t* __restrict__ output, int dimX, int dimY,
                 int dimZ, int radX, int radY, int radZ)
{
    auto get_1d_idx = [&] (int32_t x, int32_t y, int32_t z) {
        return clamp_mirror(z, 0, dimZ - 1) * dimY * dimX +
               clamp_mirror(y, 0, dimY - 1) * dimX + clamp_mirror(x, 0, dimX - 1);
    };

    const int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int row_idx = blockIdx.y * blockDim.y + threadIdx.y;
    const int sht_idx = blockIdx.z * blockDim.z + threadIdx.z;

    scalar_t windowVec[MAX_GPU_ARRAY_LEN] = {0.};
    int vSize = 0;

    if (col_idx < dimX && row_idx < dimY && sht_idx < dimZ) {
        for (int z = -radZ; z <= radZ; z++)
        for (int y = -radY; y <= radY; y++)
        for (int x = -radX; x <= radX; x++)
            windowVec[vSize++] = input[get_1d_idx(x + col_idx, y + row_idx, z + sht_idx)];
        output[get_1d_idx(col_idx, row_idx, sht_idx)] = get_median_of_array(windowVec, vSize);
    }
}

//template<typename scalar_t>
//__global__
//scalar_t __patch_distance (const int A_x, const int A_y, const int B_x, const int B_y,
//                           const int im_row, const int im_col, const int im_chan,
//                           const int patch_sz, scalar_t *img1, scalar_t *img2, int metric){
//    scalar_t dist = 0, temp_h;
//    int c, x, y, count = 0;
//    /* only move around patchB */
//    int pre = im_col * im_chan;
//    scalar_t patch_sum = 0;
//
//    switch(metric) {
//        case 0: // L1
//            for(y = -patch_sz; y <= patch_sz; y++) {
//            for(x = -patch_sz; x <= patch_sz; x++) {
//                if((A_x + x) >= 0 && (A_y + y) >= 0 && (A_x + x) < im_row && (A_y + y) < im_col &&
//                   (B_x + x) >= 0 && (B_y + y) >= 0 && (B_x + x) < im_row && (B_y + y) < im_col) {
//                    for(c = 0; c < im_chan; c++) {
//                        temp_h = img1[(A_x + x)*pre + (A_y + y)*im_chan + c] -
//                                 img2[(B_x + x)*pre + (B_y + y)*im_chan + c];
//                        dist += fabsf(temp_h);
//                        count++;
//                    }
//                }
//            }}
//            break;
//        case 1: // relative L1
//            for(y=-patch_sz; y<=patch_sz; y++){
//                for(x=-patch_sz; x<=patch_sz; x++){
//                    if((A_x + x)>=0 && (A_y + y)>=0 && (A_x + x)<im_row && (A_y + y)<im_col
//                       && (B_x + x)>=0 && (B_y + y)>=0 && (B_x + x)<im_row && (B_y + y)<im_col){
//                        for(c=0; c<im_chan; c++){
//                            temp_h = img1[(A_x + x)*pre + (A_y + y)*im_chan + c] -
//                                     img2[(B_x + x)*pre + (B_y + y)*im_chan + c];
//                            dist += fabsf(temp_h);
//                            patch_sum += img1[(A_x + x)*pre + (A_y + y)*im_chan + c];
//                            //dist+=temp_h*temp_h;
//                            count++;
//                        }
//                    }
//                }
//            }
//            dist = dist/patch_sum;
//            break;
//    }
//    return dist/count;
//}
////
//template<typename scalar_t>
//__global__
//void __idm_dist(scalar_t* img1, scalar_t* img2, scalar_t* dis,
//                int im_row, int im_col, int im_chan,
//                int patch_sz, int warp_sz, int step, int metric) {
//    /* assume same size img */
//    scalar_t best_dis, temp;
//    int xx, yy;
//    const int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
//    const int32_t y = blockIdx.y * blockDim.y + threadIdx.y;
//    /* 3) Return distance */
//    int count = 0;
//    best_dis = FLT_MAX;
//    for(xx = x - warp_sz; xx <= x + warp_sz; xx++){
//    for(yy = y - warp_sz; yy <= y + warp_sz; yy++){
//        if(xx >= 0 && yy >= 0 && xx < im_row && yy < im_col) {
//                temp = patch_distance(x, y, xx, yy, im_row, im_col, im_chan,
//                                      patch_sz, img1, img2, metric);
//                if(temp < best_dis)
//                    best_dis = temp;
//        }
//    }}
//    dis[count] = best_dis;
//    count++;
//}



torch::Tensor cuda_median_3d(const torch::Tensor& tensor, const int radX, const int radY, const int radZ) {

    torch::Tensor out = torch::zeros_like(tensor);
    const int dimX = tensor.size(2), dimY = tensor.size(1), dimZ = tensor.size(0);
    const dim3 blockDim(BLOCK_DIM_LEN, BLOCK_DIM_LEN, BLOCK_DIM_LEN);
    const dim3 gridDim(
            (dimX / blockDim.x + ((dimX % blockDim.x) ? 1 : 0)),
            (dimY / blockDim.y + ((dimY % blockDim.y) ? 1 : 0)),
            (dimZ / blockDim.z + ((dimZ % blockDim.z) ? 1 : 0)));

    AT_DISPATCH_FLOATING_TYPES(tensor.scalar_type(), "__median_3d", ([&] {
        __median_3d<scalar_t><<<gridDim, blockDim>>>(
                tensor.data_ptr<scalar_t>(),
                out.data_ptr<scalar_t>(),
                dimX,
                dimY,
                dimZ,
                radX,
                radY,
                radZ);
    }));
    return out;
}

torch::Tensor cuda_idm(const torch::Tensor& tensor1,
                       const torch::Tensor& tensor2,
                       int patch_size,
                       int warp_size,
                       int step,
                       int metric) {
    const int dimX = tensor1.size(1), dimY = tensor1.size(0);
    torch::Tensor out = torch::zeros()

    const dim3 blockDim(BLOCK_DIM_LEN, BLOCK_DIM_LEN, 0);
    const dim3 gridDim((dimX / blockDim.x + ((dimX % blockDim.x) ? 1 : 0)),
                       (dimY / blockDim.y + ((dimY % blockDim.y) ? 1 : 0)), 0);

    AT_DISPATCH_FLOATING_TYPES(sliceStack.type(), "__idm_dist", ([&] {
        __median_3d<scalar_t><<<gridDim, blockDim>>>(
                sliceStack.data_ptr<scalar_t>(),
                out.data_ptr<scalar_t>(),
                dimX,
                dimY,
                dimZ);
    }));
    return out;

}
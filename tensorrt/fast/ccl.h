#ifndef _CCL_PLUGIN_H
#define _CCL_PLUGIN_H
//#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/script.h>
//#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

namespace cc2d {
    __global__ void init_labeling(int32_t *label, const uint32_t W, const uint32_t H);
    __global__ void merge(uint8_t *img, int32_t *label, const uint32_t W, const uint32_t H);
    __global__ void compression(int32_t *label, const int32_t W, const int32_t H);
    __global__ void final_labeling(const uint8_t *img, int32_t *label, const int32_t W, const int32_t H);
};

torch::Tensor connected_componnets_labeling_2d(const torch::Tensor &input);
torch::Tensor connected_componnets_labeling_2d_batch(const torch::Tensor &input);
#endif
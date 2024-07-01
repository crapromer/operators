#include "../../../devices/cuda/handle_pool.h"
#include "../../utils.h"
#include "batch_norm.cuh"
#include <cub/block/block_load.cuh>
#include <cub/block/block_reduce.cuh>

void batch_norm_nv_gpu_f16(Tensor y, Tensor x, Tensor w, float epsilon, void *stream){
    ASSERT_EQ(w.layout.ndim, 4);
    float alpha = 1.f, beta = 0.f;
    cudnnTensorDescriptor_t inDesc;
    cudnnCreateTensorDescriptor(&inDesc);
    cudnnSetTensorNdDescriptor(
            inDesc, CUDNN_DATA_FLOAT, x.layout.ndim, reinterpret_cast<const int *>(x.layout.shape), reinterpret_cast<const int *>(x.layout.strides));
    // get bnScaleBiasMeanVarDesc
    cudnnTensorDescriptor_t paraDesc;
    cudnnCreateTensorDescriptor(&paraDesc);
    cudnnSetTensorNdDescriptor(
            paraDesc, CUDNN_DATA_FLOAT, y.layout.ndim, reinterpret_cast<const int *>(y.layout.shape), reinterpret_cast<const int *>(y.layout.strides));
    float biasHost[4] = {0.0f, 0.0f, 0.0f,0.0f};
    float *biasDevice = nullptr;
    cudaMalloc((void**)&biasHost, 4 * sizeof(float));
    cudaMemcpy(biasDevice, biasHost, 4 * sizeof(float), cudaMemcpyHostToDevice);
    float scaleHost[4] = {1.0f, 1.0f, 1.0f,1.0f};
    float *scaleDevice = nullptr;
    cudaMalloc((void**)&scaleHost, 4 * sizeof(float));
    cudaMemcpy(scaleDevice, scaleHost, 4 * sizeof(float), cudaMemcpyHostToDevice);

    use_cudnn((cudaStream_t) stream,
               [&](cudnnHandle_t handle) { cudnnBatchNormalizationForwardInference(
                                                handle,
                                                CUDNN_BATCHNORM_SPATIAL,
                                                &alpha, 
                                                &beta,
                                                inDesc,
                                                x.data,
                                                inDesc,
                                                y.data,
                                                paraDesc,
                                                scaleHost,
                                                biasDevice,
                                                biasDevice,
                                                scaleHost,
                                                epsilon); });
}
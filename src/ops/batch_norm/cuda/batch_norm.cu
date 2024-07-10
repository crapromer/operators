#include "../../../devices/cuda/handle_pool.h"
#include "../../utils.h"
#include "batch_norm.cuh"
#include <cub/block/block_load.cuh>
#include <cub/block/block_reduce.cuh>

__global__ void dev_const(float *px, float k) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  px[tid] = k;
}

__global__ void dev_iota(float *px, float bias) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  px[tid] = tid + bias;
}


void batch_norm_nv_gpu_f16(Tensor y, Tensor x, float epsilon, void *stream){
    ASSERT_EQ(x.layout->ndim, 4);
    float alpha = 1.f, beta = 0.f;
    const int ndim = static_cast<int>(x.layout->ndim);
    const int *shape = reinterpret_cast<const int *>(x.layout->shape);
    const int *strides = reinterpret_cast<const int *>(x.layout->strides);
    printf("%d %d %d %d\n",shape[0],shape[1],shape[2],shape[3]);
    printf("%d %d %d %d\n",strides[0],strides[1],strides[2],strides[3]);
    cudnnTensorDescriptor_t inDesc;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&inDesc));
    CUDNN_CALL(cudnnSetTensorNdDescriptor(inDesc, CUDNN_DATA_FLOAT, ndim, shape, strides));
    // get bnScaleBiasMeanVarDesc
    cudnnTensorDescriptor_t paraDesc;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&paraDesc));
    CUDNN_CALL(cudnnSetTensorNdDescriptor(
            paraDesc, CUDNN_DATA_FLOAT, y.layout->ndim, reinterpret_cast<const int *>(y.layout->shape), reinterpret_cast<const int *>(y.layout->strides)));
    auto bias = allocate<float>(x.layout->shape[0]*x.layout->shape[1]*x.layout->shape[2]*x.layout->shape[3]);
    dev_iota<<<x.layout->shape[0]*x.layout->shape[1],x.layout->shape[2]*x.layout->shape[3]>>>(bias.get(), 0);
    auto scale = allocate<float>(x.layout->shape[0]*x.layout->shape[1]*x.layout->shape[2]*x.layout->shape[3]);
    dev_iota<<<x.layout->shape[0]*x.layout->shape[1],x.layout->shape[2]*x.layout->shape[3]>>>(scale.get(), 1);

    use_cudnn((cudaStream_t) stream,
               [&](cudnnHandle_t handle) { CUDNN_CALL(cudnnBatchNormalizationForwardInference(
                                                handle,
                                                CUDNN_BATCHNORM_PER_ACTIVATION,
                                                &alpha, 
                                                &beta,
                                                inDesc,
                                                x.data,
                                                inDesc,
                                                y.data,
                                                paraDesc,
                                                static_cast<const void*>(scale.get()),
                                                static_cast<const void*>(bias.get()),
                                                static_cast<const void*>(bias.get()),
                                                static_cast<const void*>(scale.get()),
                                                epsilon)); });
}
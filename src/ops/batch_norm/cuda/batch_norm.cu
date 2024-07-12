#include "../../../devices/cuda/handle_pool.h"
#include "../../utils.h"
#include "batch_norm.cuh"
#include <cub/block/block_load.cuh>
#include <cub/block/block_reduce.cuh>
#include <cuda_fp16.h>
__global__ void dev_const(float *px, int k) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  px[tid] = k;
}

void batch_norm_nv_gpu_f16(Tensor y, Tensor x, float epsilon, void *stream){
    ASSERT_EQ(x.layout->ndim, 4);
    float alpha = 1.f, beta = 0.f;
    const int ndim = static_cast<int>(x.layout->ndim);
    int *shape = new int[ndim];
    int *strides = new int[ndim];
    int *para_shape = new int[ndim];
    int *para_strides = new int[ndim];
    for(int i = 0;i<ndim;i++){
      shape[i] = x.layout->shape[i];
      strides[i] = x.layout->strides[i];
      para_shape[i] = 1;
      para_strides[i] = 1;
    }
    para_shape[1] = shape[1];
    para_strides[0] = shape[1];
    auto data_ = reinterpret_cast<half*>(y.data);
    // printf("%d %d\n",data_[0],data_[1]);
    cudnnTensorDescriptor_t inDesc;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&inDesc));
    CUDNN_CALL(cudnnSetTensorNdDescriptor(inDesc, CUDNN_DATA_BFLOAT16, ndim, shape, strides));
    // get bnScaleBiasMeanVarDesc
    cudnnTensorDescriptor_t paraDesc;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&paraDesc));
    CUDNN_CALL(cudnnSetTensorNdDescriptor(paraDesc, CUDNN_DATA_FLOAT, ndim, para_shape, para_strides));
    auto bias = allocate<float>(shape[1]);
    dev_const<<<1,shape[1]>>>(bias.get(), 0);
    auto scale = allocate<float>(shape[1]);
    dev_const<<<1,shape[1]>>>(scale.get(), 1);
    use_cudnn((cudaStream_t) stream,
               [&](cudnnHandle_t handle) { CUDNN_CALL(cudnnBatchNormalizationForwardInference(
                                                handle,
                                                CUDNN_BATCHNORM_SPATIAL,
                                                &alpha, 
                                                &beta,
                                                inDesc,
                                                reinterpret_cast<void*>(x.data),
                                                inDesc,
                                                reinterpret_cast<void*>(y.data),
                                                paraDesc,
                                                reinterpret_cast<void*>(scale.get()),
                                                reinterpret_cast<void*>(bias.get()),
                                                reinterpret_cast<void*>(bias.get()),
                                                reinterpret_cast<void*>(scale.get()),
                                                epsilon)); });
}
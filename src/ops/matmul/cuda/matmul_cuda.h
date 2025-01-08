#ifndef __CUDA_MATMUL_H__
#define __CUDA_MATMUL_H__

#include "../../../devices/cuda/cuda_handle.h"
#include "../blas.h"
#include "operators.h"
#include <memory>

typedef struct MatmulCudaDescriptor {
    Device device;
    DT dtype;
    int device_id;
    MatmulInfo info;
    float alpha;
    float beta;
    std::shared_ptr<Pool<cublasHandle_t>> cublas_handles_t;
} MatmulCudaDescriptor;

typedef struct MatmulCudaDescriptor *MatmulCudaDescriptor_t;

infiniopStatus_t cudaCreateMatmulDescriptor(CudaHandle_t handle,
                                            MatmulCudaDescriptor_t *desc_ptr,
                                            infiniopTensorDescriptor_t c_desc,
                                            float alpha,
                                            infiniopTensorDescriptor_t a_desc,
                                            infiniopTensorDescriptor_t b_desc,
                                            float beta);

infiniopStatus_t cudaGetMatmulWorkspaceSize(MatmulCudaDescriptor_t desc, uint64_t *size);

infiniopStatus_t cudaMatmul(MatmulCudaDescriptor_t desc,
                            void *workspace,
                            uint64_t workspace_size,
                            void *c,
                            void const *a,
                            void const *b,
                            void *stream);

infiniopStatus_t cudaDestroyMatmulDescriptor(MatmulCudaDescriptor_t desc);

#endif// __CUDA_MATMUL_H__

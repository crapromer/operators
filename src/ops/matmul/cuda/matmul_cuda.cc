#include "matmul_cuda.h"
#include "../../../devices/cuda/common_cuda.h"
#include "../../utils.h"

infiniopStatus_t cudaCreateMatmulDescriptor(CudaHandle_t handle,
                                            MatmulCudaDescriptor_t *desc_ptr,
                                            infiniopTensorDescriptor_t c_desc,
                                            float alpha,
                                            infiniopTensorDescriptor_t a_desc,
                                            infiniopTensorDescriptor_t b_desc,
                                            float beta) {
    DT dtype = c_desc->dt;

    if (dtype != F16 && dtype != F32) {
        return STATUS_BAD_TENSOR_DTYPE;
    }

    infiniopStatus_t *status = new infiniopStatus_t{STATUS_EXECUTION_FAILED};
    auto info = MatmulInfo(c_desc, a_desc, b_desc, status);
    if (*status != STATUS_SUCCESS) {
        return *status;
    }

    *desc_ptr = new MatmulCudaDescriptor{
        DevNvGpu,
        dtype,
        handle->device_id,
        info,
        alpha,
        beta,
        handle->cublas_handles_t};
    return STATUS_SUCCESS;
}

infiniopStatus_t cudaGetMatmulWorkspaceSize(MatmulCudaDescriptor_t desc, uint64_t *size) {
    *size = 0;
    return STATUS_SUCCESS;
}

infiniopStatus_t cudaDestroyMatmulDescriptor(MatmulCudaDescriptor_t desc) {
    desc->cublas_handles_t = nullptr;
    delete desc;
    return STATUS_SUCCESS;
}

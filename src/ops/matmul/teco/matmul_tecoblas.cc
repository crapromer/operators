#include "matmul_tecoblas.h"

infiniopStatus_t tecoCreateMatmulDescriptor(TecoHandle_t handle, MatmulTecoDescriptor_t *desc_ptr, infiniopTensorDescriptor_t c_desc, float alpha, infiniopTensorDescriptor_t a_desc, infiniopTensorDescriptor_t b_desc, float beta) {
    if (a_desc->ndim == 2 && b_desc->ndim == 2){
        
        *desc_ptr = new MatmulTecoDescriptor{handle->device};
        (*desc_ptr)->batch = -1;
        (*desc_ptr)->handle = handle->handle;
        (*desc_ptr)->device = handle->device;
        (*desc_ptr)->stream = handle->stream;
        (*desc_ptr)->m = a_desc->shape[0];
        (*desc_ptr)->k = a_desc->shape[1];
        (*desc_ptr)->n = b_desc->shape[1];
        (*desc_ptr)->transa = TECOBLAS_OP_N;
        (*desc_ptr)->transb = TECOBLAS_OP_N;
        (*desc_ptr)->lda = a_desc->strides[0];
        (*desc_ptr)->ldb = b_desc->strides[0];
        (*desc_ptr)->ldc = c_desc->strides[0];
        (*desc_ptr)->alpha = 1.0f;
        (*desc_ptr)->beta = 0.0f;
        return STATUS_SUCCESS;
    }
    if (a_desc->ndim == 3 && b_desc->ndim == 3){
        *desc_ptr = new MatmulTecoDescriptor{handle->device};
        (*desc_ptr)->batch = a_desc->shape[0];
        (*desc_ptr)->handle = handle->handle;
        (*desc_ptr)->device = handle->device;
        (*desc_ptr)->stream = handle->stream;
        (*desc_ptr)->m = a_desc->shape[1];
        (*desc_ptr)->k = a_desc->shape[2];
        (*desc_ptr)->n = b_desc->shape[2];
        (*desc_ptr)->transa = TECOBLAS_OP_N;
        (*desc_ptr)->transb = TECOBLAS_OP_N;
        (*desc_ptr)->lda = a_desc->strides[1];
        (*desc_ptr)->ldb = b_desc->strides[1];
        (*desc_ptr)->ldc = c_desc->strides[1];
        (*desc_ptr)->strideA = a_desc->strides[0];
        (*desc_ptr)->strideB = b_desc->strides[0];
        (*desc_ptr)->strideC = c_desc->strides[0];
        (*desc_ptr)->alpha = 1.0f;
        (*desc_ptr)->beta = 0.0f;
        return STATUS_SUCCESS;
    }
    return STATUS_BAD_PARAM;

    
}

infiniopStatus_t tecoGetMatmulWorkspaceSize(MatmulTecoDescriptor_t desc, uint64_t *size) {
    tecoblasStatus_t status = tecoblasGetWorkspaceSize(desc->handle, desc->transa, desc->transb, desc->m, desc->n, desc->k, desc->alpha, TECOBLAS_DATA_HALF,desc->lda, 1, TECOBLAS_DATA_HALF, desc->ldb, 1, desc->beta, TECOBLAS_DATA_HALF, desc->ldc, 1, 1, TECOBLAS_HGEMM,reinterpret_cast<size_t*>(size));
    if (status != TECOBLAS_STATUS_SUCCESS) {
        return STATUS_EXECUTION_FAILED;
    }else{
        return STATUS_SUCCESS;
    }
}

infiniopStatus_t tecoMatmul(MatmulTecoDescriptor_t desc, void *workspace, uint64_t workspace_size, void *c, const void *a, const void *b, void *stream) {
    tecoblasSetStream(desc->handle, desc->stream);
    tecoblasSetWorkspace(desc->handle, workspace, workspace_size);
    tecoblasStatus_t status;
    if(desc->batch<0)
        status = tecoblasHgemm(desc->handle, desc->transa, desc->transb, desc->m, desc->n, desc->k, 1.0f, a, desc->lda, b, desc->ldb, 0.0f, c, desc->ldc);
    else
        status = tecoblasHgemmStridedBatched(desc->handle, desc->transa, desc->transb, desc->m, desc->n, desc->k, 1.0f, a, desc->lda,desc->strideA, b, desc->ldb,desc->strideB, 0.0f, c, desc->ldc,desc->strideC,desc->batch);
    sdaaStreamSynchronize(desc->stream);
    if (status != TECOBLAS_STATUS_SUCCESS) {
        return STATUS_EXECUTION_FAILED;
    }else{
        return STATUS_SUCCESS;
    }
}

infiniopStatus_t tecoDestroyMatmulDescriptor(MatmulTecoDescriptor_t desc) {
    return STATUS_SUCCESS;
}

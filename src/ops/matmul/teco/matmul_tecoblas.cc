#include "matmul_tecoblas.h"

infiniopStatus_t tecoCreateMatmulDescriptor(TecoHandle_t handle, MatmulTecoDescriptor_t *desc_ptr, infiniopTensorDescriptor_t c_desc, float alpha, infiniopTensorDescriptor_t a_desc, infiniopTensorDescriptor_t b_desc, float beta) {
    long long int batch,batch_count;
    if (a_desc->ndim == 2 && b_desc->ndim == 2) {
        batch = 0;
        batch_count = 1;
    }else if(a_desc->ndim == 3 && b_desc->ndim == 3){
        batch = 1;
        batch_count = a_desc->shape[0];
    }else{
        return STATUS_BAD_TENSOR_SHAPE;
    }

    tecoblasDataType_t datatype;
    if(a_desc->dt==F16 && b_desc->dt==F16){
        datatype = TECOBLAS_DATA_HALF;
    }else if(a_desc->dt==F32 && b_desc->dt==F32){
        datatype = TECOBLAS_DATA_FLOAT;
    }else{
        return STATUS_BAD_TENSOR_DTYPE;
    }

    tecoblasHandle_t tecoblas_handle;
    tecoblasCreate(&tecoblas_handle);
    // sdaaStream_t stream;
    

    *desc_ptr = new MatmulTecoDescriptor{
        handle->device,
        handle->device_id,
        tecoblas_handle,
        handle->stream,
        datatype,
        TECOBLAS_OP_N,
        TECOBLAS_OP_N,
        a_desc->shape[0+batch],
        a_desc->shape[1+batch],
        b_desc->shape[1+batch],
        1.0f,
        0.0f,
        a_desc->strides[0+batch],
        b_desc->strides[0+batch],
        c_desc->strides[0+batch],
        batch,
        batch_count,
        a_desc->strides[0],
        b_desc->strides[0],
        c_desc->strides[0],
        };
    tecoblasSetStream((*desc_ptr)->handle,(*desc_ptr)->stream);
        
    return STATUS_SUCCESS;
}

infiniopStatus_t tecoGetMatmulWorkspaceSize(MatmulTecoDescriptor_t desc, uint64_t *size) {
    tecoblasStatus_t status;
    if(desc->batch==0){
        if(desc->datatype == TECOBLAS_DATA_HALF)
            status = tecoblasGetWorkspaceSize(desc->handle, desc->transa, desc->transb, desc->m, desc->n, desc->k, desc->alpha, TECOBLAS_DATA_HALF,desc->lda, 1, TECOBLAS_DATA_HALF, desc->ldb, 1, desc->beta, TECOBLAS_DATA_HALF, desc->ldc, 1, desc->batch_count, TECOBLAS_HGEMM,reinterpret_cast<size_t*>(size));
        else
            status = tecoblasGetWorkspaceSize(desc->handle, desc->transa, desc->transb, desc->m, desc->n, desc->k, desc->alpha, TECOBLAS_DATA_FLOAT,desc->lda, 1, TECOBLAS_DATA_FLOAT, desc->ldb, 1, desc->beta, TECOBLAS_DATA_FLOAT, desc->ldc, 1, desc->batch_count, TECOBLAS_SGEMM,reinterpret_cast<size_t*>(size));
    }else{
        if(desc->datatype == TECOBLAS_DATA_HALF)
            status = tecoblasGetWorkspaceSize(desc->handle, desc->transa, desc->transb, desc->m, desc->n, desc->k, desc->alpha, TECOBLAS_DATA_HALF,desc->lda, desc->strideA, TECOBLAS_DATA_HALF, desc->ldb, desc->strideB, desc->beta, TECOBLAS_DATA_HALF, desc->ldc, desc->strideC, desc->batch_count, TECOBLAS_HGEMM_STRIDED_BATCHED,reinterpret_cast<size_t*>(size));
        else
            status = tecoblasGetWorkspaceSize(desc->handle, desc->transa, desc->transb, desc->m, desc->n, desc->k, desc->alpha, TECOBLAS_DATA_FLOAT,desc->lda, desc->strideA, TECOBLAS_DATA_FLOAT, desc->ldb, desc->strideB, desc->beta, TECOBLAS_DATA_FLOAT, desc->ldc, desc->strideC, desc->batch_count, TECOBLAS_SGEMM_STRIDED_BATCHED,reinterpret_cast<size_t*>(size));
    }
    
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
    if(desc->batch==0){
        if(desc->datatype == TECOBLAS_DATA_HALF)
            status = tecoblasHgemm(desc->handle, desc->transa, desc->transb, desc->m, desc->n, desc->k, 1.0f, a, desc->lda, b, desc->ldb, 0.0f, c, desc->ldc);
        else
            status = tecoblasSgemm(desc->handle, desc->transa, desc->transb, desc->m, desc->n, desc->k, 1.0f, a, desc->lda, b, desc->ldb, 0.0f, c, desc->ldc);
    }else{
        if(desc->datatype == TECOBLAS_DATA_HALF)
            status = tecoblasHgemmStridedBatched(desc->handle, desc->transa, desc->transb, desc->m, desc->n, desc->k, 1.0f, a, desc->lda,desc->strideA, b, desc->ldb,desc->strideB, 0.0f, c, desc->ldc,desc->strideC,desc->batch_count);
        else
            status = tecoblasSgemmStridedBatched(desc->handle, desc->transa, desc->transb, desc->m, desc->n, desc->k, 1.0f, a, desc->lda,desc->strideA, b, desc->ldb,desc->strideB, 0.0f, c, desc->ldc,desc->strideC,desc->batch_count);
    }
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

#include "matmul_tecoblas.h"

infiniopStatus_t tecoCreateMatmulDescriptor(TecoHandle_t handle, MatmulTecoDescriptor_t *desc_ptr, infiniopTensorDescriptor_t c_desc, float alpha, infiniopTensorDescriptor_t a_desc, infiniopTensorDescriptor_t b_desc, float beta) {
    infiniopStatus_t status = STATUS_SUCCESS;
    tecoblasDataType_t datatype;
    tecoblasOperation_t transA,transB,transC;
    uint64_t m,k,n;
    long long int lda,ldb,ldc;
    long long int batch,batch_count;
    long int strideA = 1,strideB = 1,strideC = 1;
    if (a_desc->ndim == 2 && b_desc->ndim == 2) {
        batch = 0;
        batch_count = 1;
    }else if(a_desc->ndim == 3 && b_desc->ndim == 3){
        batch = 1;
        batch_count = a_desc->shape[0];
        strideA = a_desc->strides[0];
        strideB = b_desc->strides[0];
        strideC = c_desc->strides[0];
    }else{
        return STATUS_BAD_TENSOR_SHAPE;
    }
    /*MatrixA*/
    if(a_desc->strides[1+batch] == 1 && (uint64_t)a_desc->strides[0+batch] >= a_desc->shape[1+batch]){
        transA = TECOBLAS_OP_N;
        m = a_desc->shape[0+batch];
        k = a_desc->shape[1+batch];
        lda = a_desc->strides[0+batch];
    }else if(a_desc->strides[0+batch] == 1 && (uint64_t)a_desc->strides[1+batch] >= a_desc->shape[0+batch]){
        transA = TECOBLAS_OP_T;
        m = a_desc->shape[0+batch];
        k = a_desc->shape[1+batch];
        lda = a_desc->strides[1+batch];
    }else{
        return STATUS_BAD_TENSOR_SHAPE; 
    }
    /*MatrixB*/
    if(b_desc->strides[1+batch] == 1 && (uint64_t)b_desc->strides[0+batch] >= b_desc->shape[1+batch]){
        transB = TECOBLAS_OP_N;
        k = b_desc->shape[0+batch];
        n = b_desc->shape[1+batch];
        ldb = b_desc->strides[0+batch];
    }else if(b_desc->strides[0+batch] == 1 && (uint64_t)b_desc->strides[1+batch] >= b_desc->shape[0+batch]){
        transB = TECOBLAS_OP_T;
        k = b_desc->shape[0+batch];
        n = b_desc->shape[1+batch];
        ldb = b_desc->strides[1+batch];
    }else{
        return STATUS_BAD_TENSOR_SHAPE; 
    }
    /*MatrixC*/
    if(c_desc->strides[1+batch] == 1 && (uint64_t)c_desc->strides[0+batch] >= c_desc->shape[1+batch]){
        transC = TECOBLAS_OP_N;
        m = c_desc->shape[0+batch];
        n = c_desc->shape[1+batch];
        ldc = c_desc->strides[0+batch];
    }else if(c_desc->strides[0+batch] == 1 && (uint64_t)c_desc->strides[1+batch] >= c_desc->shape[0+batch]){
        transC = TECOBLAS_OP_T;
        m = c_desc->shape[0+batch];
        n = c_desc->shape[1+batch];
        ldc = c_desc->strides[1+batch];
    }else{
        return STATUS_BAD_TENSOR_SHAPE; 
    }
    
    if(a_desc->dt==F16 && b_desc->dt==F16){
        datatype = TECOBLAS_DATA_HALF;
    }else if(a_desc->dt==F32 && b_desc->dt==F32){
        datatype = TECOBLAS_DATA_FLOAT;
    }else{
        return STATUS_BAD_TENSOR_DTYPE;
    }

    tecoblasHandle_t tecoblas_handle;
    tecoblasCreate(&tecoblas_handle);

    *desc_ptr = new MatmulTecoDescriptor{
        handle->device,
        handle->device_id,
        tecoblas_handle,
        handle->stream,
        datatype,
        transA,
        transB,
        transC,
        m,
        k,
        n,
        alpha,
        beta,
        lda,
        ldb,
        ldc,
        batch,
        batch_count,
        strideA,
        strideB,
        strideC,
        };
    tecoblasSetStream((*desc_ptr)->handle,(*desc_ptr)->stream);
        
    return status;
}

infiniopStatus_t tecoGetMatmulWorkspaceSize(MatmulTecoDescriptor_t desc, uint64_t *size) {
    tecoblasAPIName_t apiName;
    if (desc->batch == 0)
    {
        if(desc->datatype == TECOBLAS_DATA_HALF)
            apiName = TECOBLAS_HGEMM;
        else
            apiName = TECOBLAS_SGEMM;
    }else{
        if(desc->datatype == TECOBLAS_DATA_HALF)
            apiName = TECOBLAS_HGEMM_STRIDED_BATCHED;
        else
            apiName = TECOBLAS_SGEMM_STRIDED_BATCHED;
    }
    CHECK_TECOBLAS(tecoblasGetWorkspaceSize(
        desc->handle, 
        desc->transa, 
        desc->transb, 
        desc->m, 
        desc->n, 
        desc->k, 
        desc->alpha, 
        desc->datatype,
        desc->lda, 
        desc->strideA, 
        desc->datatype, 
        desc->ldb, 
        desc->strideB, 
        desc->beta, 
        desc->datatype, 
        desc->ldc, 
        desc->strideC, 
        desc->batch_count, 
        apiName,
        reinterpret_cast<size_t*>(size)))
    

        
    return STATUS_SUCCESS;
}

infiniopStatus_t tecoMatmul(MatmulTecoDescriptor_t desc, void *workspace, uint64_t workspace_size, void *c, const void *a, const void *b, void *stream) {
    tecoblasSetStream(desc->handle, desc->stream);
    tecoblasSetWorkspace(desc->handle, workspace, workspace_size);
    if(desc->batch==0){
        if(desc->datatype == TECOBLAS_DATA_HALF)
            CHECK_TECOBLAS(tecoblasHgemm(desc->handle, desc->transa, desc->transb, desc->m, desc->n, desc->k, desc->alpha, a, desc->lda, b, desc->ldb, desc->beta, c, desc->ldc))
        else
            CHECK_TECOBLAS(tecoblasSgemm(desc->handle, desc->transa, desc->transb, desc->m, desc->n, desc->k, desc->alpha, a, desc->lda, b, desc->ldb, desc->beta, c, desc->ldc))
    }else{
        if(desc->datatype == TECOBLAS_DATA_HALF)
            CHECK_TECOBLAS(tecoblasHgemmStridedBatched(desc->handle, desc->transa, desc->transb, desc->m, desc->n, desc->k, desc->alpha, a, desc->lda,desc->strideA, b, desc->ldb,desc->strideB, desc->beta, c, desc->ldc,desc->strideC,desc->batch_count))
        else
            CHECK_TECOBLAS(tecoblasSgemmStridedBatched(desc->handle, desc->transa, desc->transb, desc->m, desc->n, desc->k, desc->alpha, a, desc->lda,desc->strideA, b, desc->ldb,desc->strideB, desc->beta, c, desc->ldc,desc->strideC,desc->batch_count))
    }
    sdaaStreamSynchronize(desc->stream);
    return STATUS_SUCCESS;
}

infiniopStatus_t tecoDestroyMatmulDescriptor(MatmulTecoDescriptor_t desc) {
    return STATUS_SUCCESS;
}

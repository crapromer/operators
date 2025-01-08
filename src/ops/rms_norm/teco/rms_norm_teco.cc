#include "rms_norm_teco.h"


infiniopStatus_t tecoCreateRMSNormDescriptor(TecoHandle_t handle, RMSNormTecoDescriptor_t *desc_ptr, infiniopTensorDescriptor_t y_desc, infiniopTensorDescriptor_t x_desc, infiniopTensorDescriptor_t w_desc, float epsilon) {
    if (y_desc->ndim != 2 || x_desc->ndim != 2 || w_desc->ndim != 1) {
        return STATUS_BAD_TENSOR_SHAPE;
    }

    auto n = y_desc->shape[0],
         c = y_desc->shape[1];
    unsigned long h = 1,
         w = 1;

    if (x_desc->shape[0] != n || x_desc->shape[1] != c || w_desc->shape[0] != c) {
        return STATUS_BAD_TENSOR_SHAPE;
    }

    tecodnnHandle_t tecodnn_handle;
    tecodnnCreate(&tecodnn_handle);
    // sdaaStream_t stream;
    // sdaaStreamCreate(&stream);
    tecodnnTensorDescriptor_t x_desc_teco,y_desc_teco,w_desc_teco,rms_desc_teco;
    tecodnnCreateTensorDescriptor(&x_desc_teco);
    tecodnnCreateTensorDescriptor(&y_desc_teco);
    tecodnnCreateTensorDescriptor(&w_desc_teco);
    tecodnnCreateTensorDescriptor(&rms_desc_teco);
    // toTecodnnTensorDescriptor(x_desc,x_desc_teco);
    // toTecodnnTensorDescriptor(y_desc,y_desc_teco);
    // toTecodnnTensorDescriptor(w_desc,w_desc_teco);
    // tecodnnSetTensor4dDescriptor(x_desc_teco,TECODNN_TENSOR_NCHW,TECODNN_DATA_HALF,x_desc->shape[0],1,1,x_desc->shape[1]);
    // tecodnnSetTensor4dDescriptor(y_desc_teco,TECODNN_TENSOR_NCHW,TECODNN_DATA_HALF,y_desc->shape[0],1,1,y_desc->shape[1]);
    // if(w_desc->dt==F16)
    //     tecodnnSetTensor4dDescriptor(w_desc_teco,TECODNN_TENSOR_NCHW,TECODNN_DATA_HALF,1,1,1,w_desc->shape[0]);
    // if(w_desc->dt==F32)
    //     tecodnnSetTensor4dDescriptor(w_desc_teco,TECODNN_TENSOR_NCHW,TECODNN_DATA_FLOAT,1,1,1,w_desc->shape[0]);
    // tecodnnSetTensor4dDescriptor(rms_desc_teco,TECODNN_TENSOR_NCHW,TECODNN_DATA_FLOAT,n,1,1,1);
    
    if(w_desc->dt==F16){
        tecodnnSetTensor4dDescriptor(x_desc_teco,TECODNN_TENSOR_NCHW,TECODNN_DATA_HALF,n,h,w,c);
        tecodnnSetTensor4dDescriptor(y_desc_teco,TECODNN_TENSOR_NCHW,TECODNN_DATA_HALF,n,h,w,c);
        tecodnnSetTensor4dDescriptor(w_desc_teco,TECODNN_TENSOR_NCHW,TECODNN_DATA_HALF,1,1,1,c);
        tecodnnSetTensor4dDescriptor(rms_desc_teco,TECODNN_TENSOR_NCHW,TECODNN_DATA_FLOAT,n,h,w,1);
    }
        
    if(w_desc->dt==F32){
        tecodnnSetTensor4dDescriptor(x_desc_teco,TECODNN_TENSOR_NCHW,TECODNN_DATA_HALF,n,h,w,c);
        tecodnnSetTensor4dDescriptor(y_desc_teco,TECODNN_TENSOR_NCHW,TECODNN_DATA_HALF,n,h,w,c);
        tecodnnSetTensor4dDescriptor(w_desc_teco,TECODNN_TENSOR_NCHW,TECODNN_DATA_FLOAT,1,1,1,c);
        tecodnnSetTensor4dDescriptor(rms_desc_teco,TECODNN_TENSOR_NCHW,TECODNN_DATA_FLOAT,n,h,w,1);
    }
    *desc_ptr = new RMSNormTecoDescriptor{
        handle->device,
        tecodnn_handle,
        handle->stream,
        epsilon,
        x_desc_teco,
        y_desc_teco,
        w_desc_teco,
        rms_desc_teco,
        n,
        c,
        };
    tecodnnSetStream((*desc_ptr)->handle,(*desc_ptr)->stream);
    return STATUS_SUCCESS;
}

infiniopStatus_t tecoGetRMSNormWorkspaceSize(RMSNormTecoDescriptor_t desc, uint64_t *size) {
    *size = (desc->n)*(desc->c)*4;
    return STATUS_SUCCESS;
}

infiniopStatus_t tecoRMSNorm(RMSNormTecoDescriptor_t desc, void *workspace, uint64_t workspace_size, void *y, void const *x, void const *w, void *stream) {
    tecodnnSetStream(desc->handle, desc->stream);
    tecodnnStatus_t status;
    
    // void *rms = malloc(workspace_size * sizeof(uint16_t));
    status = tecodnnRMSNormForward(desc->handle, desc->eps, desc->xDesc,x,desc->wDesc,w,desc->yDesc,y,desc->rmsDesc,workspace);
    sdaaStreamSynchronize(desc->stream);
    if (status != TECODNN_STATUS_SUCCESS) {
        printf("%s\n",tecodnnGetErrorString(status));
        return STATUS_EXECUTION_FAILED;
    }else{
        return STATUS_SUCCESS;
    }
}

infiniopStatus_t tecoDestroyRMSNormDescriptor(RMSNormTecoDescriptor_t desc) {
    delete desc;
    return STATUS_SUCCESS;
}

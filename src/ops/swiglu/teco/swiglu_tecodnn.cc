#include "swiglu_tecodnn.h"

infiniopStatus_t tecoCreateSwiGLUDescriptor(TecoHandle_t handle, SwiGLUTecoDescriptor_t *desc_ptr, infiniopTensorDescriptor_t c_desc, infiniopTensorDescriptor_t a_desc,infiniopTensorDescriptor_t b_desc) {
    //create tecodnn hanele
    long int N = c_desc->shape[0],C = c_desc->shape[1];
    tecodnnHandle_t tecodnn_handle;
    tecodnnCreate(&tecodnn_handle);

    tecodnnActivationDescriptor_t activationDesc;
    tecodnnCreateActivationDescriptor(&activationDesc);
    tecodnnSetActivationDescriptor(activationDesc, TECODNN_ACTIVATION_SILU, TECODNN_NOT_PROPAGATE_NAN, 0.0);

    tecodnnTensorDescriptor_t aDesc,bDesc,cDesc;
    tecodnnCreateTensorDescriptor(&aDesc);
    tecodnnCreateTensorDescriptor(&bDesc);
    tecodnnCreateTensorDescriptor(&cDesc);

    tecodnnSetTensor4dDescriptor(aDesc,TECODNN_TENSOR_NCHW,TECODNN_DATA_HALF,N,1,1,C);
    tecodnnSetTensor4dDescriptor(bDesc,TECODNN_TENSOR_NCHW,TECODNN_DATA_HALF,N,1,1,C);
    tecodnnSetTensor4dDescriptor(cDesc,TECODNN_TENSOR_NCHW,TECODNN_DATA_HALF,N,1,1,C);

    *desc_ptr = new SwiGLUTecoDescriptor{
        handle->device,
        handle->device_id,
        handle->stream,
        tecodnn_handle,
        activationDesc,
        aDesc,
        bDesc,
        cDesc,
    };
    return STATUS_SUCCESS;
}

infiniopStatus_t tecoSwiGLU(SwiGLUTecoDescriptor_t desc, void *c, void const *a, void const *b, void *stream) {
    float alpha = 1.0f,beta = 0.0f;
    tecodnnActivationForward(desc->handle,desc->activationDesc,(void*)&alpha,desc->bDesc,b,(void*)&beta,desc->cDesc,c);
    tecodnnMulTensorEx(desc->handle, desc->aDesc, a, desc->cDesc, c, desc->cDesc, c);
    return STATUS_SUCCESS;
}

infiniopStatus_t tecoDestroySwiGLUDescriptor(SwiGLUTecoDescriptor_t desc) {
    return STATUS_SUCCESS;
}

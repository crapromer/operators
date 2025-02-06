#include "rearrange_tecodnn.h"

infiniopStatus_t tecoCreateRearrangeDescriptor(TecoHandle_t handle, RearrangeTecoDescriptor_t *desc_ptr, infiniopTensorDescriptor_t dst, infiniopTensorDescriptor_t src) {
    tecodnnHandle_t tecodnn_handle;
    tecodnnCreate(&tecodnn_handle);

    tecodnnTensorDescriptor_t srcDesc,dstDesc;
    tecodnnCreateTensorDescriptor(&srcDesc);
    tecodnnCreateTensorDescriptor(&dstDesc);

    int nbDims = dst->ndim;

    int *shape = new int[nbDims];
    int *src_strides = new int[nbDims];
    int *dst_strides = new int[nbDims];
    for (size_t i = 0; i < (size_t)nbDims; i++)
    {
        shape[i] = dst->shape[i];
        src_strides[i] = src->strides[i];
        dst_strides[i] = dst->strides[i];
    }

    tecodnnSetTensorNdDescriptor(srcDesc, TECODNN_DATA_HALF, nbDims, shape, src_strides);
    tecodnnSetTensorNdDescriptor(dstDesc, TECODNN_DATA_HALF, nbDims, shape, dst_strides);
    
    
    *desc_ptr = new RearrangeTecoDescriptor{
        DevTecoSDAA,
        handle->device_id,
        handle->stream,
        tecodnn_handle,
        nbDims,
        shape,
        src_strides,
        dst_strides,
        srcDesc,
        dstDesc,
    };


    return STATUS_SUCCESS;
}

infiniopStatus_t tecoRearrange(RearrangeTecoDescriptor_t desc, void *dst, void const *src, void *stream) {
    tecodnnCopyStride(desc->handle, desc->srcDesc, src, desc->dstDesc, dst);
    return STATUS_SUCCESS;
}

infiniopStatus_t tecoDestroyRearrangeDescriptor(RearrangeTecoDescriptor_t desc) {
    return STATUS_SUCCESS;
}

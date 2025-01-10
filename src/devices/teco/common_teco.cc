#include "common_teco.h"
void const** convertToBatch(void const* data, int batch, int stride, size_t typeSize){
    void const **output = (void const **)malloc(batch * sizeof(void const *));
    if (output == NULL) {
        return NULL;
    }

    const uint8_t *charData = (const uint8_t *)data;

    for (int i = 0; i < batch; i++) {
        output[i] = (const void *)(charData + i * stride * typeSize);
    }

    return output;
}

bool is_contiguous(infiniopTensorDescriptor_t desc) {
    uint64_t ndim = desc->ndim;
    if (desc->strides[ndim-1] != 1) {
        return false;
    }else
        return true;  
}

infiniopStatus_t restoreTensor(infiniopTensorDescriptor_t desc, void *data,tecodnnDataType_t datatype) {
    tecodnnHandle_t tecodnn_handle;
    tecodnnCreate(&tecodnn_handle);
    tecodnnTensorDescriptor_t src,dst;
    tecodnnCreateTensorDescriptor(&src);
    tecodnnCreateTensorDescriptor(&dst);
    int *strides = new int[desc->ndim];
    int *old_strides = new int[desc->ndim];
    int *shape = new int[desc->ndim];
    strides[desc->ndim - 1] = 1;  // 最后一维的 stride 为 1
    old_strides[desc->ndim - 1] = desc->strides[desc->ndim - 1];
    shape[desc->ndim - 1] = desc->shape[desc->ndim - 1];
    for (int i = desc->ndim - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * desc->shape[i + 1];  // 当前维度的 stride
        shape[i] = desc->shape[i];
        old_strides[i] = desc->strides[i];
    }
    size_t size = strides[0]*desc->shape[0];
    if(datatype==TECODNN_DATA_HALF)
        size*=sizeof(uint16_t);
    else
        size*=sizeof(uint32_t);
    void *temp;
    sdaaMalloc(&temp,size);
    tecodnnSetTensorNdDescriptor(src,datatype,desc->ndim,shape,strides);
    tecodnnSetTensorNdDescriptor(dst,datatype,desc->ndim,shape,old_strides);
    tecodnnCopyStride(tecodnn_handle,src,data,dst,temp);
    sdaaMemcpy(data, temp, size, sdaaMemcpyDeviceToDevice);
    sdaaFree(temp);

    return STATUS_SUCCESS;
}

infiniopStatus_t toContiguous(infiniopTensorDescriptor_t desc, void *data,tecodnnDataType_t datatype) {
    tecodnnHandle_t tecodnn_handle;
    tecodnnCreate(&tecodnn_handle);
    tecodnnTensorDescriptor_t src,dst;
    tecodnnCreateTensorDescriptor(&src);
    tecodnnCreateTensorDescriptor(&dst);
    int *strides = new int[desc->ndim];
    int *old_strides = new int[desc->ndim];
    int *shape = new int[desc->ndim];
    strides[desc->ndim - 1] = 1; 
    old_strides[desc->ndim - 1] = desc->strides[desc->ndim - 1];
    shape[desc->ndim - 1] = desc->shape[desc->ndim - 1];
    for (int i = desc->ndim - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * desc->shape[i + 1]; 
        shape[i] = desc->shape[i];
        old_strides[i] = desc->strides[i];
    }
    size_t size = strides[0]*desc->shape[0];
    if(datatype==TECODNN_DATA_HALF){
        size*=sizeof(uint16_t);
    }
    else{
        size*=sizeof(uint32_t);
    }
    void *temp;
    sdaaMalloc(&temp,size);
    tecodnnSetTensorNdDescriptor(src,datatype,desc->ndim,shape,old_strides);
    tecodnnSetTensorNdDescriptor(dst,datatype,desc->ndim,shape,strides);
    tecodnnCopyStride(tecodnn_handle,src,data,dst,temp);
    sdaaMemcpy(data, temp, size, sdaaMemcpyDeviceToDevice);
    sdaaFree(temp);

    return STATUS_SUCCESS;
}

infiniopStatus_t toTecodnnTensorDescriptor(infiniopTensorDescriptor_t src, tecodnnTensorDescriptor_t des) {
    tecodnnDataType_t data_type;
    if(src->dt==F16)
        data_type = TECODNN_DATA_HALF;
    tecodnnSetTensor4dDescriptor(des,TECODNN_TENSOR_NCHW,data_type,src->shape[0],src->shape[1],1,1);
    return STATUS_SUCCESS;
}



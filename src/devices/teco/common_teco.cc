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

bool is_contiguous(MatrixInfo desc) {
    if (desc.ei!= 1) {
        return false;
    }else
        return true;  
}

infiniopStatus_t restoreTensor(MatrixInfo desc, void *data,tecodnnDataType_t datatype) {
    tecodnnHandle_t tecodnn_handle;
    tecodnnCreate(&tecodnn_handle);
    tecodnnTensorDescriptor_t src,dst;
    tecodnnCreateTensorDescriptor(&src);
    tecodnnCreateTensorDescriptor(&dst);
    int *dst_strides = new int[desc.ndim];
    int *src_strides = new int[desc.ndim];
    int *shape = new int[desc.ndim];
    dst_strides[0] = desc.cols;
    dst_strides[1] = 1; 
    src_strides[0] = desc.ld;
    src_strides[1] = desc.ei;
    shape[0] = desc.rows;
    shape[1] = desc.cols;
    size_t size = shape[1]*shape[0];
    if(datatype==TECODNN_DATA_HALF)
        size*=sizeof(uint16_t);
    else
        size*=sizeof(uint32_t);
    void *temp;
    sdaaMalloc(&temp,size);
    tecodnnSetTensorNdDescriptor(src,datatype,desc.ndim,shape,dst_strides);
    tecodnnSetTensorNdDescriptor(dst,datatype,desc.ndim,shape,src_strides);
    tecodnnCopyStride(tecodnn_handle,src,data,dst,temp);
    sdaaMemcpy(data, temp, size, sdaaMemcpyDeviceToDevice);
    sdaaFree(temp);

    return STATUS_SUCCESS;
}

infiniopStatus_t toContiguous(MatrixInfo desc, void *data,tecodnnDataType_t datatype) {
    tecodnnHandle_t tecodnn_handle;
    tecodnnCreate(&tecodnn_handle);
    tecodnnTensorDescriptor_t src,dst;
    tecodnnCreateTensorDescriptor(&src);
    tecodnnCreateTensorDescriptor(&dst);
    int *dst_strides = new int[desc.ndim];
    int *src_strides = new int[desc.ndim];
    int *shape = new int[desc.ndim];
    dst_strides[0] = desc.cols;
    dst_strides[1] = 1; 
    src_strides[0] = desc.ld;
    src_strides[1] = desc.ei;
    shape[0] = desc.rows;
    shape[1] = desc.cols;
    size_t size = shape[1]*shape[0];
    if(datatype==TECODNN_DATA_HALF){
        size*=sizeof(uint16_t);
    }
    else{
        size*=sizeof(uint32_t);
    }
    void *temp;
    sdaaMalloc(&temp,size);
    tecodnnSetTensorNdDescriptor(src,datatype,desc.ndim,shape,src_strides);
    tecodnnSetTensorNdDescriptor(dst,datatype,desc.ndim,shape,dst_strides);
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



#ifndef __TECO_REARRANGE_H__
#define __TECO_REARRANGE_H__

#include "operators.h"
#include <sdaa_runtime.h>
#include <tecodnn.h>
#include "../../../devices/teco/teco_handle.h"
struct RearrangeTecoDescriptor {
    Device device;
    int device_id;
    sdaaStream_t stream;
    tecodnnHandle_t handle;
    int nbDims;
    int *shape,*src_strides,*dst_strides;
    tecodnnTensorDescriptor_t srcDesc,dstDesc;
};

typedef struct RearrangeTecoDescriptor *RearrangeTecoDescriptor_t;

infiniopStatus_t tecoCreateRearrangeDescriptor(TecoHandle_t handle,
                                             RearrangeTecoDescriptor_t *desc_ptr,
                                             infiniopTensorDescriptor_t dst,                                             
                                             infiniopTensorDescriptor_t src);

infiniopStatus_t tecoRearrange(RearrangeTecoDescriptor_t desc,
                            void *dst,
                            void const *src,
                            void *stream);

infiniopStatus_t tecoDestroyRearrangeDescriptor(RearrangeTecoDescriptor_t desc);


#endif
#ifndef __BANG_RMS_NORM_H__
#define __BANG_RMS_NORM_H__

#include "../../../devices/bang/bang_handle.h"
#include "../../utils.h"
#include "operators.h"

struct RMSNormBangDescriptor {
    Device device;
    int device_id;
    DT dtype;
    uint64_t n;
    uint64_t d;
    uint64_t stride_y;
    uint64_t stride_x;
    DT w_datatype;
    float epsilon;
};

typedef struct RMSNormBangDescriptor *RMSNormBangDescriptor_t;

infiniopStatus_t bangCreateRMSNormDescriptor(BangHandle_t handle,
                                             RMSNormBangDescriptor_t *desc_ptr,
                                             infiniopTensorDescriptor_t y_desc,
                                             infiniopTensorDescriptor_t x_desc,
                                             infiniopTensorDescriptor_t w_desc,
                                             float epsilon);

infiniopStatus_t bangGetRMSNormWorkspaceSize(RMSNormBangDescriptor_t desc, uint64_t *size);

infiniopStatus_t bangRMSNorm(RMSNormBangDescriptor_t desc,
                             void *workspace,
                             uint64_t workspace_size,
                             void *y, void const *x, void const *w,
                             void *stream);

infiniopStatus_t bangDestroyRMSNormDescriptor(RMSNormBangDescriptor_t desc);

#endif// __BANG_RMS_NORM_H__

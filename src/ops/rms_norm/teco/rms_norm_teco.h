#ifndef __TECO_RMS_NORM_H__
#define __TECO_RMS_NORM_H__

#include "operators.h"
#include <sdaa_runtime.h>
#include <tecodnn.h>
#include "../../../devices/teco/teco_handle.h"

struct RMSNormTecoDescriptor {
    Device device;
    tecodnnHandle_t handle;
    sdaaStream_t stream;
    float eps;
    tecodnnTensorDescriptor_t xDesc,yDesc,wDesc,rmsDesc;
    unsigned long n,c;
};

typedef struct RMSNormTecoDescriptor *RMSNormTecoDescriptor_t;

infiniopStatus_t tecoCreateRMSNormDescriptor(TecoHandle_t handle, RMSNormTecoDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t w_desc, float epsilon);

infiniopStatus_t tecoGetRMSNormWorkspaceSize(RMSNormTecoDescriptor_t desc, uint64_t *size);

infiniopStatus_t tecoRMSNorm(RMSNormTecoDescriptor_t desc,
                                  void *workspace,
                                  uint64_t workspace_size,
                                  void *y, void const *x, void const *w, 
                                  void *stream);

infiniopStatus_t tecoDestroyRMSNormDescriptor(RMSNormTecoDescriptor_t desc);

#endif
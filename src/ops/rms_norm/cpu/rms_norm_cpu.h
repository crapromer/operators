﻿#ifndef __CPU_RMS_NORM_H__
#define __CPU_RMS_NORM_H__

#include "operators.h"

struct RMSNormCpuDescriptor {
    Device device;
    DT dtype;
    uint64_t n;
    uint64_t d;
    uint64_t stride_y;
    uint64_t stride_x;
    DT w_datatype;
    float epsilon;
};

typedef struct RMSNormCpuDescriptor *RMSNormCpuDescriptor_t;

infiniopStatus_t cpuCreateRMSNormDescriptor(infiniopHandle_t handle, RMSNormCpuDescriptor_t *desc_ptr,
                                            infiniopTensorDescriptor_t y_desc,
                                            infiniopTensorDescriptor_t x_desc,
                                            infiniopTensorDescriptor_t w_desc, float epsilon);

infiniopStatus_t cpuGetRMSNormWorkspaceSize(RMSNormCpuDescriptor_t desc, uint64_t *size);

infiniopStatus_t cpuRMSNorm(RMSNormCpuDescriptor_t desc,
                            void *workspace,
                            uint64_t workspace_size,
                            void *y, void const *x, void const *w,
                            void *stream);

infiniopStatus_t cpuDestroyRMSNormDescriptor(RMSNormCpuDescriptor_t desc);

#endif// __CPU_RMS_NORM_H__

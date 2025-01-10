#ifndef _COMMON_TECO_
#define _COMMON_TECO_

#include <stdio.h>
#include <stdlib.h>
#include <sdaa_runtime.h>
#include <tecoblas.h>
#include <tecodnn.h>
#include "device.h"
#include "operators.h"
#include <iostream>
#define CHECK_TECOBLAS(expression)                                                               \
    {                                                                                            \
        tecoblasStatus_t status = (expression);                                                  \
        if (status != TECOBLAS_STATUS_SUCCESS) {                                                 \
            fprintf(stderr, "Error at line %d: %s\n", __LINE__, tecoblasGetErrorString(status)); \
            exit(EXIT_FAILURE);                                                                  \
        }                                                                                        \
    }
void const** convertToBatch(void const* data, int batch, int stride, size_t typeSize);
bool is_contiguous(infiniopTensorDescriptor_t desc);
infiniopStatus_t toContiguous(infiniopTensorDescriptor_t desc,void *data,tecodnnDataType_t datatype);
infiniopStatus_t restoreTensor(infiniopTensorDescriptor_t desc,void *data,tecodnnDataType_t datatype);

infiniopStatus_t toTecodnnTensorDescriptor(infiniopTensorDescriptor_t src,tecodnnTensorDescriptor_t des);

#endif
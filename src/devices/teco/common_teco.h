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

typedef struct MatrixInfo {
    int ndim;
    int batch;
    int64_t stride;
    int rows;
    int cols;
    int ld;
    int ei;

    MatrixInfo() {}

    MatrixInfo(infiniopTensorDescriptor_t layout, infiniopStatus_t *status) {
        if (layout->ndim == 2) {
            this->ndim = 2;
            this->batch = 1;
            this->stride = 0;
            this->rows = layout->shape[0];
            this->cols = layout->shape[1];
            this->ld = layout->strides[0];
            this->ei = layout->strides[1];
        } else if (layout->ndim == 3) {
            this->ndim = 3;
            this->batch = layout->shape[0];
            this->stride = this->batch == 1 ? 0 : layout->strides[0];
            this->rows = layout->shape[1];
            this->cols = layout->shape[2];
            this->ld = layout->strides[1];
            this->ei = layout->strides[2];
        } else {
            *status = STATUS_BAD_TENSOR_SHAPE;
            return;
        }

        *status = STATUS_SUCCESS;
    }

} MatrixInfo;
void const** convertToBatch(void const* data, int batch, int stride, size_t typeSize);
bool is_contiguous(MatrixInfo desc);
infiniopStatus_t toContiguous(MatrixInfo desc,void *data,tecodnnDataType_t datatype);
infiniopStatus_t restoreTensor(MatrixInfo desc,void *data,tecodnnDataType_t datatype);

infiniopStatus_t toTecodnnTensorDescriptor(infiniopTensorDescriptor_t src,tecodnnTensorDescriptor_t des);

#endif
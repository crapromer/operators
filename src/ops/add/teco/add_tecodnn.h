#ifndef __TECO_ADD_H__
#define __TECO_ADD_H__

#include "operators.h"
#include <sdaa_runtime.h>
#include <tecoblas.h>
#include <tecodnn.h>
#include "../../../devices/teco/teco_handle.h"

struct AddTecoDescriptor {
    Device device;
    int device_id;
    tecodnnHandle_t handle;
    sdaaStream_t stream;
    tecoblasOperation_t transa,transb;
    int m,n,k;
    float alpha,beta;
    int lda,ldb,ldc;
    int batch;
    long long int strideA,strideB,strideC;
};

typedef struct AddTecoDescriptor *AddTecoDescriptor_t;


infiniopStatus_t tecoCreateAddDescriptor(TecoHandle_t handle,
                                             AddTecoDescriptor_t *desc_ptr,
                                             infiniopTensorDescriptor_t c_desc
                                             infiniopTensorDescriptor_t a_desc,
                                             infiniopTensorDescriptor_t b_desc);

infiniopStatus_t tecoAdd(AddTecoDescriptor_t desc,
                             void *c,
                             const void *a,
                             const void *b,
                             void *stream);

infiniopStatus_t tecoDestroyAddDescriptor(AddTecoDescriptor_t desc);

#endif
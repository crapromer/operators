#ifndef __TECO_MATMUL_H__
#define __TECO_MATMUL_H__
#include "operators.h"
#include <sdaa_runtime.h>
#include <tecoblas.h>
#include <tecodnn.h>
#include "../../../devices/teco/teco_handle.h"
struct MatmulTecoDescriptor {
    Device device;
    int device_id;
    tecoblasHandle_t handle;
    sdaaStream_t stream;
    tecoblasDataType_t datatype;
    tecoblasOperation_t transa,transb;
    uint64_t m,k,n;
    float alpha,beta;
    long long int lda,ldb,ldc;
    long long int batch,batch_count;
    long int strideA,strideB,strideC;
    MatrixInfo a_desc,b_desc,c_desc;
};

typedef struct MatmulTecoDescriptor *MatmulTecoDescriptor_t;

infiniopStatus_t tecoCreateMatmulDescriptor(TecoHandle_t handle,
                                             MatmulTecoDescriptor_t *desc_ptr,
                                             infiniopTensorDescriptor_t c_desc,
                                             float alpha,
                                             infiniopTensorDescriptor_t a_desc,
                                             infiniopTensorDescriptor_t b_desc,
                                             float beta);

infiniopStatus_t tecoGetMatmulWorkspaceSize(MatmulTecoDescriptor_t desc,
                                             uint64_t *size);

infiniopStatus_t tecoMatmul(MatmulTecoDescriptor_t desc,
                             void *workspace,
                             uint64_t workspace_size,
                             void *c,
                             const void *a,
                             const void *b,
                             void *stream);

infiniopStatus_t tecoDestroyMatmulDescriptor(MatmulTecoDescriptor_t desc);


#endif
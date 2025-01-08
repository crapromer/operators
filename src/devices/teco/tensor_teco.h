#ifndef __TECO_TENSOR__
#define __TECO_TENSOR__

#include "operators.h"
#include "tensor.h"
#include <sdaa_runtime.h>
#include <tecoblas.h>
#include <vector>

struct tecoTensorDescriptor {
    uint64_t ndim;
    int64_t *shape;
    int64_t *strides;
    tecoblasDataType_t data_type;
    tecodnnDataType_t data_type;
    tecodnnTensorDescriptor_t 
    infiniopStatus_t fromInfiniOpTensorDescriptor(infiniopTensorDescriptor_t y_desc);
    infiniopStatus_t createTensor();
    infiniopStatus_t destroyTensor();
    ~tecoTensorDescriptor();

};

typedef tecoTensorDescriptor *tecoTensorDescriptor_t;

#endif
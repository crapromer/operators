#include "tensor_teco.h"


infiniopStatus_t tecoTensorDescriptor::fromInfiniOpTensorDescriptor(infiniopTensorDescriptor_t y_desc) {
    uint64_t ndim = y->ndim;
    // Cast shape type
    auto shape = new std::vector<int64_t>(ndim);
    auto strides = new std::vector<int64_t>(ndim);
    for (uint64_t i = 0; i < ndim; ++i) {
        (*shape)[i] = static_cast<int64_t>(y->shape[i]);
        (*strides)[i] = y->strides[i];
    }
    tecoblasDataType_t dt;
    if (dtype_eq(y->dt, F16)) {
        dt = tecoblasDataType_t::TECOBLAS_DATA_FLOAT;
    } else if (dtype_eq(y->dt, F32)) {
        dt = aclDataType::TECOBLAS_DATA_DOUBLE;
    } else {
        return STATUS_BAD_TENSOR_DTYPE;
    }

    return STATUS_SUCCESS;
}
#include "../utils.h"
#include "ops/matmul/matmul.h"
#include "ops/mlp/mlp.h"
#include "ops/swiglu/swiglu.h"
#include "tensor/tensor_descriptor.h"

struct _MLPDescriptor {
    Device device;
    infiniopMatmulDescriptor_t matmul_desc1;
    infiniopMatmulDescriptor_t matmul_desc2;
    infiniopSwiGLUDescriptor_t swiglu_desc;
    uint64_t w2_offset_by_bytes;
    uint64_t workspace_size;
    uint64_t matmul1_workspace_size;
    uint64_t matmul2_workspace_size;
    uint64_t matmul1_tensor_size;
    uint64_t swiglu_tensor_size;
};

typedef struct _MLPDescriptor *_MLPDescriptor_t;

__C __export infiniopStatus_t infiniopCreateMLPDescriptor(infiniopHandle_t handle,
                                                          infiniopMLPDescriptor_t *desc_ptr,
                                                          infiniopTensorDescriptor_t y_desc,
                                                          infiniopTensorDescriptor_t x_desc,
                                                          infiniopTensorDescriptor_t w12_desc,
                                                          infiniopTensorDescriptor_t w3_desc,
                                                          float alpha,
                                                          char residual) {
    if (y_desc->ndim != 2 || x_desc->ndim != 2 || w12_desc->ndim != 2 || w3_desc->ndim != 2) {
        return STATUS_BAD_TENSOR_SHAPE;
    }

    if (x_desc->strides[1] != 1 || y_desc->strides[1] != 1) {
        return STATUS_BAD_TENSOR_STRIDES;
    }

    // matmul1 desc
    infiniopTensorDescriptor_t desc1 = new TensorDescriptor;
    uint64_t shape1[2] = {x_desc->shape[0], w12_desc->shape[1]};// [num_tokens, 2 * intermediate_size]
    CHECK_STATUS(infiniopCreateTensorDescriptor(&desc1, 2, shape1, nullptr, x_desc->dt), STATUS_SUCCESS);
    infiniopMatmulDescriptor_t matmul_desc1 = new MatmulDescriptor{handle->device};
    CHECK_STATUS(infiniopCreateMatmulDescriptor(handle, &matmul_desc1, desc1, 1.0, x_desc, w12_desc, 0.0), STATUS_SUCCESS);
    uint64_t matmul1_tensor_size = get_byte_size(desc1);
    uint64_t matmul1_workspace_size = 0;
    CHECK_STATUS(infiniopGetMatmulWorkspaceSize(matmul_desc1, &matmul1_workspace_size), STATUS_SUCCESS);

    // swiglu desc
    infiniopTensorDescriptor_t desc2 = new TensorDescriptor;
    uint64_t w2_offset_by_bytes = w12_desc->shape[1] / 2 * w12_desc->dt.size;
    uint64_t shape2[2] = {x_desc->shape[0], w12_desc->shape[1] / 2};// [num_tokens, itermediate_size]
    CHECK_STATUS(infiniopCreateTensorDescriptor(&desc2, 2, shape2, nullptr, x_desc->dt), STATUS_SUCCESS);
    infiniopTensorDescriptor_t desc3 = new TensorDescriptor;
    int64_t strides3[2] = {desc1->strides[0], desc1->strides[1]};
    CHECK_STATUS(infiniopCreateTensorDescriptor(&desc3, 2, shape2, strides3, x_desc->dt), STATUS_SUCCESS);
    infiniopSwiGLUDescriptor_t swiglu_desc = new SwiGLUDescriptor{handle->device};
    CHECK_STATUS(infiniopCreateSwiGLUDescriptor(handle, &swiglu_desc, desc2, desc3, desc3), STATUS_SUCCESS);
    uint64_t swiglu_tensor_size = get_byte_size(desc2);

    // matmul2 desc
    infiniopMatmulDescriptor_t matmul_desc2 = new MatmulDescriptor{handle->device};
    CHECK_STATUS(infiniopCreateMatmulDescriptor(handle, &matmul_desc2, y_desc, alpha, desc2, w3_desc, residual ? 1.0 : 0.0), STATUS_SUCCESS);
    uint64_t matmul2_workspace_size = 0;
    CHECK_STATUS(infiniopGetMatmulWorkspaceSize(matmul_desc2, &matmul2_workspace_size), STATUS_SUCCESS);

    // calculate workspace size
    uint64_t workspace_size = std::max(std::max(matmul1_workspace_size + matmul1_tensor_size,
                                                matmul1_tensor_size + swiglu_tensor_size),
                                       swiglu_tensor_size + matmul2_workspace_size);

    // create descriptor
    *(_MLPDescriptor_t *) desc_ptr = new _MLPDescriptor{
        handle->device,
        matmul_desc1,
        matmul_desc2,
        swiglu_desc,
        w2_offset_by_bytes,
        workspace_size,
        matmul1_workspace_size,
        matmul2_workspace_size,
        matmul1_tensor_size,
        swiglu_tensor_size};

    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopGetMLPWorkspaceSize(infiniopMLPDescriptor_t desc, uint64_t *size) {
    // compute order: matmul1, swiglu, matmul2
    *size = ((_MLPDescriptor_t) desc)->workspace_size;
    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopMLP(infiniopMLPDescriptor_t desc,
                                          void *workspace,
                                          uint64_t workspace_size,
                                          void *y,
                                          void const *x,
                                          void const *w12,
                                          void const *w3,
                                          void *stream) {
    auto _desc = (_MLPDescriptor_t) desc;
    if (workspace_size < _desc->workspace_size) {
        return STATUS_MEMORY_NOT_ALLOCATED;
    }

    CHECK_STATUS(infiniopMatmul(_desc->matmul_desc1,
                                (char *) workspace + _desc->matmul1_tensor_size,
                                workspace_size - _desc->matmul1_tensor_size,
                                workspace, x, w12, stream),
                 STATUS_SUCCESS);
    CHECK_STATUS(infiniopSwiGLU(_desc->swiglu_desc,
                                (char *) workspace + _desc->matmul1_tensor_size,
                                (char *) workspace + _desc->w2_offset_by_bytes,
                                workspace, stream),
                 STATUS_SUCCESS);
    CHECK_STATUS(infiniopMatmul(_desc->matmul_desc2, (char *) workspace + _desc->matmul1_tensor_size + _desc->swiglu_tensor_size,
                                workspace_size - _desc->matmul1_tensor_size - _desc->swiglu_tensor_size,
                                y, (char *) workspace + _desc->matmul1_tensor_size, w3, stream),
                 STATUS_SUCCESS);

    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopDestroyMLPDescriptor(infiniopMLPDescriptor_t desc) {
    CHECK_STATUS(infiniopDestroyMatmulDescriptor(((_MLPDescriptor_t) desc)->matmul_desc1), STATUS_SUCCESS);
    CHECK_STATUS(infiniopDestroyMatmulDescriptor(((_MLPDescriptor_t) desc)->matmul_desc2), STATUS_SUCCESS);
    CHECK_STATUS(infiniopDestroySwiGLUDescriptor(((_MLPDescriptor_t) desc)->swiglu_desc), STATUS_SUCCESS);

    return STATUS_SUCCESS;
}

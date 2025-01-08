#include "causal_softmax_aclnn.h"
#include "../../utils.h"

CausalSoftmaxAclnnDescriptor::CausalSoftmaxAclnnDescriptor(Device _device) {
    device = _device;
    device_id = 0;
    aDesc = new aclnnTensorDescriptor();
    maskDesc = new aclnnTensorDescriptor();
    outDesc = new aclnnTensorDescriptor();
    executor = nullptr;
    workspaceSize = 0;
    maskAddr = nullptr;
}

infiniopStatus_t aclnnCreateCausalSoftmaxDescriptor(AscendHandle_t handle,
                                                    CausalSoftmaxAclnnDescriptor_t *desc_ptr,
                                                    infiniopTensorDescriptor_t y) {
    if (y->ndim < 2 || y->ndim >= 4) {
        return STATUS_BAD_TENSOR_SHAPE;
    }

    if (!is_contiguous(y, 0, y->ndim - 1)) {
        return STATUS_BAD_TENSOR_STRIDES;
    }

    // Construct CausalSoftmaxAclnnDescriptor
    *desc_ptr = new CausalSoftmaxAclnnDescriptor(handle->device);
    (*desc_ptr)->device_id = handle->device_id;

    // Set value from infiniopTensorDescriptor
    auto &aDesc = (*desc_ptr)->aDesc;
    auto &outDesc = (*desc_ptr)->outDesc;

    uint64_t ndim = y->ndim;
    uint64_t *shape = y->shape;
    int64_t *strides = y->strides;
    int64_t total_seq_len = static_cast<int64_t>(shape[ndim - 1]);
    int64_t seq_len = static_cast<int64_t>(shape[ndim - 2]);

    if (total_seq_len < seq_len) {
        return STATUS_BAD_TENSOR_SHAPE;
    }

    // Change input shape and stride
    auto aclnn_shape = std::vector<int64_t>(4);
    auto aclnn_strides = std::vector<int64_t>(4);
    for (uint64_t i = 0; i < ndim; ++i) {
        aclnn_shape[4 - i - 1] = shape[ndim - i - 1];
        aclnn_strides[4 - i - 1] = strides[ndim - i - 1];
    }
    // Add padding to input shape and stride if ndim < 4
    for (uint64_t i = 0; i < 4 - ndim; ++i) {
        aclnn_shape[i] = 1;
        aclnn_strides[i] = aclnn_shape[i + 1] * aclnn_strides[i + 1];
    }

    CHECK_STATUS(aDesc->setDescriptor(toAclDataType(y->dt), aclnn_shape, aclnn_strides), STATUS_SUCCESS);
    CHECK_STATUS(outDesc->setDescriptor(toAclDataType(y->dt), aclnn_shape, aclnn_strides), STATUS_SUCCESS);

    // Set mask Desc
    auto &maskDesc = (*desc_ptr)->maskDesc;
    auto mask_shape = std::vector<int64_t>(3);

    mask_shape[2] = total_seq_len;
    mask_shape[1] = seq_len;
    if (ndim == 2) {
        mask_shape[0] = 1;
    } else {
        mask_shape[0] = static_cast<int64_t>(shape[0]);
    }
    auto mask_strides = std::vector<int64_t>{total_seq_len * seq_len, total_seq_len, 1};

    CHECK_STATUS(maskDesc->setDescriptor(toAclDataType(y->dt), mask_shape, mask_strides), STATUS_SUCCESS);

    // Create aclTensor
    CHECK_STATUS(aDesc->createTensor(), STATUS_SUCCESS);
    CHECK_STATUS(maskDesc->createTensor(), STATUS_SUCCESS);
    CHECK_STATUS(outDesc->createTensor(), STATUS_SUCCESS);

    // Get Tensor
    aclTensor *ta = aDesc->t;
    aclTensor *tmask = maskDesc->t;
    aclTensor *tout = outDesc->t;

    auto &workspaceSize = (*desc_ptr)->workspaceSize;
    auto &executor = (*desc_ptr)->executor;
    auto ret = aclnnMaskedSoftmaxWithRelPosBiasGetWorkspaceSize(ta,
                                                                nullptr,
                                                                tmask,
                                                                1.0, 0,
                                                                tout,
                                                                &workspaceSize,
                                                                &executor);
    aclSetAclOpExecutorRepeatable(executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnMaskedSoftmaxWithRelPosBiasGetWorkspaceSize failed. ERROR: %d\n", ret);
              return STATUS_EXECUTION_FAILED);

    // Fill upgrade matrix
    uint16_t mask_matrix[maskDesc->shape[0]][maskDesc->shape[1]][maskDesc->shape[2]];
    auto &dims = maskDesc->shape;
    auto ele_size = aclDataTypeSize(maskDesc->dataType);

    // float neg_inf = -100000000;
    for (int i = 0; i < dims[0]; ++i) {
        for (int m = 0; m < dims[1]; ++m) {
            for (int n = 0; n < dims[2]; ++n) {
                if (n - m > dims[2] - dims[1]) {
                    // 0xF939 = -10240 half
                    mask_matrix[i][m][n] = 0xF880;
                } else {
                    mask_matrix[i][m][n] = 0;
                }
            }
        }
    }

    // malloc mask space
    auto &maskAddr = (*desc_ptr)->maskAddr;
    auto mask_size = numElements(maskDesc->shape.data(), maskDesc->ndim) * ele_size;
    CHECK_STATUS(mallocWorkspace(&maskAddr, mask_size), STATUS_SUCCESS);

    // copy mask matrix to device mem
    ret = aclrtMemcpy(maskAddr,
                      mask_size,
                      mask_matrix,
                      mask_size,
                      ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
              return STATUS_EXECUTION_FAILED);

    return STATUS_SUCCESS;
}

infiniopStatus_t aclnnGetCausalSoftmaxWorkspaceSize(CausalSoftmaxAclnnDescriptor_t desc, uint64_t *size) {

    *size = desc->workspaceSize;

    return STATUS_SUCCESS;
}

infiniopStatus_t aclnnCausalSoftmax(CausalSoftmaxAclnnDescriptor_t desc,
                                    void *workspace,
                                    uint64_t workspace_size,
                                    void *data,
                                    void *stream) {
    auto &aDesc = desc->aDesc;
    auto &maskDesc = desc->maskDesc;
    auto &outDesc = desc->outDesc;


    // Get aclTensor pt
    aclTensor *ta = aDesc->t;
    aclTensor *tmask = maskDesc->t;
    aclTensor *tout = outDesc->t;

    auto &executor = desc->executor;
    auto &workspaceSize = desc->workspaceSize;
    auto &maskAddr = desc->maskAddr;

    // Set runing on handle device
    aclrtSetDevice(desc->device_id);

    AclSetTensorAddr(executor, 0, ta, data);
    AclSetTensorAddr(executor, 2, tmask, maskAddr);
    AclSetTensorAddr(executor, 3, tout, data);

    auto ret = aclnnMaskedSoftmaxWithRelPosBias(workspace,
                                                workspaceSize,
                                                executor,
                                                stream);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnMaskedSoftmaxWithRelPosBias failed. ERROR: %d\n", ret));

    return STATUS_SUCCESS;
}

infiniopStatus_t aclnnDestroyCausalSoftmaxDescriptor(CausalSoftmaxAclnnDescriptor_t desc) {
    delete desc->aDesc;
    delete desc->maskDesc;
    delete desc->outDesc;
    aclDestroyAclOpExecutor(desc->executor);
    CHECK_STATUS(freeWorkspace(desc->maskAddr), STATUS_SUCCESS);
    delete desc;
    return STATUS_SUCCESS;
}

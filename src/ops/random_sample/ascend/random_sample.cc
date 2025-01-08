#include "random_sample.h"

RandomSampleAscendDescriptor::RandomSampleAscendDescriptor(Device _device) {
    device = _device;
    device_id = 0;
    pDesc = new aclnnTensorDescriptor();
    topkIdxDesc = new aclnnTensorDescriptor();
    topkValDesc = new aclnnTensorDescriptor();
    resDesc = new aclnnTensorDescriptor();
}

infiniopStatus_t ascendCreateRandomSampleDescriptor(AscendHandle_t handle,
                                                    RandomSampleAscendDescriptor_t *desc_ptr,
                                                    infiniopTensorDescriptor_t result,
                                                    infiniopTensorDescriptor_t probs) {
    if (probs->ndim != 1) {
        return STATUS_BAD_TENSOR_SHAPE;
    }
    if (!dtype_eq(result->dt, U64))
        return STATUS_BAD_TENSOR_DTYPE;
    if (result->ndim != 1 && result->shape[0] != 1) {
        return STATUS_BAD_TENSOR_SHAPE;
    }

    (*desc_ptr) = new RandomSampleAscendDescriptor(handle->device);
    (*desc_ptr)->device_id = handle->device_id;

    CHECK_STATUS((*desc_ptr)->pDesc->fromInfiniOpTensorDescriptor(probs), STATUS_SUCCESS);
    CHECK_STATUS((*desc_ptr)->resDesc->fromInfiniOpTensorDescriptor(result), STATUS_SUCCESS);
    // Ascend aclnnTopk doesn't support U64 type
    (*desc_ptr)->resDesc->dataType = aclDataType::ACL_INT64;

    return STATUS_SUCCESS;
}


infiniopStatus_t ascendGetRandomSampleWorkspaceSize(RandomSampleAscendDescriptor_t desc,
                                                    uint64_t *size) {
    auto &pDesc = desc->pDesc;
    *size = numElements(pDesc->shape.data(), pDesc->ndim) * aclDataTypeSize(pDesc->dataType) +
            numElements(pDesc->shape.data(), pDesc->ndim) * sizeof(I64);

    return STATUS_SUCCESS;
}

infiniopStatus_t ascendRandomSample(RandomSampleAscendDescriptor_t desc,
                                    void *workspace,
                                    uint64_t workspace_size,
                                    void *result,
                                    void const *probs,
                                    float random_val,
                                    float topp,
                                    int topk,
                                    float temperature,
                                    void *stream) {
    if (topk <= 0 || topp < 0 || topp > 1.0) {
        return STATUS_BAD_PARAM;
    }

    if (random_val < 0 || random_val > 1.0) {
        return STATUS_BAD_PARAM;
    }

    auto &pDesc = desc->pDesc;
    auto &topkIdxDesc = desc->topkIdxDesc;
    auto &topkValDesc = desc->topkValDesc;
    auto ndim = static_cast<int64_t>(pDesc->ndim);
    auto voc = pDesc->shape[0];
    auto topk_ = topk <= voc ? topk : voc;
    bool doSample = topk_ > 1 && temperature != 0 && topp != 0;

    auto topkShape = std::vector<int64_t>(pDesc->shape);
    topkShape[ndim - 1] = doSample ? topk_ : 1;

    auto topkStrides = std::vector<int64_t>(pDesc->strides);
    // Infer contiguous strides
    topkStrides[ndim - 1] = 1;
    for (int64_t i = ndim - 2; i >= 0; --i) {
        topkStrides[i] = topkStrides[i + 1] * topkShape[i + 1];
    }

    CHECK_STATUS(topkValDesc->setDescriptor(pDesc->dataType, topkShape, topkStrides), STATUS_SUCCESS);
    CHECK_STATUS(topkIdxDesc->setDescriptor(aclDataType::ACL_INT64, topkShape, topkStrides), STATUS_SUCCESS);

    // Infer data ptr
    auto workspaceTmp = workspace;
    auto topkValAddr = workspaceTmp;
    workspaceTmp = (void *) ((uint8_t *) workspace +
                             numElements(topkValDesc->shape.data(), topkValDesc->ndim) * aclDataTypeSize(topkValDesc->dataType));
    auto topkIdxAddr = workspaceTmp;
    auto pAddr = (void *) probs;

    // Create aclTensor
    CHECK_STATUS(pDesc->createTensor(pAddr), STATUS_SUCCESS);
    CHECK_STATUS(topkValDesc->createTensor(topkValAddr), STATUS_SUCCESS);
    CHECK_STATUS(topkIdxDesc->createTensor(topkIdxAddr), STATUS_SUCCESS);
    if (!doSample) {
        CHECK_STATUS(desc->resDesc->createTensor(result), STATUS_SUCCESS);
    }

    // Do Topk calculate
    uint64_t topkWorkspaceSize = 0;
    aclOpExecutor *topkExecutor = nullptr;
    auto ret = aclnnTopkGetWorkspaceSize(pDesc->t,
                                         topkShape[ndim - 1],
                                         ndim - 1,
                                         true,
                                         true,
                                         topkValDesc->t,
                                         doSample ? topkIdxDesc->t
                                                  : desc->resDesc->t,
                                         &topkWorkspaceSize,
                                         &topkExecutor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnTopkGetWorkspaceSize failed ERROR: %d\n", ret);
              return STATUS_EXECUTION_FAILED);
    void *topkWorkspace;
    CHECK_STATUS(mallocWorkspace(&topkWorkspace, topkWorkspaceSize), STATUS_SUCCESS);
    ret = aclnnTopk(topkWorkspace,
                    topkWorkspaceSize,
                    topkExecutor,
                    stream);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnTopk failed ERROR: %d\n", ret);
              return STATUS_EXECUTION_FAILED);
    CHECK_STATUS(freeWorkspace(topkWorkspace), STATUS_SUCCESS);

    if (doSample) {
        // Do softmax and topp random sample
        CHECK_STATUS(random_sample_do(
                         pAddr,
                         result,
                         topkValAddr,
                         topkIdxAddr,
                         topk,
                         static_cast<int>(pDesc->shape[0]),
                         topp,
                         temperature,
                         random_val,
                         pDesc->dataType,
                         stream),
                     STATUS_SUCCESS);
    }
    return STATUS_SUCCESS;
}

infiniopStatus_t ascendDestroyRandomSampleDescriptor(RandomSampleAscendDescriptor_t desc) {
    delete desc->pDesc;
    delete desc->topkIdxDesc;
    delete desc->topkValDesc;
    delete desc;
    return STATUS_SUCCESS;
}

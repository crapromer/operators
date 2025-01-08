#ifndef __ACLNN_REARRANGE_H__
#define __ACLNN_REARRANGE_H__

#include "../../../devices/ascend/ascend_handle.h"
#include "../../../devices/ascend/tensor_aclnn.h"
#include "operators.h"
#include <acl/acl_base.h>
#include <aclnn/acl_meta.h>
#include <aclnnop/aclnn_copy.h>

struct RearrangeAclnnDescriptor {
    Device device;
    int device_id;
    aclOpExecutor *executor;
    aclnnTensorDescriptor_t dstDesc, srcDesc;
    uint64_t workspaceSize;
    void *workspaceAddr;

    RearrangeAclnnDescriptor(Device device);
};

typedef struct RearrangeAclnnDescriptor *RearrangeAclnnDescriptor_t;

infiniopStatus_t aclnnCreateRearrangeDescriptor(AscendHandle_t handle,
                                                RearrangeAclnnDescriptor_t *desc_ptr,
                                                infiniopTensorDescriptor_t dst,
                                                infiniopTensorDescriptor_t src);

infiniopStatus_t aclnnRearrange(RearrangeAclnnDescriptor_t desc,
                                void *dst,
                                void const *src,
                                void *stream);

infiniopStatus_t aclnnDestroyRearrangeDescriptor(RearrangeAclnnDescriptor_t desc);

#endif

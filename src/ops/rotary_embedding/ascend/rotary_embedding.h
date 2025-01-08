#ifndef __ASCEND_ROTARY_EMBEDDING_H__
#define __ASCEND_ROTARY_EMBEDDING_H__

#include "../../../devices/ascend/ascend_handle.h"
#include "operators.h"

struct RoPEAscendDescriptor {
    Device device;
    int device_id;
    aclDataType dt;
    uint64_t seq_len;
    uint64_t nhead;
    uint64_t dim;
    uint64_t total_seq_len;
    int64_t stride_seq;
    int64_t stride_head;
};

typedef struct RoPEAscendDescriptor *RoPEAscendDescriptor_t;

infiniopStatus_t ascendCreateRoPEDescriptor(AscendHandle_t handle,
                                            RoPEAscendDescriptor_t *desc_ptr,
                                            infiniopTensorDescriptor_t t,
                                            infiniopTensorDescriptor_t pos_ids,
                                            infiniopTensorDescriptor_t sin_table,
                                            infiniopTensorDescriptor_t cos_table);

infiniopStatus_t ascendGetRoPEWorkspaceSize(RoPEAscendDescriptor_t desc,
                                            uint64_t *size);

infiniopStatus_t ascendRoPE(RoPEAscendDescriptor_t desc,
                            void *workspace,
                            uint64_t workspace_size,
                            void *t,
                            void const *pos_ids,
                            void const *sin_table,
                            void const *cos_table,
                            void *stream);

infiniopStatus_t ascendDestroyRoPEDescriptor(RoPEAscendDescriptor_t desc);

extern "C" infiniopStatus_t rope_kernel_do(void *t, void *pos, void *sin, void *cos,
                               int32_t nt, int32_t nh, int32_t dh, int32_t stt,
                               int32_t sth, int dtype, void *stream);

#endif

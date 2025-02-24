#ifndef __SDAA_SWIGLU_H__
#define __SDAA_SWIGLU_H__
#include "operators.h"
#include <sdaa_runtime.h>
#include "../../../devices/teco/teco_handle.h"
struct SwiGLUTecoDescriptor {
    Device device;
    int device_id;
    sdaaStream_t stream;
    uint64_t rows,cols;
    int64_t lda,ldb,ldc;
};

typedef struct SwiGLUTecoDescriptor *SwiGLUTecoDescriptor_t;


infiniopStatus_t tecoCreateSwiGLUDescriptor(TecoHandle_t handle,
                                             SwiGLUTecoDescriptor_t *desc_ptr,
                                             infiniopTensorDescriptor_t c_desc,
                                             infiniopTensorDescriptor_t a_desc,
                                             infiniopTensorDescriptor_t b_desc);

infiniopStatus_t tecoSwiGLU(SwiGLUTecoDescriptor_t desc,
                            void *c,
                            void const *a,
                            void const *b,
                            void *stream);

infiniopStatus_t tecoDestroySwiGLUDescriptor(SwiGLUTecoDescriptor_t desc);


#endif
#ifndef BATCH_NORM_H
#define BATCH_NORM_H

#include "../../export.h"
#include "../../operators.h"

__C __export void *createBatchNormDescriptor(Device, void *config);
__C __export void destroyBatchNormDescriptor(void *descriptor);
__C __export void batchNorm(void *descriptor, Tensor y, Tensor x, Tensor w, float epsilon, void *stream);

typedef struct BatchNormDescriptor {
    Device device;
} BatchNormDescriptor;

#endif

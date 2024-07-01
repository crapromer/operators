#ifndef __CPU_BATCH_NORM_H__
#define __CPU_BATCH_NORM_H__

#include "../../../operators.h"
typedef struct BatchNormCpuDescriptor {
    Device device;
} BatchNormCpuDescriptor;

void batch_norm_cpu_f16(Tensor c, Tensor beta, Tensor a, float alpha);

#endif// __CPU_BATCH_NORM_H__
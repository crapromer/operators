#ifndef __NV_GPU_BATCH_NORM_H__
#define __NV_GPU_BATCH_NORM_H__

#include "../../../operators.h"
#include <cudnn.h>
void batch_norm_nv_gpu_f16(Tensor y, Tensor x, float epsilon, void *stream);

#endif// __NV_GPU_RMS_NORM_H__
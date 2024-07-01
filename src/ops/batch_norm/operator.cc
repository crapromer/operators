#include "../utils.h"
#include "batch_norm.h"

#ifdef ENABLE_CPU
#include "cpu/batch_norm_cpu.h"
#endif
#ifdef ENABLE_NV_GPU
#include "cuda/batch_norm.cuh"
#endif

__C void *createBatchNormDescriptor(Device device, void *config) {
    return new BatchNormDescriptor{device};
}

__C void destroyBatchNormDescriptor(void *descriptor) {
    delete (BatchNormDescriptor *) descriptor;
}

__C void batchNorm(void *descriptor, Tensor y, Tensor x, Tensor w, float epsilon, void *stream) {
    auto desc = (BatchNormDescriptor *) descriptor;
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            batch_norm_cpu_f16(y, x, w, epsilon);
            break;
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu:
            batch_norm_nv_gpu_f16(y, x, w, epsilon, stream);
            break;
#endif
        default:
            PANIC(UnsupportedDevice);
    }
}

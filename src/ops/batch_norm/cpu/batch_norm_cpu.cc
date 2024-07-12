#include "batch_norm_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../utils.h"
#include <cmath>
#include <cstring>

void batch_norm_cpu_f16(Tensor y, Tensor x, float epsilon){
    int ndim = x.layout->ndim;
    int total_elements = 1;
    for (int i = 0; i < ndim; ++i) {
        total_elements *= x.layout->shape[i];
    }
    std::memcpy(y.data, x.data, total_elements * (x.layout->dt.size));
}
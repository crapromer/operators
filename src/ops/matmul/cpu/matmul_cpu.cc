#include "matmul_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../utils.h"
#include <cmath>

infiniopStatus_t cpuCreateMatmulDescriptor(CpuHandle_t handle,
                                           MatmulCpuDescriptor_t *desc_ptr,
                                           infiniopTensorDescriptor_t c_desc,
                                           float alpha,
                                           infiniopTensorDescriptor_t a_desc,
                                           infiniopTensorDescriptor_t b_desc,
                                           float beta) {
    DT dtype = c_desc->dt;

    if (dtype != F16 && dtype != F32) {
        return STATUS_BAD_TENSOR_DTYPE;
    }

    infiniopStatus_t *status = new infiniopStatus_t{STATUS_EXECUTION_FAILED};
    auto info = MatmulInfo(c_desc, a_desc, b_desc, status);
    if (*status != STATUS_SUCCESS) {
        return *status;
    }

    *desc_ptr = new MatmulCpuDescriptor{
        DevCpu,
        dtype,
        info,
        alpha,
        beta};
    return STATUS_SUCCESS;
}

infiniopStatus_t cpuGetMatmulWorkspaceSize(MatmulCpuDescriptor_t desc, uint64_t *size) {
    *size = 0;
    return STATUS_SUCCESS;
}

infiniopStatus_t cpuDestroyMatmulDescriptor(MatmulCpuDescriptor_t desc) {
    delete desc;
    return STATUS_SUCCESS;
}

template<typename Tdata>
infiniopStatus_t matmul_cpu(MatmulCpuDescriptor_t desc, void *c, float beta, void const *a, void const *b, float alpha) {
    auto info = desc->info;

    if (info.is_transed) {
        std::swap(a, b);
    }

    for (int i = 0; i < info.batch; ++i) {
        for (int m_ = 0; m_ < info.m; ++m_) {
            for (int n_ = 0; n_ < info.n; ++n_) {
                auto c_ = reinterpret_cast<Tdata *>(c) + i * info.c_matrix.stride + m_ * info.c_matrix.row_stride + n_ * info.c_matrix.col_stride;
                float sum = 0;
                for (int k_ = 0; k_ < info.k; ++k_) {
                    auto a_ = reinterpret_cast<Tdata const *>(a) + i * info.a_matrix.stride + m_ * info.a_matrix.row_stride + k_ * info.a_matrix.col_stride;
                    auto b_ = reinterpret_cast<Tdata const *>(b) + i * info.b_matrix.stride + n_ * info.b_matrix.col_stride + k_ * info.b_matrix.row_stride;
                    if constexpr (std::is_same<Tdata, uint16_t>::value) {
                        sum += f16_to_f32(*a_) * f16_to_f32(*b_);
                    } else {
                        sum += *a_ * (*b_);
                    }
                }
                if constexpr (std::is_same<Tdata, uint16_t>::value) {
                    if (beta == 0) {
                        *c_ = f32_to_f16(alpha * sum);
                    } else {
                        *c_ = f32_to_f16(beta * f16_to_f32(*c_) + alpha * sum);
                    }
                } else {
                    *c_ = beta * (*c_) + alpha * sum;
                }
            }
        }
    }
    return STATUS_SUCCESS;
}

infiniopStatus_t cpuMatmul(MatmulCpuDescriptor_t desc,
                           void *workspace,
                           uint64_t workspace_size,
                           void *c,
                           void const *a,
                           void const *b) {
    if (desc->dtype == F16) {
        return matmul_cpu<uint16_t>(desc, c, desc->beta, a, b, desc->alpha);
    }
    if (desc->dtype == F32) {
        return matmul_cpu<float>(desc, c, desc->beta, a, b, desc->alpha);
    }
    return STATUS_BAD_TENSOR_DTYPE;
}

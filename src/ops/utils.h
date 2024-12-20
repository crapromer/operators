#ifndef __UTILS_H__
#define __UTILS_H__

#include "data_type.h"
#include "tensor.h"
#include <algorithm>
#include <functional>
#include <numeric>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

/* This file contains some useful macros and helper functions */

// check if an expression is true, and if not, print an error message and abort the program
inline void assert_true(int expr, const char *msg, const char *file, int line) {
    if (!expr) {
        fprintf(stderr, "\033[31mAssertion failed:\033[0m %s at file %s, line %d\n", msg, file, line);
        exit(EXIT_FAILURE);
    }
}

#define ASSERT(expr) assert_true(expr, #expr " is false", __FILE__, __LINE__)
#define ASSERT_EQ(a, b) assert_true((a) == (b), #a " != " #b, __FILE__, __LINE__)
#define ASSERT_VALID_PTR(a) assert_true((a) != nullptr, #a " is nullptr", __FILE__, __LINE__)

#define PANIC(EXPR)                                             \
    printf("Error at %s:%d - %s\n", __FILE__, __LINE__, #EXPR); \
    exit(EXIT_FAILURE)

#define ROUND_UP_DIV(x, y) ((x + y - 1) / y)

#define CHECK_ERROR(call, target, errCode)            \
    do {                                              \
        if (auto value = (call); value == (target)) { \
            return (errCode);                         \
        }                                             \
    } while (0)
#define CREATE_CHECK_ERROR(expr, value, target, errCode) \
    expr;                                                \
    CHECK_ERROR(value, target, errCode)

#define CHECK_STATUS(call, target)                    \
    do {                                              \
        if (auto value = (call); value != (target)) { \
            return value;                             \
        }                                             \
    } while (0)

// check if two data layouts (types) are equal
inline bool dtype_eq(DataLayout a, DataLayout b) {
    union TypePun {
        DataLayout layout;
        int i;
    } pun;
    pun.layout = a;
    auto a_ = pun.i;
    pun.layout = b;
    auto b_ = pun.i;
    return a_ == b_;
}

inline std::vector<int64_t> get_byte_strides(infiniopTensorDescriptor_t desc) {
    int64_t dsize = desc->dt.size;
    std::vector<int64_t> strides(desc->ndim);
    for (uint64_t i = 0; i < desc->ndim; i++) {
        strides[i] = dsize * desc->strides[i];
    }

    return strides;
}

// calculate the broadcasted shape for two tensors
inline bool getBroadcastShape(const uint64_t *shape1, uint64_t ndim1,
                              const uint64_t *shape2, uint64_t ndim2,
                              uint64_t *broadcast_shape, uint64_t *padded_shape1,
                              uint64_t *padded_shape2, uint64_t max_rank) {
    // prepending and initializing
    std::fill(padded_shape1, padded_shape1 + max_rank, 1);
    std::fill(padded_shape2, padded_shape2 + max_rank, 1);
    std::copy(shape1, shape1 + ndim1, padded_shape1 + max_rank - ndim1);
    std::copy(shape2, shape2 + ndim2, padded_shape2 + max_rank - ndim2);

    // compute broadcasted shape
    for (size_t i = 0; i < max_rank; ++i) {
        if (padded_shape1[i] == padded_shape2[i] || padded_shape1[i] == 1 || padded_shape2[i] == 1) {
            broadcast_shape[i] = std::max(padded_shape1[i], padded_shape2[i]);
        } else {
            return false;
        }
    }

    return true;
}

// check if the shape of tensor c is valid after broadcasting tensors a and b and also get the broadcasted shapes
inline bool isValidBroadcastShape(infiniopTensorDescriptor_t a, infiniopTensorDescriptor_t b, infiniopTensorDescriptor_t c,
                                  uint64_t *broadcast_shape, uint64_t *padded_shape1, uint64_t *padded_shape2, uint64_t broadcast_ndim) {
    if (broadcast_ndim != c->ndim || !getBroadcastShape(a->shape, a->ndim, b->shape, b->ndim, broadcast_shape, padded_shape1, padded_shape2, broadcast_ndim)) {
        return false;
    }
    return std::equal(broadcast_shape, broadcast_shape + broadcast_ndim, c->shape);
}

// check if the shape of tensor c is valid after broadcasting tensors a and b
inline bool isValidBroadcastShape(infiniopTensorDescriptor_t a, infiniopTensorDescriptor_t b, infiniopTensorDescriptor_t c) {
    uint64_t broadcast_ndim = std::max(a->ndim, b->ndim);
    uint64_t broadcast_shape[broadcast_ndim];
    uint64_t padded_shape1[broadcast_ndim];
    uint64_t padded_shape2[broadcast_ndim];
    return isValidBroadcastShape(a, b, c, broadcast_shape, padded_shape1, padded_shape2, broadcast_ndim);
}

inline uint64_t get_byte_size(infiniopTensorDescriptor_t desc) {
    uint64_t dsize = desc->dt.size;
    uint64_t size = 1;
    for (uint64_t i = 0; i < desc->ndim; i++) {
        size *= desc->shape[i];
    }
    return size * dsize;
}

// permute the dimensions of a tensor descriptor
inline infiniopTensorDescriptor_t permute(infiniopTensorDescriptor_t desc, const std::vector<uint64_t> &order) {
    uint64_t ndim = desc->ndim;
    if (order.size() != ndim) {
        return nullptr;
    }
    uint64_t *shape = new uint64_t[ndim];
    int64_t *strides = new int64_t[ndim];
    for (uint64_t i = 0; i < ndim; i++) {
        if (std::find(order.begin(), order.end(), i) == order.end()) {
            return nullptr;
        }
        shape[i] = desc->shape[order[i]];
        strides[i] = desc->strides[order[i]];
    }
    return new TensorDescriptor{
        desc->dt, ndim, shape, strides};
}

// check if the dimensions [dim_start, dim_end] of a tensor descriptor are contiguous
inline bool is_contiguous(const infiniopTensorDescriptor_t &desc, uint64_t dim_start, uint64_t dim_end) {
    for (size_t i = dim_start + 1; i <= dim_end; i++) {
        if (desc->strides[i - 1] != static_cast<int64_t>(desc->shape[i]) * desc->strides[i]) {
            return false;
        }
    }
    return true;
}

inline bool is_contiguous(const infiniopTensorDescriptor_t &desc) {
    if (desc->ndim == 0) {
        return true;
    }
    return is_contiguous(desc, 0, desc->ndim - 1);
}

// merge the dimensions [dim_start, dim_end] of a tensor descriptor
inline infiniopTensorDescriptor_t dim_merge(infiniopTensorDescriptor_t desc, uint64_t dim_start, uint64_t dim_end) {
    uint64_t ndim = desc->ndim;
    if (dim_start > dim_end || dim_end >= ndim) {
        return nullptr;
    }

    uint64_t new_ndim = ndim - (dim_end - dim_start);
    uint64_t *new_shape = new uint64_t[new_ndim];
    int64_t *new_strides = new int64_t[new_ndim];
    uint64_t index = 0;
    for (size_t i = 0; i < dim_start; i++) {
        new_shape[index] = desc->shape[i];
        new_strides[index] = desc->strides[i];
        index++;
    }
    if (!is_contiguous(desc, dim_start, dim_end)) {
        return nullptr;
    }
    new_shape[index] = 1;
    for (size_t i = dim_start; i <= dim_end; i++) {
        new_shape[index] *= desc->shape[i];
    }
    new_strides[index] = desc->strides[dim_end];
    index++;
    for (size_t i = dim_end + 1; i < ndim; i++) {
        new_shape[index] = desc->shape[i];
        new_strides[index] = desc->strides[i];
        index++;
    }
    return new TensorDescriptor{
        desc->dt, new_ndim, new_shape, new_strides};
}

// split the dimension dim of a tensor descriptor into multiple dimensions
inline infiniopTensorDescriptor_t dim_split(infiniopTensorDescriptor_t desc, uint64_t dim, const std::vector<uint64_t> &dims) {
    uint64_t ndim = desc->ndim;
    if (static_cast<int64_t>(desc->shape[dim]) != std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<uint64_t>())) {
        return nullptr;
    }
    uint64_t new_ndim = ndim + dims.size() - 1;
    uint64_t *new_shape = new uint64_t[new_ndim];
    int64_t *new_strides = new int64_t[new_ndim];
    uint64_t index = 0;
    for (size_t i = 0; i < dim; i++) {
        new_shape[index] = desc->shape[i];
        new_strides[index] = desc->strides[i];
        index++;
    }
    for (size_t i = 0; i < dims.size(); i++) {
        new_shape[index] = dims[i];
        new_strides[index] = desc->strides[dim] * desc->shape[dim] / std::accumulate(dims.begin(), dims.begin() + i + 1, 1, std::multiplies<uint64_t>());
        index++;
    }
    for (size_t i = dim + 1; i < ndim; i++) {
        new_shape[index] = desc->shape[i];
        new_strides[index] = desc->strides[i];
        index++;
    }
    return new TensorDescriptor{
        desc->dt, new_ndim, new_shape, new_strides};
}

#endif// __UTILS_H__

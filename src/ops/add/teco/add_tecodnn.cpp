#include "add_tecodnn.h"

infiniopStatus_t tecoCreateAddDescriptor(TecoHandle_t handle, AddTecoDescriptor_t *desc_ptr, infiniopTensorDescriptor_t c_desc infiniopTensorDescriptor_t a_desc, infiniopTensorDescriptor_t b_desc) {
    return STATUS_SUCCESS;
}

template<typename Tdata>
infiniopStatus_t add_teco(AddCpuDescriptor_t desc, void *c, void const *a, void const *b){
    return STATUS_SUCCESS;
}

infiniopStatus_t tecoAdd(AddTecoDescriptor_t desc, void *c, const void *a, const void *b, void *stream) {
    return STATUS_SUCCESS;
}

infiniopStatus_t tecoDestroyAddDescriptor(AddTecoDescriptor_t desc) {
    return STATUS_SUCCESS;
}

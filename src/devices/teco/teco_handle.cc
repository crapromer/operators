#include "teco_handle.h"

infiniopStatus_t createTecoHandle(TecoHandle_t *handle_ptr, int device_id) {
    uint32_t device_count;
    sdaaGetDeviceCount(reinterpret_cast<int*>(&device_count));
    if (device_id >= static_cast<int>(device_count)) {
        return STATUS_BAD_DEVICE;
    }

    sdaaSetDevice(device_id);

    *handle_ptr = new TecoContext{DevTecoSDAA, device_id};
    tecoblasCreate(&(*handle_ptr)->handle);
    sdaaStreamCreate(&(*handle_ptr)->stream);
    tecoblasSetStream((*handle_ptr)->handle,(*handle_ptr)->stream);

    return STATUS_SUCCESS;
}

infiniopStatus_t deleteTecoHandle(TecoHandle_t handle_ptr) {
    sdaaStreamDestroy(handle_ptr->stream);
    tecoblasDestroy(handle_ptr->handle);
    delete handle_ptr;
    return STATUS_SUCCESS;
}

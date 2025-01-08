#include "teco_handle.h"

infiniopStatus_t createTecoHandle(TecoHandle_t *handle_ptr, int device_id) {
    uint32_t device_count;
    sdaaGetDeviceCount(reinterpret_cast<int*>(&device_count));
    if (device_id >= static_cast<int>(device_count)) {
        return STATUS_BAD_DEVICE;
    }

    sdaaSetDevice(device_id);
    sdaaStream_t stream;
    sdaaStreamCreate(&stream);
    *handle_ptr = new TecoContext{DevTecoSDAA, device_id,stream};
    
    return STATUS_SUCCESS;
}

infiniopStatus_t deleteTecoHandle(TecoHandle_t handle_ptr) {
    sdaaStreamDestroy(handle_ptr->stream);
    delete handle_ptr;
    return STATUS_SUCCESS;
}

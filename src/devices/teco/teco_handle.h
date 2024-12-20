#ifndef __TECO_HANDLE__
#define __TECO_HANDLE__
#include "common_teco.h"
#include "status.h"
#include "../pool.h"
struct TecoContext {
    Device device;
    int device_id;
    tecoblasHandle_t handle;
    sdaaStream_t stream;
};
typedef struct TecoContext *TecoHandle_t;

infiniopStatus_t createTecoHandle(TecoHandle_t *handle_ptr, int device_id);

infiniopStatus_t deleteTecoHandle(TecoHandle_t handle_ptr);

#endif

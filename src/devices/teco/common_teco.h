#ifndef _COMMON_TECO_
#define _COMMON_TECO_

#include <stdio.h>
#include <stdlib.h>
#include <sdaa_runtime.h>
#include <tecoblas.h>
#include "device.h"
#define CHECK_TECOBLAS(expression)                                                               \
    {                                                                                            \
        tecoblasStatus_t status = (expression);                                                  \
        if (status != TECOBLAS_STATUS_SUCCESS) {                                                 \
            fprintf(stderr, "Error at line %d: %s\n", __LINE__, tecoblasGetErrorString(status)); \
            exit(EXIT_FAILURE);                                                                  \
        }                                                                                        \
    }


#endif
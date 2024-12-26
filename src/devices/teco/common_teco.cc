#include "common_teco.h"
void** convertToBatch(void* data, int batch, int m, int n, size_t typeSize){
    // Dynamically allocate memory for the output array of pointers
    void** output = new void*[batch];

    // Treat the void* data as a pointer to raw memory and use pointer arithmetic
    for (int i = 0; i < batch; i++) {
        // Output[i] will point to the i-th 2D slice (this is done in raw pointer arithmetic)
        output[i] = static_cast<void*>(static_cast<char*>(data) + i * m * n * typeSize);
    }

    // Return the output array of pointers
    return output;
}
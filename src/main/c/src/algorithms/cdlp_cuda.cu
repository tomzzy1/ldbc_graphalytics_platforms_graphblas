#include <iostream>
#include "cdlp_cuda.cuh"
#include "common/fmt.hpp"
#include "common/utils.hpp"

#define PRINT(...) LOG(info, std::string(fmt::format(__VA_ARGS__)))


namespace CUDA_CDLP {

int test_cuda_device_query(){

    PRINT("running test_cuda_device_query");

    int deviceCount;

    cudaGetDeviceCount(&deviceCount);

    timer_start("[CUDA][TIMER] Getting GPU Data."); //@@ start a timer

    for (int dev = 0; dev < deviceCount; dev++) {
        cudaDeviceProp deviceProp;

        cudaGetDeviceProperties(&deviceProp, dev);

        if (dev == 0) {
            if (deviceProp.major == 9999 && deviceProp.minor == 9999) {
                PRINT("No CUDA GPU has been detected");
                return -1;
            } else if (deviceCount == 1) {
                //@@ WbLog is a provided logging API (similar to Log4J).
                //@@ The logging function wbLog takes a level which is either
                //@@ OFF, FATAL, ERROR, WARN, INFO, DEBUG, or trace and a
                //@@ message to be printed.
                PRINT("There is 1 device supporting CUDA");
            } else {
                PRINT("There are {} devices supporting CUDA", deviceCount);
            }
        }

        PRINT("Device {} name {}", dev, deviceProp.name);
        PRINT("\tComputational Capabilities: {}.{}", deviceProp.major, deviceProp.minor);
        PRINT("\tMaximum global memory size: {}", deviceProp.totalGlobalMem);
        PRINT("\tMaximum constant memory size: {}", deviceProp.totalConstMem);
        PRINT("\tMaximum shared memory size per block: {}", deviceProp.sharedMemPerBlock);
        PRINT("\tMaximum block dimensions: {}x{}x{}", deviceProp.maxThreadsDim[0],
                                                    deviceProp.maxThreadsDim[1],
                                                    deviceProp.maxThreadsDim[2]);
        PRINT("\tMaximum grid dimensions: {}x{}x{}", deviceProp.maxGridSize[0],
                                                   deviceProp.maxGridSize[1],
                                                   deviceProp.maxGridSize[2]);
        PRINT("\tWarp size: {}", deviceProp.warpSize);
    }

    timer_stop(); //@@ stop the timer

    return 0;
}

int LAGraph_cdlp_gpu(
    GrB_Vector *CDLP_handle, // output vector
    const GrB_Matrix A,      // input matrix
    bool symmetric,          // denote whether the matrix is symmetric
    bool sanitize,           // if true, ensure A is binary
    int itermax,             // max number of iterations,
    double *t                // t [0] = sanitize time, t [1] = cdlp time, in seconds
){
    // check input
    if (CDLP_handle == NULL || t == NULL)
    {
        return GrB_NULL_POINTER;
    }
    // set timing to zero
    t [0] = 0;         // sanitize time
    t [1] = 0;         // CDLP time

    // check input matrix
    GrB_Index n, nz, nnz;
    GrB_Matrix_nrows(&n, A);
    GrB_Matrix_nvals(&nz, A);
    if (!symmetric){
        nnz = 2 * nz;
    }else{
        nnz = nz;
    }
    PRINT("nnz value is {}", nnz);
    
    GrB_Matrix AT = NULL;
    GrB_Matrix_new (&AT, GrB_UINT64, n, n);
    GrB_transpose (AT, NULL, NULL, A, NULL);

    return GrB_SUCCESS;
}


}
#pragma once

#include <cuda.h>

#include "graphio.h"

namespace CUDA_CDLP {

int test_cuda_device_query();

int LAGraph_cdlp_gpu(
    GrB_Vector *CDLP_handle, // output vector
    GrB_Matrix A,      // input matrix
    bool symmetric,          // denote whether the matrix is symmetric
    bool sanitize,           // if true, ensure A is binary
    int itermax,             // max number of iterations,
    double *t                // t [0] = sanitize time, t [1] = cdlp time, in seconds
);

}
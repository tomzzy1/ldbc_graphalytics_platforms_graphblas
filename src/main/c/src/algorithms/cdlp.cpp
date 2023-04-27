/*
 * CDLP algorithm implementation in GraphBLAS.
 */

#include <iostream>
#include <ctime>
#include <fstream>
#include <vector>

#include "utils.h"
#include "graphio.h"
#include "computation_timer.hpp"

#include "cdlp_cuda.cuh"

#define USE_GPU_CDLP 1

/*
 * Result serializer function
 */
void SerializeCDLPResult(
    GrB_Vector result,
    const std::vector<GrB_Index> &mapping,
    const BenchmarkParameters &parameters)
{

    std::ofstream file{parameters.output_file};
    if (!file.is_open())
    {
        std::cerr << "Output file " << parameters.output_file << " does not exists" << std::endl;
        exit(-1);
    }
    file.precision(16);
    file << std::scientific;

    GrB_Info info;
    GrB_Index n = mapping.size();
    GrB_Index nvals;
    OK(GrB_Vector_nvals(&nvals, result))

    uint64_t *X = NULL;
    X = (uint64_t *)malloc(n * sizeof(uint64_t));
    OK(GrB_Vector_extractTuples_UINT64(GrB_NULL, X, &nvals, result));

    for (GrB_Index matrix_index = 0; matrix_index < n; matrix_index++)
    {
        GrB_Index original_index = mapping[matrix_index];
        file << original_index << " " << mapping[X[matrix_index]] << std::endl;
    }

    free(X);
}

GrB_Vector LA_CDLP_CPU(GrB_Matrix A, bool symmetric, int itermax)
{
    GrB_Info info;
    GrB_Vector l;

    ComputationTimer timer{"CDLP"};
    double timing[2];
    char msg[LAGRAPH_MSG_LEN];
    std::cout << "Processing starts at: " << GetCurrentMilliseconds() << std::endl;
    LAGraph_cdlp(&l, A, symmetric, false, itermax, timing, msg);
    std::cout << "Processing ends at: " << GetCurrentMilliseconds() << std::endl;

    return l;
}

GrB_Vector MY_CDLP_GPU(GrB_Matrix A, bool symmetric, int itermax)
{
    GrB_Info info;
    GrB_Vector l;

    ComputationTimer timer{"CDLP"};
    double timing[2];
    char msg[LAGRAPH_MSG_LEN];

    CUDA_CDLP::LAGraph_cdlp_gpu(&l, A, symmetric, false, itermax, timing);

    return l;
}

int main(int argc, char **argv)
{

    // test basic cuda device query
    // CUDA_CDLP::test_cuda_device_query();

    BenchmarkParameters parameters = ParseBenchmarkParameters(argc, argv);

    LAGraph_Init(NULL);
    GxB_Global_Option_set(GxB_GLOBAL_NTHREADS, parameters.thread_num);

    GrB_Matrix A = ReadMatrixMarket(parameters);
    std::vector<GrB_Index> mapping = ReadMapping(parameters);

    // std::cout << "Processing starts at: " << GetCurrentMilliseconds() << std::endl;
#if USE_GPU_CDLP != 0
    GrB_Vector result = MY_CDLP_GPU(A, !parameters.directed, parameters.max_iteration);
#else 
    GrB_Vector result = LA_CDLP_CPU(A, !parameters.directed, parameters.max_iteration);
#endif

    SerializeCDLPResult(result, mapping, parameters);

    GrB_Matrix_free(&A);
    GrB_Vector_free(&result);
}

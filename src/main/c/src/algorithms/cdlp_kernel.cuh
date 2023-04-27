#ifndef CDLP_KERNEL_CUH
#define CDLP_KERNEL_CUH

#include <cuda.h>
#include "graphio.h"
#include "common/fmt.hpp"
#include "common/utils.hpp"

#define PRINT(...) LOG(info, std::string(fmt::format(__VA_ARGS__)))

#define DEBUG_PRINT 1


__global__ void initialize_label(GrB_Index *labels, GrB_Index N);

__global__ void check_equality(GrB_Index *labels, GrB_Index *new_labels, GrB_Index N, int *is_equal_k);

__global__ void cdlp_base(GrB_Index *Ap, GrB_Index Ap_size, GrB_Index *Aj, uint64_t *labels, GrB_Index N, bool symmetric, GrB_Index *bin_count_k, GrB_Index *bin_label_k);

__global__ void cdlp_optimized1(GrB_Index *Ap, GrB_Index Ap_size, GrB_Index *Aj, uint64_t *labels, GrB_Index N, bool symmetric, GrB_Index *bin_count_k, GrB_Index *bin_label_k, int *bin_index);

__host__ void cdlp_gpu(GrB_Index *Ap, GrB_Index Ap_size, GrB_Index *Aj, GrB_Index Aj_size, GrB_Vector *CDLP_handle, GrB_Index N, GrB_Index nnz, bool symmetric, int itermax);

#endif
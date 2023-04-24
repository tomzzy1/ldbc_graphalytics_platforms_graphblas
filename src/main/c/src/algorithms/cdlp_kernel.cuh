#ifndef CDLP_KERNEL_CUH
#define CDLP_KERNEL_CUH

#include <cuda.h>
#include "graphio.h"

__global__ void cdlp_base(GrB_Index *Ap, GrB_Index Ap_size, GrB_Index *Aj, uint64_t *labels, GrB_Index N, bool symmetric, int itermax, GrB_Index *bin_count_k, GrB_Index  *bin_label_k);

__host__ void cdlp_gpu(GrB_Index *Ap, GrB_Index Ap_size, GrB_Index *Aj, GrB_Index Aj_size, GrB_Vector *CDLP_handle, GrB_Index N, bool symmetric, int itermax);

#endif
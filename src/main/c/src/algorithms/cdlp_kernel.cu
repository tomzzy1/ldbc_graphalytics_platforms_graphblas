#include "cdlp_kernel.cuh"

#define GRID_DIM    1
#define BLOCK_DIM   1024

__global__ void cdlp_base(
    GrB_Index *Ap,      // Row pointers
    GrB_Index Ap_size, 
    GrB_Index *Aj,      // Column indices
    GrB_Index *labels,  // Labels for each node
    GrB_Index N,        // Number of nodes
    bool symmetric,     // Is the matrix symmetric (aka is the graph undirected)
    int itermax,        // Numher of iterations
    GrB_Index *bin_count, GrB_Index *bin_label
) {
    GrB_Index ti = blockDim.x * blockIdx.x + threadIdx.x;

    // Initalize each node with a label
    for (GrB_Index i = ti; i < N; i += BLOCK_DIM) {
        if (i < N)
            labels[i] = i;
    }
    __syncthreads();

    // Iterate until converge or reaching maximum number
    for (int n = 0; n < itermax; n++) {
        // Loop through all nodes
        for (GrB_Index srcNode = ti; srcNode < N; srcNode += BLOCK_DIM) {
            if (srcNode < N) {
                // 1. Count neighbors' labels
                GrB_Index j_base = Ap[srcNode];
                GrB_Index j_max = Ap[srcNode+1];
                for (GrB_Index j = j_base; j < j_max; j++) {
                    GrB_Index desNode = Aj[j];
                    GrB_Index label = labels[desNode];      // Label of destination node
                    
                    // 1.1 If is a directed graph
                    GrB_Index incr = 1;
                    // if (!symmetric) {
                    //     // Find whether the arc is dual
                    //     for (GrB_Index i = Ap[desNode]; i < Ap[desNode+1]; i++) {
                    //         if (Aj[i] == srcNode) {
                    //             incr = 2;
                    //             break;
                    //         }
                    //     }
                    // }

                    // 1.2 Initalize bin & count label
                    bool isNew = true;
                    // Whether the label is presented in bin
                    for (GrB_Index b = j_base; b < j; b++) {
                        if (bin_label[b] == label) {
                            bin_count[b] += incr;
                            isNew = false;
                            break;
                        }
                    }
                    if (isNew) {
                        bin_label[j] = label;
                        bin_count[j] = incr;
                    } else {
                        bin_label[j] = (GrB_Index) -1;
                        bin_count[j] = (GrB_Index) 0;
                    }
                }

                // 2. Find label with maximum frequence
                GrB_Index max_count = (GrB_Index) 0;
                GrB_Index min_label = (GrB_Index) -1;
                for (GrB_Index j = j_base; j < j_max; j++) {
                    if (max_count < bin_count[j]) {
                        max_count = bin_count[j];
                        min_label = bin_label[j];
                    } else if (max_count == bin_count[j] && min_label > bin_label[j] && bin_label[j] != (GrB_Index) -1) {
                        min_label = bin_label[j];
                    } else {}
                }

                // 3. Update label
                if (min_label != (GrB_Index) -1) {
                    labels[srcNode] = min_label;    // TODO: potential overflow
                }
            }
            __syncthreads();
        }
    }
}


__host__ void cdlp_gpu(GrB_Index *Ap, GrB_Index Ap_size, GrB_Index *Aj, GrB_Index Aj_size, GrB_Vector *CDLP_handle, GrB_Index N, bool symmetric, int itermax) {
    GrB_Index *Ap_k;
    GrB_Index *Aj_k;
    GrB_Index *labels_k;
    GrB_Index *labels;
    GrB_Index *bin_count_k, *bin_label_k;   // For dynamically counting labels (can be optimized using shared memory plus a overflow global memory)

    cudaMalloc((void **) &Ap_k, Ap_size*sizeof(GrB_Index));
    cudaMalloc((void **) &Aj_k, Aj_size*sizeof(GrB_Index));
    cudaMalloc((void **) &labels_k, N*sizeof(GrB_Index));
    cudaMallocHost((void **) &labels, N*sizeof(GrB_Index));
    cudaMalloc((void **) &bin_count_k, Aj_size*sizeof(GrB_Index));
    cudaMalloc((void **) &bin_label_k, Aj_size*sizeof(GrB_Index));

    cudaMemcpy(Ap_k, Ap, Ap_size*sizeof(GrB_Index), cudaMemcpyHostToDevice);
    cudaMemcpy(Aj_k, Aj, Aj_size*sizeof(GrB_Index), cudaMemcpyHostToDevice);
    
    dim3 DimGrid(GRID_DIM, 1, 1);
    dim3 DimBlock(BLOCK_DIM, 1);
    cdlp_base<<<DimGrid, DimBlock>>>(Ap_k, Ap_size, Aj_k, labels_k, N, symmetric, itermax, bin_count_k, bin_label_k);
    cudaDeviceSynchronize();

    cudaMemcpy(labels, labels_k, N*sizeof(GrB_Index), cudaMemcpyDeviceToHost);

    cudaFree(Ap_k);
    cudaFree(Aj_k);
    cudaFree(labels_k);

    GrB_Vector CDLP = NULL;
    GrB_Vector_new(&CDLP, GrB_UINT64, N);
    for (GrB_Index i = 0; i < N; i++) {
        GrB_Vector_setElement_UINT64(CDLP, labels[i], i);
    }
    (*CDLP_handle) = CDLP;

    cudaFree(labels);
}
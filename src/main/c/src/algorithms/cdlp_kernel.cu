#include "cdlp_kernel.cuh"
#include <iostream>

#define PRINT(...) LOG(info, std::string(fmt::format(__VA_ARGS__)))

constexpr int GRID_DIM = 1;
constexpr int BLOCK_DIM = 1024;

__host__ __device__ static inline int ceil_div(int x, int y)
{
    return (x - 1) / y + 1;
}

__global__ void initialize_label(GrB_Index *labels, GrB_Index N)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i = 0; i < ceil_div(N, gridDim.x * blockDim.x); ++i)
    {
        int idx = i * gridDim.x * blockDim.x + x;
        if (idx < N)
        {
            labels[idx] = idx;
        }
    }
}

__global__ void check_equality(GrB_Index *labels, GrB_Index *new_labels, GrB_Index N, int *is_equal_k)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i = 0; i < ceil_div(N, gridDim.x * blockDim.x); ++i)
    {
        int idx = i * gridDim.x * blockDim.x + x;
        if (idx < N && *is_equal_k)
        {
            if (labels[idx] != new_labels[idx])
            {
                atomicAnd(is_equal_k, 0);
            }
        }
    }
    __syncthreads();
}

__global__ void cdlp_base(
    GrB_Index *Ap, // Row pointers
    GrB_Index Ap_size,
    GrB_Index *Aj,         // Column indices
    GrB_Index *labels,     // Labels for each node
    GrB_Index *new_labels, // new labels after each iteration
    GrB_Index N,           // Number of nodes
    bool symmetric,        // Is the matrix symmetric (aka is the graph undirected)
    GrB_Index *bin_count, GrB_Index *bin_label)
{
    GrB_Index ti = blockDim.x * blockIdx.x + threadIdx.x;
    // Iterate until converge or reaching maximum number
    // Loop through all nodes
    for (GrB_Index srcNode = ti; srcNode < N; srcNode += BLOCK_DIM)
    {
        if (srcNode < N)
        {
            // 1. Count neighbors' labels
            GrB_Index j_base = Ap[srcNode];
            GrB_Index j_max = Ap[srcNode + 1];
            for (GrB_Index j = j_base; j < j_max; j++)
            {
                GrB_Index desNode = Aj[j];
                GrB_Index label = labels[desNode]; // Label of destination node

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
                for (GrB_Index b = j_base; b < j; b++)
                {
                    if (bin_label[b] == label)
                    {
                        bin_count[b] += incr;
                        isNew = false;
                        break;
                    }
                }
                if (isNew)
                {
                    bin_label[j] = label;
                    bin_count[j] = incr;
                }
                else
                {
                    bin_label[j] = (GrB_Index)-1;
                    bin_count[j] = (GrB_Index)0;
                }
            }

            // 2. Find label with maximum frequence
            GrB_Index max_count = (GrB_Index)0;
            GrB_Index min_label = (GrB_Index)-1;
            for (GrB_Index j = j_base; j < j_max; j++)
            {
                if (max_count < bin_count[j])
                {
                    max_count = bin_count[j];
                    min_label = bin_label[j];
                }
                else if (max_count == bin_count[j] && min_label > bin_label[j] && bin_label[j] != (GrB_Index)-1)
                {
                    min_label = bin_label[j];
                }
                else
                {
                }
            }

            // 3. Update label
            if (min_label != (GrB_Index)-1)
            {
                // labels[srcNode] = min_label; // TODO: potential overflow
                new_labels[srcNode] = min_label;
            }
        }
        __syncthreads();
    }
}

__host__ void cdlp_gpu(GrB_Index *Ap, GrB_Index Ap_size, GrB_Index *Aj, GrB_Index Aj_size, GrB_Vector *CDLP_handle, GrB_Index N, bool symmetric, int itermax)
{
    GrB_Index *Ap_k;
    GrB_Index *Aj_k;
    GrB_Index *labels_k;
    GrB_Index *new_labels_k;
    GrB_Index *labels;
    GrB_Index *bin_count_k, *bin_label_k; // For dynamically counting labels (can be optimized using shared memory plus a overflow global memory)
    int is_equal = 1;
    int *is_equal_k;

    cudaMalloc((void **)&Ap_k, Ap_size * sizeof(GrB_Index));
    cudaMalloc((void **)&Aj_k, Aj_size * sizeof(GrB_Index));
    cudaMalloc((void **)&labels_k, N * sizeof(GrB_Index));
    cudaMalloc((void **)&new_labels_k, N * sizeof(GrB_Index));
    cudaMallocHost((void **)&labels, N * sizeof(GrB_Index));
    cudaMalloc((void **)&bin_count_k, Aj_size * sizeof(GrB_Index));
    cudaMalloc((void **)&bin_label_k, Aj_size * sizeof(GrB_Index));
    cudaMalloc((void **)&is_equal_k, sizeof(int));

    cudaMemcpy(Ap_k, Ap, Ap_size * sizeof(GrB_Index), cudaMemcpyHostToDevice);
    cudaMemcpy(Aj_k, Aj, Aj_size * sizeof(GrB_Index), cudaMemcpyHostToDevice);

    dim3 DimGrid(GRID_DIM, 1, 1);
    dim3 DimBlock(BLOCK_DIM, 1);

    initialize_label<<<DimGrid, DimBlock>>>(labels_k, N);

    for (int i = 0; i < itermax; ++i)
    {
        cdlp_base<<<DimGrid, DimBlock>>>(Ap_k, Ap_size, Aj_k, labels_k, new_labels_k, N, symmetric, bin_count_k, bin_label_k);
        //cudaDeviceSynchronize();
        cudaMemset(is_equal_k, 1, sizeof(int));
        check_equality<<<DimGrid, DimBlock>>>(labels_k, new_labels_k, N, is_equal_k);
        //cudaDeviceSynchronize();
        cudaMemcpy(&is_equal, is_equal_k, sizeof(int), cudaMemcpyDeviceToHost);
        if (is_equal)
            break;
        else
        {
            cudaMemcpy(labels_k, new_labels_k, N * sizeof(GrB_Index), cudaMemcpyDeviceToDevice);
        }
    }

    cudaDeviceSynchronize();

    cudaMemcpy(labels, labels_k, N * sizeof(GrB_Index), cudaMemcpyDeviceToHost);

    cudaFree(Ap_k);
    cudaFree(Aj_k);
    cudaFree(labels_k);
    cudaFree(new_labels_k);

    GrB_Vector CDLP = NULL;
    GrB_Vector_new(&CDLP, GrB_UINT64, N);
    for (GrB_Index i = 0; i < N; i++)
    {
        GrB_Vector_setElement_UINT64(CDLP, labels[i], i);
    }
    (*CDLP_handle) = CDLP;

    cudaFree(labels);
}
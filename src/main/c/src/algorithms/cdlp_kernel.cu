#include "cdlp_kernel.cuh"
#include <iostream>

#define optimized1 0
#define optimized_hash 0
#define optimized_hash1 1

constexpr int GRID_DIM = 256;
constexpr int BLOCK_DIM = 1024;
constexpr int LOCAL_BIN_SIZE = 16;
constexpr int PAR_THRESHOLD = 256;
constexpr int BLOCK_DIM2 = 256;

__host__ __device__ static inline int ceil_div(int x, int y)
{
    return (x - 1) / y + 1;
}

// initialize labels for CDLP
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

// check if two arrays are equal
// possible optimization: avoid atomicAdd
// think of this problem as a reduction problem, then optimize the memeory access
// TODO: use reduction optimizations from ECE408
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
    GrB_Index *Ap,         // Row pointers
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
    GrB_Index stride = gridDim.x * blockDim.x;
    for (GrB_Index srcNode = ti; srcNode < N; srcNode += stride)
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
        // __syncthreads();
    }
}

__global__ void cdlp_optimized1(
    GrB_Index *Ap,         // Row pointers
    GrB_Index *Aj,         // Column indices
    GrB_Index *labels,     // Labels for each node
    GrB_Index *new_labels, // new labels after each iteration
    GrB_Index N,           // Number of nodes
    bool symmetric,        // Is the matrix symmetric (aka is the graph undirected)
    GrB_Index *bin_count, GrB_Index *bin_label, int *bin_index)
{
    GrB_Index ti = blockDim.x * blockIdx.x + threadIdx.x;
    // Iterate until converge or reaching maximum number
    // Loop through all nodes
    GrB_Index stride = gridDim.x * blockDim.x;
    for (GrB_Index srcNode = ti; srcNode < N; srcNode += stride)
    {
        if (srcNode < N)
        {
            GrB_Index local_bin_count[LOCAL_BIN_SIZE];
            GrB_Index local_bin_label[LOCAL_BIN_SIZE];
            // 1. Count neighbors' labels
            GrB_Index j_base = Ap[srcNode];
            GrB_Index j_max = Ap[srcNode + 1];
            auto neighbor_n = j_max - j_base;
            auto local_n = min(static_cast<unsigned long>(LOCAL_BIN_SIZE), static_cast<unsigned long>(neighbor_n));
            GrB_Index bin_base = j_base;
            for (GrB_Index j = 0; j < local_n; j++)
            {
                GrB_Index desNode = Aj[j + j_base];
                GrB_Index label = labels[desNode]; // Label of destination node

                // 1.1 If is a directed graph
                GrB_Index incr = 1;

                // 1.2 Initalize bin & count label
                bool isNew = true;
                // Whether the label is presented in bin
                for (GrB_Index b = 0; b < j; b++)
                {
                    if (local_bin_label[b] == label)
                    {
                        local_bin_count[b] += incr;
                        isNew = false;
                        break;
                    }
                }
                if (isNew)
                {
                    local_bin_label[j] = label;
                    local_bin_count[j] = incr;
                }
                else
                {
                    local_bin_label[j] = (GrB_Index)-1;
                    local_bin_count[j] = (GrB_Index)0;
                }
            }
            if (neighbor_n > LOCAL_BIN_SIZE)
            {
                // For next optimization, parallelize this part if neighbor_n >> LOCAL_BIN_SIZE
                // bin_base = atomicAdd(bin_index, neighbor_n - LOCAL_BIN_SIZE); // allocate space in the global bin
                for (GrB_Index j = 0; j < neighbor_n - LOCAL_BIN_SIZE; j++)
                {
                    GrB_Index desNode = Aj[j + LOCAL_BIN_SIZE + j_base];
                    GrB_Index label = labels[desNode]; // Label of destination node

                    // 1.1 If is a directed graph
                    GrB_Index incr = 1;

                    // 1.2 Initalize bin & count label
                    bool isNew = true;
                    // Whether the label is presented in local bin
                    for (GrB_Index b = 0; b < LOCAL_BIN_SIZE; b++)
                    {
                        if (local_bin_label[b] == label)
                        {
                            local_bin_count[b] += incr;
                            isNew = false;
                            break;
                        }
                    }
                    if (isNew)
                    {
                        for (GrB_Index b = bin_base; b < bin_base + j; b++)
                        {
                            if (bin_label[b] == label)
                            {
                                bin_count[b] += incr;
                                isNew = false;
                                break;
                            }
                        }
                        auto b = bin_base + j;
                        if (isNew)
                        {
                            bin_label[b] = label;
                            bin_count[b] = incr;
                        }
                        else
                        {
                            bin_label[b] = (GrB_Index)-1;
                            bin_count[b] = (GrB_Index)0;
                        }
                    }
                    else
                    {
                        auto b = bin_base + j;
                        bin_label[b] = (GrB_Index)-1;
                        bin_count[b] = (GrB_Index)0;
                    }
                }
            }

            // 2. Find label with maximum frequence
            GrB_Index max_count = (GrB_Index)0;
            GrB_Index min_label = (GrB_Index)-1;
            for (GrB_Index j = 0; j < local_n; j++)
            {
                if (max_count < local_bin_count[j])
                {
                    max_count = local_bin_count[j];
                    min_label = local_bin_label[j];
                }
                else if (max_count == local_bin_count[j] && min_label > local_bin_label[j] && local_bin_label[j] != (GrB_Index)-1)
                {
                    min_label = local_bin_label[j];
                }
            }
            if (neighbor_n > LOCAL_BIN_SIZE)
            {
                for (GrB_Index j = bin_base; j < bin_base + neighbor_n - LOCAL_BIN_SIZE; j++)
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
                }
            }

            // 3. Update label
            new_labels[srcNode] = min_label;
        }
    }
}

// hash table optimization
#define HASH_TABLE_SIZE_FACTOR 2

typedef struct
{
    int iter_count;   // version number
    int relative_idx; // index number in the bin
    GrB_Index label;  // key for hashing
    GrB_Index count;  // value for hashing
} hash_table_item;

// hash function for key = label and capacity = n
__device__ __forceinline__ int hash_func(GrB_Index label, GrB_Index n)
{
    const GrB_Index prime = 2147483647; // A large prime number
    GrB_Index hash = label % prime;
    hash %= n;
    return (int)hash;
}

__device__ __forceinline__ GrB_Index hash_table_get_count(int iter, hash_table_item *htable, GrB_Index start, GrB_Index end, GrB_Index label)
{
    return 0;
}

// __device__ __forceinline__ GrB_Index hash_table_set_count(int iter, hash_table_item* htable, GrB_Index start, GrB_Index end, GrB_Index label, GrB_Index value)
// {
//     return 0;
// }

__device__ __forceinline__ GrB_Index hash_table_inc_count(int iter, hash_table_item *htable, GrB_Index start, GrB_Index end, GrB_Index label, GrB_Index delta)
{
#define gethtable(idx) htable[start + idx]
    GrB_Index capacity = end - start + 1; // capacity should not overflow int32_t even though we use uint64_t, otherwise too large
    int relative_idx = hash_func(label, capacity);
    int location = -1;
    // linear probing
    while (gethtable(relative_idx).iter_count == iter)
    {
        // same iteration number, key already exist, add count
        if (gethtable(relative_idx).label == label)
        {
            gethtable(relative_idx).count += delta;
            return gethtable(relative_idx).count;
        }
        relative_idx = (relative_idx + 1) % capacity;
    }
    // break out of the loop, key not exist, add new key
    if (location < 0)
    {
        location = relative_idx;
    }
    gethtable(location).iter_count = iter;
    gethtable(location).relative_idx = relative_idx;
    gethtable(location).label = label;
    gethtable(location).count = delta;
    return gethtable(location).count;
#undef gethtable
}

__device__ __forceinline__ GrB_Index hash_table_inc_count_atomic(int iter, hash_table_item *htable, GrB_Index start, GrB_Index end, GrB_Index label, GrB_Index delta)
{
#define gethtable(idx) htable[start + idx]
    GrB_Index capacity = end - start + 1; // capacity should not overflow int32_t even though we use uint64_t, otherwise too large
    int relative_idx = hash_func(label, capacity);
    int location = -1;
    // linear probing
    while (gethtable(relative_idx).iter_count == iter)
    {
        // same iteration number, key already exist, add count
        if (gethtable(relative_idx).label == label)
        {
            // gethtable(relative_idx).count += delta;
            auto old_count = atomicAdd(reinterpret_cast<unsigned long long int *>(&(gethtable(relative_idx).count)), static_cast<unsigned long long int>(delta));
            // return gethtable(relative_idx).count;
            return old_count + delta;
        }
        relative_idx = (relative_idx + 1) % capacity;
    }
    // break out of the loop, key not exist, add new key
    if (location < 0)
    {
        location = relative_idx;
    }
    gethtable(location).iter_count = iter;
    gethtable(location).relative_idx = relative_idx;
    gethtable(location).label = label;
    // gethtable(location).count = delta;
    auto old_count = atomicAdd(reinterpret_cast<unsigned long long int *>(&(gethtable(location).count)), static_cast<unsigned long long int>(delta));
    return old_count + delta;
#undef gethtable
}

__global__ void cdlp_base_with_hashing(
    int iteration_count,   // Current iteration, start counting from 1 so that 0 means not yet used
    GrB_Index *Ap,         // Row pointers
    GrB_Index *Aj,         // Column indices
    GrB_Index *labels,     // Labels for each node
    GrB_Index *new_labels, // new labels after each iteration
    GrB_Index N,           // Number of nodes
    bool symmetric,        // Is the matrix symmetric (aka is the graph undirected)
    hash_table_item *htable)
{
    GrB_Index ti = blockDim.x * blockIdx.x + threadIdx.x;
    // Iterate until converge or reaching maximum number
    // Loop through all nodes
    GrB_Index stride = gridDim.x * blockDim.x;
    for (GrB_Index srcNode = ti; srcNode < N; srcNode += stride)
    {
        if (srcNode < N)
        {
            // 1. Count neighbors' labels
            GrB_Index j_base = Ap[srcNode];
            GrB_Index j_max = Ap[srcNode + 1];
            GrB_Index max_count = (GrB_Index)0;
            GrB_Index min_label = (GrB_Index)-1;

            for (GrB_Index j = j_base; j < j_max; j++)
            {
                GrB_Index desNode = Aj[j];
                GrB_Index label = labels[desNode]; // Label of destination node

                // 1.1 If is a directed graph
                GrB_Index incr = 1;

                // 1.2 build hash table
                int segment_start = j_base * HASH_TABLE_SIZE_FACTOR;
                int segment_end = j_max * HASH_TABLE_SIZE_FACTOR - 1; // inclusive index
                GrB_Index new_count = hash_table_inc_count(iteration_count, htable, segment_start, segment_end, label, incr);
                if (new_count > max_count)
                {
                    max_count = new_count;
                    min_label = label;
                }
                else if (new_count == max_count && label < min_label)
                {
                    min_label = label;
                }
            }

            // 2. Update label
            if (min_label != (GrB_Index)-1)
            {
                // labels[srcNode] = min_label; // TODO: potential overflow
                new_labels[srcNode] = min_label;
            }
        }
        // __syncthreads();
    }
}

__global__ void cdlp_child(GrB_Index srcNode, GrB_Index neighbor_n, int iteration_count, GrB_Index *Aj, GrB_Index *labels, GrB_Index *new_labels, GrB_Index j_base, GrB_Index j_max, hash_table_item *htable, GrB_Index *max_counts)
{
    GrB_Index ti = threadIdx.x;
    GrB_Index max_count = (GrB_Index)0;
    GrB_Index min_label = (GrB_Index)-1;
    int segment_start = j_base * HASH_TABLE_SIZE_FACTOR;
    int segment_end = j_max * HASH_TABLE_SIZE_FACTOR - 1; // inclusive index
    __shared__ GrB_Index local_max_counts[BLOCK_DIM2];
    for (GrB_Index j = ti; j < neighbor_n; j += blockDim.x)
    {
        if (j < neighbor_n)
        {
            GrB_Index desNode = Aj[j_base + j];
            GrB_Index label = labels[desNode]; // Label of destination node

            // 1.1 If is a directed graph
            GrB_Index incr = 1;

            // 1.2 build hash table
            GrB_Index new_count = hash_table_inc_count_atomic(iteration_count, htable, segment_start, segment_end, label, incr);
            // atomicMax(reinterpret_cast<unsigned long long int *>(&max_counts[srcNode]), static_cast<unsigned long long int>(new_count));
            if (new_count > max_count)
            {
                max_count = new_count;
                min_label = label;
            }
            else if (new_count == max_count && label < min_label)
            {
                min_label = label;
            }
        }
    }
    local_max_counts[ti] = max_count;
    int stride = 1;
    while (stride < BLOCK_DIM2)
    {
        __syncthreads();
        int index = (threadIdx.x + 1) * stride * 2 - 1;
        if (index < BLOCK_DIM2 && index >= stride)
        {
            local_max_counts[index] = max(local_max_counts[index], local_max_counts[index - stride]);
        }
        stride *= 2;
    }
    __syncthreads();
    //if (max_count == max_counts[srcNode])
    if (max_count == local_max_counts[BLOCK_DIM2 - 1])
    {
        atomicMin(reinterpret_cast<unsigned long long int *>(&new_labels[srcNode]), static_cast<unsigned long long int>(min_label));
    }
    __syncthreads();
}

__global__ void cdlp_base_with_hashing1(
    int iteration_count,   // Current iteration, start counting from 1 so that 0 means not yet used
    GrB_Index *Ap,         // Row pointers
    GrB_Index *Aj,         // Column indices
    GrB_Index *labels,     // Labels for each node
    GrB_Index *new_labels, // new labels after each iteration
    GrB_Index N,           // Number of nodes
    bool symmetric,        // Is the matrix symmetric (aka is the graph undirected)
    hash_table_item *htable,
    GrB_Index *max_counts)
{
    GrB_Index ti = blockDim.x * blockIdx.x + threadIdx.x;
    // Iterate until converge or reaching maximum number
    // Loop through all nodes
    GrB_Index stride = gridDim.x * blockDim.x;
    for (GrB_Index srcNode = ti; srcNode < N; srcNode += stride)
    {
        if (srcNode < N)
        {
            // 1. Count neighbors' labels
            GrB_Index j_base = Ap[srcNode];
            GrB_Index j_max = Ap[srcNode + 1];
            GrB_Index max_count = (GrB_Index)0;
            GrB_Index min_label = (GrB_Index)-1;
            GrB_Index neighbor_n = j_max - j_base;
            if (neighbor_n > PAR_THRESHOLD)
            {
                new_labels[srcNode] = (GrB_Index)-1;
                cdlp_child<<<1, BLOCK_DIM2>>>(srcNode, neighbor_n, iteration_count, Aj, labels, new_labels, j_base, j_max, htable, max_counts);
            }
            else
            {
                // 1.2 build hash table
                int segment_start = j_base * HASH_TABLE_SIZE_FACTOR;
                int segment_end = j_max * HASH_TABLE_SIZE_FACTOR - 1; // inclusive index
                for (GrB_Index j = j_base; j < j_max; j++)
                {
                    GrB_Index desNode = Aj[j];
                    GrB_Index label = labels[desNode]; // Label of destination node

                    // 1.1 If is a directed graph
                    GrB_Index incr = 1;

                    GrB_Index new_count = hash_table_inc_count(iteration_count, htable, segment_start, segment_end, label, incr);
                    if (new_count > max_count)
                    {
                        max_count = new_count;
                        min_label = label;
                    }
                    else if (new_count == max_count && label < min_label)
                    {
                        min_label = label;
                    }
                }

                // 2. Update label
                if (min_label != (GrB_Index)-1)
                {
                    // labels[srcNode] = min_label; // TODO: potential overflow
                    new_labels[srcNode] = min_label;
                }
            }
        }
        // __syncthreads();
    }
}

__host__ void cdlp_gpu(GrB_Index *Ap, GrB_Index Ap_size, GrB_Index *Aj, GrB_Index Aj_size, GrB_Vector *CDLP_handle, GrB_Index N, GrB_Index nnz, bool symmetric, int itermax)
{
    GrB_Index *Ap_k;
    GrB_Index *Aj_k;
    GrB_Index *labels_k;
    GrB_Index *new_labels_k;
    GrB_Index *labels;
    int is_equal = 1;
    int *is_equal_k;
#if optimized_hash || optimized_hash1
    hash_table_item *htable_k;
#if DEBUG_PRINT
    hash_table_item *htable;
#endif
#else
    GrB_Index *bin_count_k, *bin_label_k; // For dynamically counting labels (can be optimized using shared memory plus a overflow global memory)
#endif

    cudaMalloc((void **)&Ap_k, Ap_size);
    cudaMalloc((void **)&Aj_k, Aj_size);
    cudaMalloc((void **)&labels_k, N * sizeof(GrB_Index));
    cudaMalloc((void **)&new_labels_k, N * sizeof(GrB_Index));
    cudaMallocHost((void **)&labels, N * sizeof(GrB_Index));
    cudaMalloc((void **)&is_equal_k, sizeof(int));
#if optimized_hash || optimized_hash1
    cudaMalloc((void **)&htable_k, HASH_TABLE_SIZE_FACTOR * nnz * sizeof(hash_table_item));
#if DEBUG_PRINT
    cudaMallocHost((void **)&htable, HASH_TABLE_SIZE_FACTOR * nnz * sizeof(hash_table_item));
#endif
#else
    cudaMalloc((void **)&bin_count_k, nnz * sizeof(GrB_Index));
    cudaMalloc((void **)&bin_label_k, nnz * sizeof(GrB_Index));
#endif

#if optimized_hash1
    GrB_Index *max_counts_k;
    cudaMalloc((void **)&max_counts_k, N * sizeof(GrB_Index));
    cudaMemset(max_counts_k, 0, N * sizeof(GrB_Index));
#endif

#if DEBUG_PRINT != 0
    // PRINT("FINISH CUDA MALLOC");
    std::cout << "FINISH CUDA MALLOC" << std::endl;
#endif

#if optimized1
    int *bin_index;
    cudaMalloc((void **)&bin_index, sizeof(int));
    cudaMemset(bin_index, 0, sizeof(int));
#endif

    cudaMemcpy(Ap_k, Ap, Ap_size, cudaMemcpyHostToDevice);
    cudaMemcpy(Aj_k, Aj, Aj_size, cudaMemcpyHostToDevice);

#if optimized_hash || optimized_hash1
    cudaMemset(htable_k, 0, nnz * sizeof(hash_table_item));
#endif

#if DEBUG_PRINT != 0
    // PRINT("FINISH CUDA MEMCPY");
    std::cout << "FINISH CUDA MEMCPY" << std::endl;
#endif

    dim3 DimGrid(GRID_DIM, 1, 1);
    dim3 DimBlock(BLOCK_DIM, 1);

    initialize_label<<<DimGrid, DimBlock>>>(labels_k, N);

    timer_start("CDLP_GPU MAIN LOOP USING CUDA KERNEL");
    for (int i = 0; i < itermax; ++i)
    {
        // PRINT("RUNNING ITERATION {}", i);
#if DEBUG_PRINT != 0
        std::cout << "RUNNING ITERATION " << i << std::endl;
#endif

#if optimized1
        cdlp_optimized1<<<DimGrid, DimBlock>>>(Ap_k, Aj_k, labels_k, new_labels_k, N, symmetric, bin_count_k, bin_label_k, bin_index);
#elif optimized_hash
        cdlp_base_with_hashing<<<DimGrid, DimBlock>>>(i + 1, Ap_k, Aj_k, labels_k, new_labels_k, N, symmetric, htable_k);
#elif optimized_hash1
        cdlp_base_with_hashing1<<<DimGrid, DimBlock>>>(i + 1, Ap_k, Aj_k, labels_k, new_labels_k, N, symmetric, htable_k, max_counts_k);
#else
        cdlp_base<<<DimGrid, DimBlock>>>(Ap_k, Aj_k, labels_k, new_labels_k, N, symmetric, bin_count_k, bin_label_k);
#endif
        // cudaDeviceSynchronize();
        cudaMemset(is_equal_k, 1, sizeof(int));
        check_equality<<<DimGrid, DimBlock>>>(labels_k, new_labels_k, N, is_equal_k);
        // cudaDeviceSynchronize();
        cudaMemcpy(&is_equal, is_equal_k, sizeof(int), cudaMemcpyDeviceToHost);
        if (is_equal)
            break;
        else
        {
            // cudaMemcpy(labels_k, new_labels_k, N * sizeof(GrB_Index), cudaMemcpyDeviceToDevice);
            //  optimization: double buffering, avoid memcpy
            std::swap(labels_k, new_labels_k);
#if DEBUG_PRINT
            if (N < 20)
            {
                cudaMemcpy(labels, labels_k, N * sizeof(GrB_Index), cudaMemcpyDeviceToHost);
                for (int j = 0; j < N; ++j)
                {
                    std::cout << labels[j] << " ";
                }
                std::cout << '\n';
                cudaMemcpy(labels, max_counts_k, N * sizeof(GrB_Index), cudaMemcpyDeviceToHost);
                for (int j = 0; j < N; ++j)
                {
                    std::cout << labels[j] << " ";
                }
                std::cout << '\n';
                cudaMemcpy(htable, htable_k, HASH_TABLE_SIZE_FACTOR * nnz * sizeof(hash_table_item), cudaMemcpyDeviceToHost);
                for (int j = 0; j < HASH_TABLE_SIZE_FACTOR * nnz; ++j)
                {
                    std::cout << htable[j].label << '@' << htable[j].count << " ";
                }
                std::cout << '\n';
            }
#endif

#if optimized_hash1
            cudaMemset(max_counts_k, 0, N * sizeof(GrB_Index));
            cudaMemset(htable_k, 0, HASH_TABLE_SIZE_FACTOR * nnz * sizeof(hash_table_item));
#endif
        }
    }
    cudaDeviceSynchronize();

    timer_stop();

    cudaMemcpy(labels, labels_k, N * sizeof(GrB_Index), cudaMemcpyDeviceToHost);

#if DEBUG_PRINT != 0
    // PRINT("RUNNING CUDA FREE");
    std::cout << "RUNNING CUDA FREE" << std::endl;
#endif

    cudaFree(Ap_k);
    cudaFree(Aj_k);
    cudaFree(labels_k);
    cudaFree(new_labels_k);
    cudaFree(is_equal_k);
#if optimized_hash || optimized_hash1
    cudaFree(htable_k);
#if DEBUG_PRINT
    cudaFreeHost(htable);
#endif
#else
    cudaFree(bin_count_k);
    cudaFree(bin_label_k);
#endif
#if optimized1
    cudaFree(bin_index);
#endif
#if optimized_hash1
    cudaFree(max_counts_k);
#endif

#if DEBUG_PRINT != 0
    // PRINT("CONVERT TO GRB_VECTOR");
    std::cout << "CONVERT TO GRB_VECTOR" << std::endl;
#endif

    GrB_Vector CDLP = NULL;
    GrB_Vector_new(&CDLP, GrB_UINT64, N);
    for (GrB_Index i = 0; i < N; i++)
    {
        GrB_Vector_setElement_UINT64(CDLP, labels[i], i);
    }
    (*CDLP_handle) = CDLP;

    cudaFreeHost(labels);
}
#include <iostream>
#include <vector>
#include "cdlp_cuda.cuh"
#include "cdlp_kernel.cuh"

namespace CUDA_CDLP
{

    const char *grb_type_to_string(GrB_Type type)
    {
        if (type == GrB_BOOL)
        {
            return "GrB_BOOL";
        }
        else if (type == GrB_INT8)
        {
            return "GrB_INT8";
        }
        else if (type == GrB_INT16)
        {
            return "GrB_INT16";
        }
        else if (type == GrB_INT32)
        {
            return "GrB_INT32";
        }
        else if (type == GrB_INT64)
        {
            return "GrB_INT64";
        }
        else if (type == GrB_UINT8)
        {
            return "GrB_UINT8";
        }
        else if (type == GrB_UINT16)
        {
            return "GrB_UINT16";
        }
        else if (type == GrB_UINT32)
        {
            return "GrB_UINT32";
        }
        else if (type == GrB_UINT64)
        {
            return "GrB_UINT64";
        }
        else if (type == GrB_FP32)
        {
            return "GrB_FP32";
        }
        else if (type == GrB_FP64)
        {
            return "GrB_FP64";
        }
        else
        {
            return "Unknown GrB_Type";
        }
    }

    int test_cuda_device_query()
    {

        PRINT("running test_cuda_device_query");

        int deviceCount;

        cudaGetDeviceCount(&deviceCount);

        timer_start("[CUDA][TIMER] Getting GPU Data."); //@@ start a timer

        for (int dev = 0; dev < deviceCount; dev++)
        {
            cudaDeviceProp deviceProp;

            cudaGetDeviceProperties(&deviceProp, dev);

            if (dev == 0)
            {
                if (deviceProp.major == 9999 && deviceProp.minor == 9999)
                {
                    PRINT("No CUDA GPU has been detected");
                    return -1;
                }
                else if (deviceCount == 1)
                {
                    //@@ WbLog is a provided logging API (similar to Log4J).
                    //@@ The logging function wbLog takes a level which is either
                    //@@ OFF, FATAL, ERROR, WARN, INFO, DEBUG, or trace and a
                    //@@ message to be printed.
                    PRINT("There is 1 device supporting CUDA");
                }
                else
                {
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
        GrB_Matrix A,            // input matrix
        bool symmetric,          // denote whether the matrix is symmetric
        bool sanitize,           // if true, ensure A is binary
        int itermax,             // max number of iterations,
        double *t                // t [0] = sanitize time, t [1] = cdlp time, in seconds
    )
    {
        // check input
        if (CDLP_handle == NULL || t == NULL)
        {
            return GrB_NULL_POINTER;
        }
#if DEBUG_PRINT != 0
        // PRINT("Starting LAGraph_cdlp_gpu");
        std::cout << "Starting LAGraph_cdlp_gpu" << std::endl;
#endif
        // set timing to zero
        t[0] = 0; // sanitize time
        t[1] = 0; // CDLP time

        // check input matrix
        GrB_Index n;   // number of nodes in the graph
        // GrB_Index nz;  // non-zero in the matrix
        // GrB_Index nnz; //
        GrB_Matrix_nrows(&n, A);
        // GrB_Matrix_nvals(&nz, A);
        // if (!symmetric)
        // {
        //     nnz = 2 * nz;
        // }
        // else
        // {
        //     nnz = nz;
        // }
        // PRINT("nnz value is {}", nnz);

        if (sanitize)
        {
            // TODO: sanitize the input matrix (make sure it is binary, not needed for basic tests though)
        }

        // convert input matrix to CUDA sparse matrix CSR format
        // variables for matrix export return
        GrB_Type type; // should be GrB_UINT64, suited for CUDA kernel
        size_t typesize;
        GrB_Index nrows, ncols;              // Matrix Dimensions, should be n x n
        GrB_Index *Ap;                       // row "pointers", Ap_size >= nrows+1
        GrB_Index *Aj;                       // row indices, Aj_size >= nvals(A)
        void *Ax;                            // value array in void*
        bool A_jumbled = false;              // If jumbled is returned as false, column indices will appear in ascending order within each row. Pass NULL to enforce sorting.
        GrB_Index Ap_size, Aj_size, Ax_size; // size of three arrays exported, in bytes
        bool A_iso = false;                  // if return value is true, A is isomorphic (used for internal optimization)

        // This method takes O(1) time if the matrix is already in CSR format internally.
        //  Otherwise, the matrix is converted to CSR format and then exported.
        //  A will be freed after this call if successful
#if DEBUG_PRINT != 0
        // PRINT("Exporting matrix to CSR format");
        std::cout << "Exporting matrix to CSR format" << std::endl;
#endif 
        GxB_Matrix_export_CSR(&A, &type, &nrows, &ncols, &Ap, &Aj,
                              &Ax, &Ap_size, &Aj_size, &Ax_size,
                              &A_iso, &A_jumbled, NULL);
        GxB_Type_size(&typesize, type);

        if (type != GrB_INT64)
        {
            PRINT("type not supported with CUDA kernel");
            PRINT("type is {}", grb_type_to_string(type));
            return GrB_NOT_IMPLEMENTED;
        }
#if DEBUG_PRINT != 0
        PRINT("nrows is {}, ncols is {}", nrows, ncols);
        PRINT("Ap_size is {}, Aj_size is {}, Ax_size is {}", Ap_size, Aj_size, Ax_size);
        PRINT("A_iso is {}, A_jumbled is {}", A_iso, A_jumbled);
        // print arrays
        // PRINT("Ap is:");
        // std::vector<uint64_t> Ap_vec(Ap, Ap + (Ap_size / sizeof(uint64_t)));
        // std::string str1;
        // for (size_t i = 0; i < Ap_vec.size(); ++i)
        // {
        //     str1 += std::to_string(Ap_vec[i]);
        //     if (i < Ap_vec.size() - 1)
        //     {
        //         str1 += ' ';
        //     }
        // }
        // PRINT("{}", str1);
        // PRINT("Aj is:");
        // std::vector<uint64_t> Aj_vec(Aj, Aj + (Aj_size / sizeof(uint64_t)));
        // std::string str2;
        // for (size_t i = 0; i < Aj_vec.size(); ++i)
        // {
        //     str2 += std::to_string(Aj_vec[i]);
        //     if (i < Aj_vec.size() - 1)
        //     {
        //         str2 += ' ';
        //     }
        // }
        // PRINT("{}", str2);
        // PRINT("Ax is:");
        // std::vector<int64_t> Ax_vec((int64_t *)Ax, (int64_t *)Ax + (Ax_size / sizeof(int64_t)));
        // std::string str3;
        // for (size_t i = 0; i < Ax_vec.size(); ++i)
        // {
        //     str3 += std::to_string(Ax_vec[i]);
        //     if (i < Ax_vec.size() - 1)
        //     {
        //         str3 += ' ';
        //     }
        // }
        // PRINT("{}", str3);
#endif

        // Call CUDA kernel
        std::cout << "Processing starts at: " << GetCurrentMilliseconds() << std::endl;
        cdlp_gpu(Ap, Ap_size, Aj, Aj_size, CDLP_handle, n, symmetric, itermax);
        std::cout << "Processing ends at: " << GetCurrentMilliseconds() << std::endl;

        // free matrix mem
        free(Ap);
        free(Aj);
        free(Ax);

        return GrB_SUCCESS;
    }

}
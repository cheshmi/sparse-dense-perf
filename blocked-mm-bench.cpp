#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>         // cusparseSpMM
#include <iomanip>
#include <iostream>
#include <ostream>
#include <random>
#include <time.h>
#include <unordered_set>
#include <vector>

#include "benchmark/benchmark.h"

// #include <iostream>
// #include <benchmark/benchmark.h>

// Utility function to check CUDA errors
#define CHECK_CUDA(func)                                                       \
{                                                                             \
    cudaError_t status = (func);                                             \
    if (status != cudaSuccess) {                                             \
        printf("CUDA API failed at %s line %d with error: %s (%d)\n",        \
               __FILE__, __LINE__, cudaGetErrorString(status), status);       \
        return;                                                 \
    }                                                                         \
}

// Utility function to check cuBLAS errors
#define CHECK_CUBLAS(func)                                                    \
{                                                                            \
    cublasStatus_t status = (func);                                          \
    if (status != CUBLAS_STATUS_SUCCESS) {                                   \
        printf("cuBLAS API failed at %s line %d with error: %d\n",          \
               __FILE__, __LINE__, status);                                   \
       return;                                                  \
    }                                                                         \
}

// Utility function to check cuSPARSE errors
#define CHECK_CUSPARSE(func)                                                  \
{                                                                            \
    cusparseStatus_t status = (func);                                        \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                 \
        printf("cuSPARSE API failed at %s line %d with error: %d\n",        \
               __FILE__, __LINE__, status);                                   \
       return;                                                   \
    }                                                                         \
}


void generate_chckerboard_pattern(float* matrix, int rows, int cols, int block_size) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i * cols + j] = (i / block_size + j / block_size) % 2 == 0 ? 1.0f : 0.0f;
        }
    }
}


// Function to generate random sparse matrix with given sparsity
void generate_sparse_matrix(float* matrix, int rows, int cols, float sparsity, int seed = 0) {
    srand(seed);
    auto total_nnz = (int)(rows * cols * (1-sparsity));
    // generate random unique integer indices between 0 and rows*cols
    std::unordered_set<int> unique_numbers;
    std::vector<int> indices;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, rows * cols - 1);

    while (unique_numbers.size() < total_nnz) {
        int num = dis(gen);
        if (unique_numbers.insert(num).second) {
            indices.push_back(num);
        }
    }
    // fill the matrix with random values
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = 0.0f;
    }
    for (int i = 0; i < total_nnz; i++) {
        matrix[indices[i]] = 1.0; //(float)rand() / RAND_MAX;
    }
}


// Function to convert dense matrix to CSR format
void dense_to_csr(float* dense, int rows, int cols,
                  float** values, int** row_ptr, int** col_ind,
                  int* nnz) {
    // First pass: count non-zero elements
    *nnz = 0;
    for (int i = 0; i < rows * cols; i++) {
        if (dense[i] != 0.0f) {
            (*nnz)++;
        }
    }

    // Allocate memory
    *values = (float*)malloc(*nnz * sizeof(float));
    *col_ind = (int*)malloc(*nnz * sizeof(int));
    *row_ptr = (int*)malloc((rows + 1) * sizeof(int));

    // Second pass: fill CSR arrays
    int count = 0;
    (*row_ptr)[0] = 0;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (dense[i * cols + j] != 0.0f) {
                (*values)[count] = dense[i * cols + j];
                (*col_ind)[count] = j;
                count++;
            }
        }
        (*row_ptr)[i + 1] = count;
    }
}


void MeasureGEMMPerformance(cublasHandle_t handle, int m, int n, int k,
                            const float* d_A, const float* d_B, float* d_C,
                            const float alpha, const float beta,
                            float& elapsed_time) {
    cudaEvent_t event_start, event_stop;
    cudaEventCreate(&event_start);
    cudaEventCreate(&event_stop);

    cudaEventRecord(event_start, 0);
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             n, m, k,
                             &alpha,
                             d_B, n,
                             d_A, k,
                             &beta,
                             d_C, n));
    cudaEventRecord(event_stop, 0);
    cudaEventSynchronize(event_stop);
    cudaEventElapsedTime(&elapsed_time, event_start, event_stop);

    cudaEventDestroy(event_start);
    cudaEventDestroy(event_stop);
}



// benchmark for sparse matrix multiplication
static void BM_cuBLAS_CUDA(benchmark::State &state) {
    int m = state.range(0);
    int n = state.range(1);
    int k = state.range(2);
    auto block_size = state.range(3);
    // Initialize CUDA handles
    cublasHandle_t cublas_handle;
    CHECK_CUBLAS(cublasCreate(&cublas_handle));

    // Allocate host memory for dense matrices
    float *h_A = (float*)malloc(m * k * sizeof(float));
    float *h_B = (float*)malloc(k * n * sizeof(float));
    float *h_C = (float*)malloc(m * n * sizeof(float));

    // Generate random sparse matrix A and dense matrix B
    generate_chckerboard_pattern(h_A, m, k, block_size);
    generate_sparse_matrix(h_B, k, n, 0.0f);  // B is dense

    // Allocate device memory for dense matrices
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc((void**)&d_A, m * k * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_B, k * n * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_C, m * n * sizeof(float)));

    // Constants for GEMM
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, m * k * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, k * n * sizeof(float), cudaMemcpyHostToDevice));

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    const char *device_name = prop.name;

    // add gpu name to the log
    // add sparsity to the log
    state.SetLabel("Arch:" + std::string(device_name));
    for (auto _: state) {
        float iteration_time_ms = 0.0f;
        MeasureGEMMPerformance(cublas_handle, m, n, k, d_A, d_B, d_C,
                               alpha, beta, iteration_time_ms);
        state.SetIterationTime(iteration_time_ms * 1e-3);
    }
    // Final cleanup
    free(h_A);
    free(h_B);
    free(h_C);
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUBLAS(cublasDestroy(cublas_handle));
}



static void BM_CUSPARSE_SPMM(benchmark::State &state) {
    auto m = state.range(0);
    auto n = state.range(1);
    auto k = state.range(2);
    auto block_size = state.range(3);
    // cudaError_t cudaStatus = cudaSetDevice(0);
    // if (cudaStatus != cudaSuccess) {
    //     printf("CUDA context creation failed with error: %s\n", cudaGetErrorString(cudaStatus));
    //     return;
    // }
    // Check CUDA context creation using cudaFree(0)
    cudaError_t cudaStatus = cudaFree(0);
    if (cudaStatus != cudaSuccess) {
        printf("CUDA context creation failed with error: %s\n", cudaGetErrorString(cudaStatus));
        return;
    }

    // Initialize CUDA handles
    cusparseHandle_t cusparse_handle;
    if (cusparseCreate(&cusparse_handle) != CUSPARSE_STATUS_SUCCESS) {
        state.SkipWithError("cusparseCreate failed");
        return;
    }

    // Allocate host memory for dense matrices
    auto *h_A = (float*)malloc(m * k * sizeof(float));
    auto *h_B = (float*)malloc(k * n * sizeof(float));
    auto *h_C = (float*)malloc(m * n * sizeof(float));

    // Generate random sparse matrix A and dense matrix B
    generate_chckerboard_pattern(h_A, m, k, block_size);
    generate_sparse_matrix(h_B, k, n, 0.0f);  // B is dense

    // Allocate device memory for dense matrices
    float  *d_B, *d_C;
    CHECK_CUDA(cudaMalloc((void**)&d_B, k * n * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_C, m * n * sizeof(float)));

    // Constants for GEMM
    const float alpha = 1.0f;
    const float beta = 0.0f;
     // Convert matrix A to CSR format
     float *h_csrVal;
     int *h_csrRowPtr, *h_csrColInd;
     int nnz;
     dense_to_csr(h_A, m, k, &h_csrVal, &h_csrRowPtr, &h_csrColInd, &nnz);


     // Allocate device memory for sparse matrix
     float *d_csrVal;
     int *d_csrRowPtr, *d_csrColInd;
     CHECK_CUDA(cudaMalloc((void**)&d_csrVal, nnz * sizeof(float)));
     CHECK_CUDA(cudaMalloc((void**)&d_csrRowPtr, (m + 1) * sizeof(int)));
     CHECK_CUDA(cudaMalloc((void**)&d_csrColInd, nnz * sizeof(int)));

     // Copy data to device
     CHECK_CUDA(cudaMemcpy(d_csrVal, h_csrVal, nnz * sizeof(float), cudaMemcpyHostToDevice));
     CHECK_CUDA(cudaMemcpy(d_csrRowPtr, h_csrRowPtr, (m + 1) * sizeof(int), cudaMemcpyHostToDevice));
     CHECK_CUDA(cudaMemcpy(d_csrColInd, h_csrColInd, nnz * sizeof(int), cudaMemcpyHostToDevice));

     // Create matrix descriptors for sparse operations
     cusparseMatDescr_t descr;
     CHECK_CUSPARSE(cusparseCreateMatDescr(&descr));
     CHECK_CUSPARSE(cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL));
     CHECK_CUSPARSE(cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO));


     // Create cusparseSpMat and cusparseDnMat descriptors
     cusparseSpMatDescr_t matA;
     cusparseDnMatDescr_t matB, matC;

     CHECK_CUSPARSE(cusparseCreateCsr(&matA, m, k, nnz,
                                      d_csrRowPtr, d_csrColInd, d_csrVal,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, k * n * sizeof(float), cudaMemcpyHostToDevice));
     CHECK_CUSPARSE(cusparseCreateDnMat(&matB, k, n, k, d_B,
                                        CUDA_R_32F, CUSPARSE_ORDER_COL));
    CHECK_CUDA(cudaMemcpy(d_C, h_C, m * n * sizeof(float), cudaMemcpyHostToDevice));
     CHECK_CUSPARSE(cusparseCreateDnMat(&matC, m, n, m, d_C,
                                        CUDA_R_32F, CUSPARSE_ORDER_COL));


     // Temporary buffer
     size_t bufferSize;
     void* dBuffer = nullptr;
     CHECK_CUSPARSE(cusparseSpMM_bufferSize(cusparse_handle,
                                            CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            &alpha, matA, matB, &beta, matC,
                                            CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize));
     CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    const char *device_name = prop.name;

    // add sparsity to the log
    state.SetLabel("Arch:" + std::string(device_name));

    for (auto _: state) {
        float iteration_time_ms = 0.0f;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start, 0);
        // Perform SpMM

     // Perform SpMM
     CHECK_CUSPARSE(cusparseSpMM(cusparse_handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, matB, &beta, matC,
                                 CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, dBuffer));
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&iteration_time_ms, start, stop);
        state.SetIterationTime(iteration_time_ms*1e-3);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

    }

     // Clean up
     CHECK_CUSPARSE(cusparseDestroySpMat(matA));
     CHECK_CUSPARSE(cusparseDestroyDnMat(matB));
     CHECK_CUSPARSE(cusparseDestroyDnMat(matC));
     //CHECK_CUDA(cudaFree(dBuffer));

    CHECK_CUDA( cudaMemcpy(h_C, d_C, m * n * sizeof(float),
               cudaMemcpyDeviceToHost) );
    for (int i = 0; i< m*n; i++) {
        if (h_C[i] != k/2) {
            printf("Error: h_C[%d] = %f\n", i, h_C[i]);
        }
    }

     // Clean up sparse matrix resources
     free(h_csrVal);
     free(h_csrRowPtr);
     free(h_csrColInd);
     CHECK_CUDA(cudaFree(d_csrVal));
     CHECK_CUDA(cudaFree(d_csrRowPtr));
     CHECK_CUDA(cudaFree(d_csrColInd));
     CHECK_CUSPARSE(cusparseDestroyMatDescr(descr));

     // Final cleanup
     free(h_A);
     free(h_B);
     free(h_C);
     CHECK_CUSPARSE(cusparseDestroy(cusparse_handle));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
}



static void BM_BELL_CUSPARSE_MM(benchmark::State &state) {
    auto m = state.range(0);
    auto n = state.range(1);
    auto k = state.range(2);
    auto block_size = state.range(3);
    // cudaError_t cudaStatus = cudaSetDevice(0);
    // if (cudaStatus != cudaSuccess) {
    //     printf("CUDA context creation failed with error: %s\n", cudaGetErrorString(cudaStatus));
    //     return;
    // }
    // Check CUDA context creation using cudaFree(0)
    cudaError_t cudaStatus = cudaFree(0);
    if (cudaStatus != cudaSuccess) {
        printf("CUDA context creation failed with error: %s\n", cudaGetErrorString(cudaStatus));
        return;
    }

    // Initialize CUDA handles
    cusparseHandle_t cusparse_handle;
    if (cusparseCreate(&cusparse_handle) != CUSPARSE_STATUS_SUCCESS) {
        state.SkipWithError("cusparseCreate failed");
        return;
    }

    // Allocate host memory for dense matrices
    auto *h_A = (float*)malloc(m * k * sizeof(float));
    auto *h_B = (float*)malloc(k * n * sizeof(float));
    auto *h_C = (float*)malloc(m * n * sizeof(float));

    // Generate random sparse matrix A and dense matrix B
    generate_chckerboard_pattern(h_A, m, k, block_size);
    generate_sparse_matrix(h_B, k, n, 0.0f);  // B is dense


    // Allocate device memory for dense matrices
    float  *d_B, *d_C;
    CHECK_CUDA(cudaMalloc((void**)&d_B, k * n * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_C, m * n * sizeof(float)));

    // Constants for GEMM
    const float alpha = 1.0f;
    const float beta = 0.0f;

     // convert dense to blocked ELL format
    // Host problem definition
    int   num_rows     = m;
    int   num_cols     = k;
    int   ld           = num_cols;
    int   dense_size   = ld * num_rows;
    float *h_dense    = h_A;
    int   ell_blk_size = block_size;
    int   ell_width    = num_cols / 2; // assuming square matrix divided into block_size

    int   bell_nnz          = ell_width * num_rows;
    int col_ind_size = bell_nnz / (ell_blk_size * ell_blk_size);
    int   *h_ell_columns = (int*)malloc(col_ind_size   * sizeof(int));
    // fill the ell_columns for checkerboard pattern
    int ell_width_block = ell_width/ell_blk_size;
    for (int i = 0; i < num_rows/ell_blk_size; i++) {
        for (int j = 0; j < ell_width_block; j++) {
            h_ell_columns[i * ell_width_block + j] = i ? i%2 == 0 : i+1;
        }
    }
    float *h_ell_values  = (float*)malloc(bell_nnz * sizeof(float));
    for (int i = 0; i < bell_nnz; i++) {
        h_ell_values[i] = 1.0f;
    }

    // Device memory management
    int   *d_ell_columns=NULL;
    float *d_ell_values=NULL;
    //CHECK_CUDA( cudaMalloc((void**) &d_dense, dense_size * sizeof(float)))
    CHECK_CUDA( cudaMalloc((void**) &d_ell_columns,
                           col_ind_size* sizeof(int)))
    CHECK_CUDA( cudaMalloc((void**) &d_ell_values,
                           bell_nnz * sizeof(float)))
    // CHECK_CUDA( cudaMemcpy(d_dense, h_dense, dense_size * sizeof(float),
    //                        cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(d_ell_columns, h_ell_columns,
                           col_ind_size * sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(d_ell_values, h_ell_values,
                           bell_nnz * sizeof(float),
                           cudaMemcpyHostToDevice) )


    // CUSPARSE APIs
    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matBell;
    CHECK_CUSPARSE( cusparseCreate(&handle) )

    // Create sparse matrix B in Blocked ELL format
    CHECK_CUSPARSE( cusparseCreateBlockedEll(&matBell, num_rows, num_cols,
                                             ell_blk_size, ell_width,
                                             d_ell_columns, d_ell_values,
                                             CUSPARSE_INDEX_32I,
                                             CUSPARSE_INDEX_BASE_ZERO,
                                             CUDA_R_32F) )

    cusparseDnMatDescr_t matB, matC;
    CHECK_CUDA(cudaMemcpy(d_B, h_B, k * n * sizeof(float), cudaMemcpyHostToDevice));
     CHECK_CUSPARSE(cusparseCreateDnMat(&matB, k, n, k, d_B,
                                        CUDA_R_32F, CUSPARSE_ORDER_COL));
    CHECK_CUDA(cudaMemcpy(d_C, h_C, k * n * sizeof(float), cudaMemcpyHostToDevice));
     CHECK_CUSPARSE(cusparseCreateDnMat(&matC, m, n, m, d_C,
                                        CUDA_R_32F, CUSPARSE_ORDER_COL));

     // Temporary buffer
     size_t bufferSize;
     void* dBuffer = nullptr;
     CHECK_CUSPARSE(cusparseSpMM_bufferSize(cusparse_handle,
                                            CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            &alpha, matBell, matB, &beta, matC,
                                            CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize));
     CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    const char *device_name = prop.name;

    // add sparsity to the log
    state.SetLabel("Arch:" + std::string(device_name));

    for (auto _: state) {
        float iteration_time_ms = 0.0f;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start, 0);
        // Perform SpMM

     // Perform SpMM
     CHECK_CUSPARSE(cusparseSpMM(cusparse_handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matBell, matB, &beta, matC,
                                 CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, dBuffer));
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&iteration_time_ms, start, stop);
        state.SetIterationTime(iteration_time_ms*1e-3);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    // verify the result
    CHECK_CUDA( cudaMemcpy(h_C, d_C, m * n * sizeof(float),
                           cudaMemcpyDeviceToHost) );
    for (int i = 0; i< m*n; i++) {
        if (h_C[i] != k/2) {
            printf("Error: h_C[%d] = %f\n", i, h_C[i]);
        }
    }

     // Clean up
     CHECK_CUSPARSE(cusparseDestroySpMat(matBell));
     CHECK_CUSPARSE(cusparseDestroyDnMat(matB));
     CHECK_CUSPARSE(cusparseDestroyDnMat(matC));

     // Final cleanup
     free(h_A);
     free(h_B);
     free(h_C);
     CHECK_CUSPARSE(cusparseDestroy(cusparse_handle));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
}

// Define constants
const int M = 1024;
const int N = 64;
const int K = 1024;
const int iters = 100;

// Register the function as a benchmark

BENCHMARK(BM_cuBLAS_CUDA)->ArgsProduct({{M}, {M}, {64, 128, 256}, {4, 8, 16, 32}})
->ArgNames({"M", "N", "K", "BlockSize"})->Unit(benchmark::kMillisecond)->UseManualTime()->Iterations(iters);


BENCHMARK(BM_CUSPARSE_SPMM)->ArgsProduct({{M}, {M}, {64, 128, 256}, {4, 8, 16, 32}})
    ->ArgNames({"M", "N", "K", "BlockSize"})->Unit(benchmark::kMillisecond)->UseManualTime()->Iterations(iters);

BENCHMARK(BM_BELL_CUSPARSE_MM)->ArgsProduct({{M}, {M}, {64, 128, 256}, {4, 8, 16, 32}})
    ->ArgNames({"M", "N", "K", "BlockSize"})->Unit(benchmark::kMillisecond)->UseManualTime()->Iterations(iters);

// NOTE: manual time is reported in ms and as real time (not CPU time)


// Run the benchmark
//BENCHMARK_MAIN();
int main(int argc, char** argv) {
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    return 0;
}

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
#include <mma.h>

using namespace nvcuda;

#include "benchmark/benchmark.h"

// The only dimensions currently supported by WMMA
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

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

// Function to generate random sparse matrix with given sparsity
void generate_sparse_matrix(float *matrix, int rows, int cols, float sparsity, int seed = 0) {
    srand(seed);
    auto total_nnz = (int) (rows * cols * (1 - sparsity));
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
        matrix[indices[i]] = (float) rand() / RAND_MAX;
    }
}

// Function to convert dense matrix to CSR format
void dense_to_csr(float *dense, int rows, int cols,
                  float **values, int **row_ptr, int **col_ind,
                  int *nnz) {
    // First pass: count non-zero elements
    *nnz = 0;
    for (int i = 0; i < rows * cols; i++) {
        if (dense[i] != 0.0f) {
            (*nnz)++;
        }
    }

    // Allocate memory
    *values = (float *) malloc(*nnz * sizeof(float));
    *col_ind = (int *) malloc(*nnz * sizeof(int));
    *row_ptr = (int *) malloc((rows + 1) * sizeof(int));

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


__global__ void wmma_example(half *a, half *b, float *c, int M, int N, int K, float alpha, float beta) {
    // Leading dimensions. Packed with no transpositions.
    int lda = M;
    int ldb = K;
    int ldc = M;

    // Tile using a 2D grid
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

    // Declare the fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fill_fragment(acc_frag, 0.0f);

    // Loop over k
    for (int i = 0; i < K; i += WMMA_K) {
        int aRow = warpM * WMMA_M;
        int aCol = i;

        int bRow = i;
        int bCol = warpN * WMMA_N;

        // Bounds checking
        if (aRow < M && aCol < K && bRow < K && bCol < N) {
            // Load the inputs
            wmma::load_matrix_sync(a_frag, a + aRow + aCol * lda, lda);
            wmma::load_matrix_sync(b_frag, b + bRow + bCol * ldb, ldb);

            // Perform the matrix multiplication
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

        }
    }

    // Load in the current value of c, scale it by beta, and add this our result scaled by alpha
    int cRow = warpM * WMMA_M;
    int cCol = warpN * WMMA_N;

    if (cRow < M && cCol < N) {
        wmma::load_matrix_sync(c_frag, c + cRow + cCol * ldc, ldc, wmma::mem_col_major);

#pragma unroll
        for (int i = 0; i < c_frag.num_elements; i++) {
            c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
        }

        // Store the output
        wmma::store_matrix_sync(c + cRow + cCol * ldc, c_frag, ldc, wmma::mem_col_major);
    }
}

__global__ void convertFp32ToFp16(half *out, float *in, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx];
    }
}

void MeasureGEMMPerformance(cublasHandle_t handle, int m, int n, int k,
                            const float *d_A, const float *d_B, float *d_C,
                            const float alpha, const float beta,
                            float &elapsed_time) {
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

void MeasureTensorCoreGEMMPerformance(cublasHandle_t handle, int m, int n, int k,
                                      const float *d_A, const float *d_B, float *d_C,
                                      const float alpha, const float beta,
                                      float &elapsed_time) {
    cudaEvent_t event_start, event_stop;
    cudaEventCreate(&event_start);
    cudaEventCreate(&event_stop);

    cublasGemmAlgo_t CuBlasALG = static_cast<cublasGemmAlgo_t>(0);

    cudaEventRecord(event_start, 0);
    CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                              n, m, k,
                              &alpha,
                              d_B, CUDA_R_32F, n,
                              d_A, CUDA_R_32F, k,
                              &beta,
                              d_C, CUDA_R_32F, n, CUDA_R_32F, CuBlasALG));
    cudaEventRecord(event_stop, 0);
    cudaEventSynchronize(event_stop);
    cudaEventElapsedTime(&elapsed_time, event_start, event_stop);

    cudaEventDestroy(event_start);
    cudaEventDestroy(event_stop);
}

void MeasureTensorCorePerformance(int m, int n, int k, half *d_A,
                                  half *d_B, float *d_C,
                                  float alpha, float beta, float &elapsed_time) {

    cudaEvent_t event_start, event_stop;
    cudaEventCreate(&event_start);
    cudaEventCreate(&event_stop);

    // First: using WMMA
    dim3 gridDim;
    dim3 blockDim;

    // blockDim.x must be a multple of warpSize
    // 128x4 means we have 16 warps and a block computes a 64x64 output tile
    blockDim.x = 128;
    blockDim.y = 4;

    gridDim.x = (m + (WMMA_M * blockDim.x / 32 - 1)) / (WMMA_M * blockDim.x / 32);
    gridDim.y = (m + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y);

    cudaEventRecord(event_start, 0);

    wmma_example <<< gridDim, blockDim >>>(d_A, d_B, d_C, m, n, k, alpha, beta);

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
    auto sparsity = state.range(3);
    float sparsity_ratio = sparsity / 100.0f;
    // Initialize CUDA handles
    cublasHandle_t cublas_handle;
    CHECK_CUBLAS(cublasCreate(&cublas_handle));

    // Allocate host memory for dense matrices
    float *h_A = (float *) malloc(m * k * sizeof(float));
    float *h_B = (float *) malloc(k * n * sizeof(float));
    float *h_C = (float *) malloc(m * n * sizeof(float));

    // Generate random sparse matrix A and dense matrix B
    generate_sparse_matrix(h_A, m, k, sparsity_ratio);
    generate_sparse_matrix(h_B, k, n, 0.0f);  // B is dense

    // Allocate device memory for dense matrices
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc((void **) &d_A, m * k * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void **) &d_B, k * n * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void **) &d_C, m * n * sizeof(float)));

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
        state.SetIterationTime(iteration_time_ms);


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

static void BM_cuBLAS_TensorCore(benchmark::State &state) {
    int m = state.range(0);
    int n = state.range(1);
    int k = state.range(2);
    auto sparsity = state.range(3);
    float sparsity_ratio = sparsity / 100.0f;
    // Initialize CUDA handles
    cublasHandle_t cublas_handle;
    CHECK_CUBLAS(cublasCreate(&cublas_handle));
    cublasSetMathMode(cublas_handle, CUBLAS_TENSOR_OP_MATH);
    // Allocate host memory for dense matrices
    float *h_A = (float *) malloc(m * k * sizeof(float));
    float *h_B = (float *) malloc(k * n * sizeof(float));
    float *h_C = (float *) malloc(m * n * sizeof(float));

    // Generate random sparse matrix A and dense matrix B
    generate_sparse_matrix(h_A, m, k, sparsity_ratio);
    generate_sparse_matrix(h_B, k, n, 0.0f);  // B is dense

    // Allocate device memory for dense matrices
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc((void **) &d_A, m * k * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void **) &d_B, k * n * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void **) &d_C, m * n * sizeof(float)));

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
        MeasureTensorCoreGEMMPerformance(cublas_handle, m, n, k, d_A, d_B, d_C,
                               alpha, beta, iteration_time_ms);
        state.SetIterationTime(iteration_time_ms);


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
    auto sparsity = state.range(3);
    float sparsity_ratio = sparsity / 100.0f;
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
    auto *h_A = (float *) malloc(m * k * sizeof(float));
    auto *h_B = (float *) malloc(k * n * sizeof(float));
    auto *h_C = (float *) malloc(m * n * sizeof(float));

    // Generate random sparse matrix A and dense matrix B
    generate_sparse_matrix(h_A, m, k, sparsity_ratio);
    generate_sparse_matrix(h_B, k, n, 0.0f);  // B is dense

    // Allocate device memory for dense matrices
    float *d_B, *d_C;
    CHECK_CUDA(cudaMalloc((void **) &d_B, k * n * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void **) &d_C, m * n * sizeof(float)));

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
    CHECK_CUDA(cudaMalloc((void **) &d_csrVal, nnz * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void **) &d_csrRowPtr, (m + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void **) &d_csrColInd, nnz * sizeof(int)));

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
    CHECK_CUSPARSE(cusparseCreateDnMat(&matB, k, n, k, d_B,
                                       CUDA_R_32F, CUSPARSE_ORDER_COL));
    CHECK_CUSPARSE(cusparseCreateDnMat(&matC, m, n, m, d_C,
                                       CUDA_R_32F, CUSPARSE_ORDER_COL));

    // Temporary buffer
    size_t bufferSize;
    void *dBuffer = nullptr;
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
        state.SetIterationTime(iteration_time_ms);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    // Clean up
    CHECK_CUSPARSE(cusparseDestroySpMat(matA));
    CHECK_CUSPARSE(cusparseDestroyDnMat(matB));
    CHECK_CUSPARSE(cusparseDestroyDnMat(matC));
    //CHECK_CUDA(cudaFree(dBuffer));


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

static void BM_TensorCore_GEMM(benchmark::State &state) {
    int m = state.range(0);
    int n = state.range(1);
    int k = state.range(2);
    auto sparsity = state.range(3);
    float sparsity_ratio = sparsity / 100.0f;

    // Allocate host memory for dense matrices
    float *h_A = (float *) malloc(m * k * sizeof(float));
    float *h_B = (float *) malloc(k * n * sizeof(float));
    float *h_C = (float *) malloc(m * n * sizeof(float));

    // Generate random sparse matrix A and dense matrix B
    generate_sparse_matrix(h_A, m, k, sparsity_ratio);
    generate_sparse_matrix(h_B, k, n, 0.0f);

    // Allocate device memory for dense matrices
    float *d_A, *d_B, *d_C;
    half *d_A_half, *d_B_half;
    CHECK_CUDA(cudaMalloc((void **) &d_A, m * k * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void **) &d_B, k * n * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void **) &d_C, m * n * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void **) &d_A_half, m * k * sizeof(half)));
    CHECK_CUDA(cudaMalloc((void **) &d_B_half, k * n * sizeof(half)));

    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, m * k * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, k * n * sizeof(float), cudaMemcpyHostToDevice));

    // convert generated matrices to half precision
    convertFp32ToFp16<<<(m * k + 255) / 256, 256>>>(d_A_half, d_A, m * k);
    convertFp32ToFp16<<<(k * n + 255) / 256, 256>>>(d_B_half, d_B, k * n);

    // Constants for GEMM
    const float alpha = 1.0f;
    const float beta = 0.0f;


    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    const char *device_name = prop.name;

    // add gpu name to the log
    // add sparsity to the log
    state.SetLabel("Arch:" + std::string(device_name));
    for (auto _: state) {
        float iteration_time_ms = 0.0f;

        MeasureTensorCorePerformance(m, n, k, d_A_half, d_B_half, d_C, alpha, beta, iteration_time_ms);

        state.SetIterationTime(iteration_time_ms);

    }
    // Final cleanup
    free(h_A);
    free(h_B);
    free(h_C);
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
}

// Define constants
const int M = 2048;
const int N = 2048;
const int K = 1024;
const int iters = 100;

// Register the function as a benchmark

BENCHMARK(BM_cuBLAS_CUDA)->Args({M, N, 32, 50})->Args({M, N, 64, 50})->Args({M, N, 128, 50})->Args({M, N, 512, 50})->
        ArgNames({"M", "N", "K", "Sparsity"})->Unit(benchmark::kMillisecond)->UseManualTime()->Iterations(iters);

BENCHMARK(BM_TensorCore_GEMM)->Args({M, N, 32, 50})->Args({M, N, 64, 50})->Args({M, N, 128, 50})->Args(
        {M, N, 512, 50})->
        ArgNames({"M", "N", "K", "Sparsity"})->Unit(benchmark::kMillisecond)->UseManualTime()->Iterations(iters);

BENCHMARK(BM_cuBLAS_TensorCore)->Args({M, N, 32, 50})->Args({M, N, 64, 50})->Args({M, N, 128, 50})->Args(
        {M, N, 512, 50})->
        ArgNames({"M", "N", "K", "Sparsity"})->Unit(benchmark::kMillisecond)->UseManualTime()->Iterations(iters);

BENCHMARK(BM_CUSPARSE_SPMM)->Args({M, N, 32, 50})->Args({M, N, 64, 50})->Args({M, N, 128, 50})->Args({M, N, 512, 50})
        ->Args({M, N, 32, 60})->Args({M, N, 64, 60})->Args({M, N, 128, 60})->Args({M, N, 512, 60})
        ->Args({M, N, 32, 70})->Args({M, N, 64, 70})->Args({M, N, 128, 70})->Args({M, N, 512, 70})
        ->Args({M, N, 32, 80})->Args({M, N, 64, 80})->Args({M, N, 128, 80})->Args({M, N, 512, 80})
        ->Args({M, N, 32, 90})->Args({M, N, 64, 90})->Args({M, N, 128, 90})->Args({M, N, 512, 90})
        ->Args({M, N, 32, 95})->Args({M, N, 64, 95})->Args({M, N, 128, 95})->Args({M, N, 512, 95})
        ->Args({M, N, 32, 99})->Args({M, N, 64, 99})->Args({M, N, 128, 99})->Args({M, N, 512, 99})
        ->ArgNames({"M", "N", "K", "Sparsity"})->Unit(benchmark::kMillisecond)->UseManualTime()->Iterations(iters);
// NOTE: manual time is reported in ms and as real time (not CPU time)



// Run the benchmark
//BENCHMARK_MAIN();
int main(int argc, char **argv) {
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    return 0;
}

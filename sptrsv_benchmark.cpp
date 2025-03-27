/*
 * Copyright 1993-2022 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>         // cusparseSpSV
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE

#include "io.h"
#include "benchmark/benchmark.h"

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        exit(-1);                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        exit(-1);                                                   \
    }                                                                          \
}


int main1(int argc, char* argv[]) {
    // Host problem definition
    // const int A_num_rows      = 4;
    // const int A_num_cols      = 4;
    // const int A_nnz           = 9;
    // int       hA_csrOffsets[] = { 0, 3, 4, 7, 9 };
    // int       hA_columns[]    = { 0, 2, 3, 1, 0, 2, 3, 1, 3 };
    // float     hA_values[]     = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
    //                               6.0f, 7.0f, 8.0f, 9.0f };
    auto file_path = argv[1];
    CSR *A;
    std::ifstream in_file(file_path);
    if (!in_file.is_open()) {
        printf("Error: Cannot open file %s\n", file_path);
        return EXIT_FAILURE;
    }
    read_mtx_csr_real(in_file, A);
    int *hA_csrOffsets = A->p;
    int *hA_columns    = A->i;
    float *hA_values   = A->x;
    int A_num_rows = A->m;
    int A_num_cols = A->n;
    int A_nnz = A->nnz;




    // float     hX[]            = { 1.0f, 8.0f, 23.0f, 52.0f };
    // float     hY[]            = { 0.0f, 0.0f, 0.0f, 0.0f };
    // float     hY_result[]     = { 1.0f, 2.0f, 3.0f, 4.0f };
    float *hX = (float*)malloc(A_num_cols * sizeof(float));
    float *hY = (float*)malloc(A_num_cols * sizeof(float));
    float *hY_result = (float*)malloc(A_num_cols * sizeof(float));
    // set hy_result to all 1.0f,
    for (int i = 0; i < A_num_rows; i++) {
        hY_result[i] = 1.0f;
        hY[i] = 0;
    }
    spmv_csr(*A, hY_result, hX);
    // for (int i = 0; i < A_num_rows; i++) {
    //     std::cout<<hX[i]<<std::endl;
    // }

    float     alpha           = 1.0f;
    //--------------------------------------------------------------------------
    // Device memory management
    int   *dA_csrOffsets, *dA_columns;
    float *dA_values, *dX, *dY;
    CHECK_CUDA( cudaMalloc((void**) &dA_csrOffsets,
                           (A_num_rows + 1) * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dA_columns, A_nnz * sizeof(int))        )
    CHECK_CUDA( cudaMalloc((void**) &dA_values,  A_nnz * sizeof(float))      )
    CHECK_CUDA( cudaMalloc((void**) &dX,         A_num_cols * sizeof(float)) )
    CHECK_CUDA( cudaMalloc((void**) &dY,         A_num_rows * sizeof(float)) )

    CHECK_CUDA( cudaMemcpy(dA_csrOffsets, hA_csrOffsets,
                           (A_num_rows + 1) * sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA_columns, hA_columns, A_nnz * sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA_values, hA_values, A_nnz * sizeof(float),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dX, hX, A_num_cols * sizeof(float),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dY, hY, A_num_rows * sizeof(float),
                           cudaMemcpyHostToDevice) )
    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;
    cusparseSpSVDescr_t  spsvDescr;
    CHECK_CUSPARSE( cusparseCreate(&handle) )
    // Create sparse matrix A in CSR format
    CHECK_CUSPARSE( cusparseCreateCsr(&matA, A_num_rows, A_num_cols, A_nnz,
                                      dA_csrOffsets, dA_columns, dA_values,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
    // Create dense vector X
    CHECK_CUSPARSE( cusparseCreateDnVec(&vecX, A_num_cols, dX, CUDA_R_32F) )
    // Create dense vector y
    CHECK_CUSPARSE( cusparseCreateDnVec(&vecY, A_num_rows, dY, CUDA_R_32F) )
    // Create opaque data structure, that holds analysis data between calls.
    CHECK_CUSPARSE( cusparseSpSV_createDescr(&spsvDescr) )
    // Specify Lower|Upper fill mode.
    cusparseFillMode_t fillmode = CUSPARSE_FILL_MODE_LOWER;
    CHECK_CUSPARSE( cusparseSpMatSetAttribute(matA, CUSPARSE_SPMAT_FILL_MODE,
                                              &fillmode, sizeof(fillmode)) )
    // Specify Unit|Non-Unit diagonal type.
    cusparseDiagType_t diagtype = CUSPARSE_DIAG_TYPE_NON_UNIT;
    CHECK_CUSPARSE( cusparseSpMatSetAttribute(matA, CUSPARSE_SPMAT_DIAG_TYPE,
                                              &diagtype, sizeof(diagtype)) )
    // allocate an external buffer for analysis
    CHECK_CUSPARSE( cusparseSpSV_bufferSize(
                                handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, matA, vecX, vecY, CUDA_R_32F,
                                CUSPARSE_SPSV_ALG_DEFAULT, spsvDescr,
                                &bufferSize) )
    CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )
    CHECK_CUSPARSE( cusparseSpSV_analysis(
                                handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, matA, vecX, vecY, CUDA_R_32F,
                                CUSPARSE_SPSV_ALG_DEFAULT, spsvDescr, dBuffer) )
    // execute SpSV
    CHECK_CUSPARSE( cusparseSpSV_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                       &alpha, matA, vecX, vecY, CUDA_R_32F,
                                       CUSPARSE_SPSV_ALG_DEFAULT, spsvDescr) )

    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
    CHECK_CUSPARSE( cusparseDestroyDnVec(vecX) )
    CHECK_CUSPARSE( cusparseDestroyDnVec(vecY) )
    CHECK_CUSPARSE( cusparseSpSV_destroyDescr(spsvDescr));
    CHECK_CUSPARSE( cusparseDestroy(handle) )
    //--------------------------------------------------------------------------
    // device result check
    CHECK_CUDA( cudaMemcpy(hY, dY, A_num_rows * sizeof(float),
                           cudaMemcpyDeviceToHost) )
    int correct = 1;
    for (int i = 0; i < A_num_rows; i++) {
        if (abs(hY[i] - hY_result[i]) > 1e-5) {
            correct = 0;
            break;
        }
    }
    if (correct)
        printf("spsv_csr_example test PASSED\n");
    else
        printf("spsv_csr_example test FAILED: wrong result\n");
    //--------------------------------------------------------------------------
    // device memory deallocation
    CHECK_CUDA( cudaFree(dBuffer) )
    CHECK_CUDA( cudaFree(dA_csrOffsets) )
    CHECK_CUDA( cudaFree(dA_columns) )
    CHECK_CUDA( cudaFree(dA_values) )
    CHECK_CUDA( cudaFree(dX) )
    CHECK_CUDA( cudaFree(dY) )
    return EXIT_SUCCESS;
}

std::vector<std::string> mat_list;

static void BM_CUSPARSE_SPSV(benchmark::State& state) {
    auto file_path_id = state.range(0);
    auto file_path = mat_list[file_path_id];
    std::cout<<file_path<<std::endl;
    CSR *A;
    std::ifstream in_file(file_path);
    if (!in_file.is_open()) {
        printf("Error: Cannot open file %s\n", file_path);
        return;
    }
    read_mtx_csr_real(in_file, A);
    int *hA_csrOffsets = A->p;
    int *hA_columns    = A->i;
    float *hA_values   = A->x;
    int A_num_rows = A->m;
    int A_num_cols = A->n;
    int A_nnz = A->nnz;

    float *hX = (float*)malloc(A_num_cols * sizeof(float));
    float *hY = (float*)malloc(A_num_cols * sizeof(float));
    float *hY_result = (float*)malloc(A_num_cols * sizeof(float));
    for (int i = 0; i < A_num_rows; i++) {
        hY_result[i] = 1.0f;
        hY[i] = 0;
    }
    spmv_csr(*A, hY_result, hX);

    float alpha = 1.0f;
    int *dA_csrOffsets, *dA_columns;
    float *dA_values, *dX, *dY;
    CHECK_CUDA(cudaMalloc((void**)&dA_csrOffsets, (A_num_rows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&dA_columns, A_nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&dA_values, A_nnz * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&dX, A_num_cols * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&dY, A_num_rows * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(dA_csrOffsets, hA_csrOffsets, (A_num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dA_columns, hA_columns, A_nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dA_values, hA_values, A_nnz * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dX, hX, A_num_cols * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dY, hY, A_num_rows * sizeof(float), cudaMemcpyHostToDevice));

    cusparseHandle_t handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;
    void* dBuffer = NULL;
    size_t bufferSize = 0;
    cusparseSpSVDescr_t spsvDescr;
    CHECK_CUSPARSE(cusparseCreate(&handle));
    CHECK_CUSPARSE(cusparseCreateCsr(&matA, A_num_rows, A_num_cols, A_nnz, dA_csrOffsets, dA_columns, dA_values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, A_num_cols, dX, CUDA_R_32F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, A_num_rows, dY, CUDA_R_32F));
    CHECK_CUSPARSE(cusparseSpSV_createDescr(&spsvDescr));
    cusparseFillMode_t fillmode = CUSPARSE_FILL_MODE_LOWER;
    CHECK_CUSPARSE(cusparseSpMatSetAttribute(matA, CUSPARSE_SPMAT_FILL_MODE, &fillmode, sizeof(fillmode)));
    cusparseDiagType_t diagtype = CUSPARSE_DIAG_TYPE_NON_UNIT;
    CHECK_CUSPARSE(cusparseSpMatSetAttribute(matA, CUSPARSE_SPMAT_DIAG_TYPE, &diagtype, sizeof(diagtype)));
    CHECK_CUSPARSE(cusparseSpSV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, vecY, CUDA_R_32F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescr, &bufferSize));
    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));


    for (auto _ : state) {
        float iteration_time_ms = 0.0f;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start, 0);
        CHECK_CUSPARSE(cusparseSpSV_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, vecY, CUDA_R_32F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescr, dBuffer));
        CHECK_CUSPARSE(cusparseSpSV_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, vecY, CUDA_R_32F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescr));
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&iteration_time_ms, start, stop);
        state.SetIterationTime(iteration_time_ms*1e-3);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    CHECK_CUSPARSE(cusparseDestroySpMat(matA));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecX));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecY));
    CHECK_CUSPARSE(cusparseSpSV_destroyDescr(spsvDescr));
    CHECK_CUSPARSE(cusparseDestroy(handle));

    CHECK_CUDA(cudaMemcpy(hY, dY, A_num_rows * sizeof(float), cudaMemcpyDeviceToHost));
    int correct = 1;
    for (int i = 0; i < A_num_rows; i++) {
        if (abs(hY[i] - hY_result[i]) > 1e-5) {
            correct = 0;
            break;
        }
    }
    if (correct)
        printf("spsv_csr_example test PASSED\n");
    else
        printf("spsv_csr_example test FAILED: wrong result\n");

    CHECK_CUDA(cudaFree(dBuffer));
    CHECK_CUDA(cudaFree(dA_csrOffsets));
    CHECK_CUDA(cudaFree(dA_columns));
    CHECK_CUDA(cudaFree(dA_values));
    CHECK_CUDA(cudaFree(dX));
    CHECK_CUDA(cudaFree(dY));
    free(hX);
    free(hY);
    free(hY_result);
}

// Run the benchmark
//BENCHMARK_MAIN();
int iters = 50;
BENCHMARK(BM_CUSPARSE_SPSV)->DenseRange(0, 40)->Unit(benchmark::kMillisecond)->UseManualTime()->Iterations(iters);;

int main(int argc, char** argv) {
    std::string base_matrix_path = "/home/kazem/UFDB/";
    get_mat_list(base_matrix_path+"/spd_list.txt", base_matrix_path, mat_list);
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    return 0;
}

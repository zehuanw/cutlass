/***************************************************************************************************
* Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
*
* Redistribution and use in source and binary forms, with or without modification, are permitted
* provided that the following conditions are met:
*     * Redistributions of source code must retain the above copyright notice, this list of
*       conditions and the following disclaimer.
*     * Redistributions in binary form must reproduce the above copyright notice, this list of
*       conditions and the following disclaimer in the documentation and/or other materials
*       provided with the distribution.
*     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
*       to endorse or promote products derived from this software without specific prior written
*       permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
* IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
* FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
* FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
* BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
* OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
* STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*
**************************************************************************************************/

#include <cutlass/cutlass.h>
#include <tools/test/unit/gemm/gemm_testbed.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>
////////////////////////////////////////////////////////////////////////////////////////////////////
#define FINAL_MASK 0xffffffff
#define JUDGESIGN(x)  ((x<0) ? -1:1)
#define MAX_BLOCK_DIM_SIZE 65535
void mm_cpu(float *A, float*B, float *C, int M, int K, int N)
{
    for(int i = 0; i < M; ++i)
    {   
        for(int j = 0; j < N; ++j)
        {
            double sum = 0;
            for(int k = 0; k < K; ++k)
            {

                double a = A[i * K + k]; 
                double b = B[k * N + j]; 
                sum += a * b;
            }
            C[i * N + j] = (float) sum;
        }
    }   
}
    __inline__ __device__
float warpReduceMax(float val)
{
#pragma unroll
    for(int mask = 16; mask > 0; mask >>= 1)
        val = max(val, __shfl_xor_sync(FINAL_MASK, val, mask, 32));
    return val;
}

    __inline__ __device__
float blockReduceMax(float val)
{
    static __shared__ float shared[32];
    int lane = threadIdx.x & 0x1f; // in-warp Idx
    int wid = threadIdx.x >> 5;  // warp Idx

    val = warpReduceMax(val); // get maxx in each warp

    if(lane == 0) // record in-warp maxx by warp Idx
        shared[wid] = val;

    __syncthreads();


    val = (threadIdx.x < (blockDim.x >> 5 )) ? shared[lane] : 0;
    val = warpReduceMax(val);

    return val;
}

template <typename T>
static void set_memory_value(T* C, size_t length, float value){
    T* temp = new T[length];
    for(int i = 0; i < length; i++){
        temp[i] = value;
    }
    cudaMemcpy(C, temp, length * sizeof(T), cudaMemcpyHostToDevice);
    delete(temp);
}

template <typename T>
static void print_memory(T* C, size_t length, size_t col){
    T* temp = new T[length];
    cudaMemcpy(temp, C, length*sizeof(T), cudaMemcpyDeviceToHost);
    FILE *fd = fopen("out", "w");
    for(int i = 0; i < length; ++i){
        fprintf(fd,"row %d col %d value %f\n", (int) i / col, (int) i % col, temp[i]);
    }
    fclose(fd);
    delete(temp);
}

    __global__
void scale_row(float* src, int8_t* dst, float* scalePtr, int width) 
{
    __shared__ float s_scaler;
    __shared__ float s_max;

    int row_id = blockIdx.x;
    int col_id = threadIdx.x;

    int offset = row_id * width;

    float local_val = fabsf(src[offset + col_id]);

    //local max reduce in width / 1024 elements
#pragma unroll
    for(int i = col_id + blockDim.x; i < width; i += blockDim.x)
        local_val = max(local_val, fabsf(src[offset + i]));

    float max_val = blockReduceMax(local_val);
    if(threadIdx.x == 0)
    {   
        s_max = max_val;
        s_scaler = ((1 << 7) - 1) / s_max;
        scalePtr[row_id] = 1 / s_scaler;

    }   
    __syncthreads();

    float val;
    for(int i = col_id; i < width; i += blockDim.x)
    {   
        val = src[offset + i]; 
        if(val >=  s_max - FLT_EPSILON)
            dst[offset + i] = (int8_t)((1 << 7) - 1); 
        else if(val < -s_max + FLT_EPSILON)
            dst[offset + i] = (int8_t)(-(1 << 7));
        else
            dst[offset + i] = (int8_t)(val * s_scaler + JUDGESIGN(val) * 0.5);
    }   
}
    __global__
void scale_col(float* src, int8_t* dst, float* scalePtr, int height) 
{
    __shared__ float s_scaler;
    __shared__ float s_max;

    int col_id = blockIdx.x;
    int row_id = threadIdx.x;

    float local_val = fabsf(src[row_id * gridDim.x + col_id]);

    //local max reduce in width / 1024 elements
#pragma unroll
    for(int i = row_id + blockDim.x; i < height; i += blockDim.x)
        local_val = max(local_val, fabsf(src[i * gridDim.x + col_id]));

    float max_val = blockReduceMax(local_val);
    if(threadIdx.x == 0)
    {
        s_max = max_val;
        s_scaler = ((1 << 7) - 1) / s_max;
        scalePtr[col_id] = 1 / s_scaler;
    }
    __syncthreads();

    float val;
    int out_index;
    for(int i = row_id; i < height; i += blockDim.x)
    {
        out_index = i * gridDim.x + col_id;
        val = src[out_index];
        if(val >=  s_max - FLT_EPSILON)
            dst[out_index] = (int8_t)((1 << 7) - 1);
        else if(val < -s_max + FLT_EPSILON)
            dst[out_index] = (int8_t)(-(1 << 7));
        else
            dst[out_index] = (int8_t)(val * s_scaler + JUDGESIGN(val) * 0.5);
    }
}
    template <typename GemmTraits_>
static void run_gemm_fused(
        int M,
        int N,
        int K,
        int lda,
        int ldb,
        int ldc,
        int alpha = int(1),
        int beta = int(0)
        ) {

    printf("M %d N %d K %d\n", M, N, K);
    typedef cutlass::gemm::Gemm<GemmTraits_> Gemm;
    typename Gemm::Params params;

    float *A, *B;
    int8_t *A_int8, *B_int8;
    float *ptr_C_initial, *ptr_C;
    float *scalerA, *scalerB;

    cudaMalloc((void**)&scalerA, sizeof(float) * M);
    cudaMalloc((void**)&scalerB, sizeof(float) * N);

    cudaMalloc(&A, M * K * sizeof(float));
    cudaMalloc(&B, N * K * sizeof(float));
    cudaMalloc(&A_int8, M * K * sizeof(int8_t));
    cudaMalloc(&B_int8, N * K * sizeof(int8_t));
    cudaMalloc(&ptr_C_initial,  M * N * sizeof(float));
    cudaMalloc(&ptr_C, M * N *sizeof(float));

    //initialization

    float *h_A, *h_B, *h_C;
    h_A = (float*)malloc(sizeof(float) * M * K);
    h_B = (float*)malloc(sizeof(float) * K * N);
    h_C = (float*)malloc(sizeof(float) * M * N);

    srand(NULL);
    for(int i = 0; i < M * K; ++i)
        h_A[i] = 0.001f;

    for(int i = 0; i < K * N; ++i)
        h_B[i] = 0.001f;

    cudaMemcpy(A, h_A, sizeof(float) * M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(B, h_B, sizeof(float) * K * N, cudaMemcpyHostToDevice);

    set_memory_value(ptr_C_initial, M * N, 0.0f);
    set_memory_value(ptr_C, M *  N, 0.0f);

    dim3 grid(M);
    dim3 block(min(1024, K));

    printf("scale_row grid.x %d block.x %d\n", grid.x, block.x);
    scale_row<<<grid, block>>>(A, A_int8, scalerA, K);

    grid.x = N;
    printf("scale_col grid.x %d block.x %d\n", grid.x, block.x);
    scale_col<<<grid, block>>>(B, B_int8, scalerB, K);

    params.initialize(
            M,
            N,
            K,
            alpha,
            A_int8,
            lda,
            B_int8,
            ldb,
            beta,
            ptr_C_initial,
            ldc,
            ptr_C,
            ldc);

    Gemm::launch(params, scalerA, scalerB, ptr_C, M, N);

    cudaError_t result = cudaDeviceSynchronize();
    ASSERT_EQ(result, cudaSuccess) << "\nCUDA kernel launch error: " << cudaGetErrorString(result)
        << "\n";

    print_memory(ptr_C, M * N, N);

    mm_cpu(h_A, h_B, h_C, M, K, N);
    
    cudaFree(A_int8);
    cudaFree(B_int8);
    cudaFree(ptr_C_initial);
    cudaFree(ptr_C);
}

template <typename GemmTraits_>
static void run_gemm_fused(
        int m,
        int n,
        int k,
        int alpha = int(1),
        int beta = int(0)) {
    printf("call here\n");
    int lda = GemmTraits_::kLayoutA == cutlass::MatrixLayout::kColumnMajor ? m : k;
    int ldb = GemmTraits_::kLayoutB == cutlass::MatrixLayout::kColumnMajor ? k : n;
    run_gemm_fused<GemmTraits_>(m, n, k, lda, ldb, m, alpha, beta);
}

////////////////////////////////////////////////////////////////////////////////////////////////////




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

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename GemmTraits_>
static void run_gemm(
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    typename test::GemmTestbedTraits<typename GemmTraits_::Epilogue::Scalar>::host_type alpha =
        typename test::GemmTestbedTraits<typename GemmTraits_::Epilogue::Scalar>::host_type(1),
    typename test::GemmTestbedTraits<typename GemmTraits_::Epilogue::Scalar>::host_type beta =
        typename test::GemmTestbedTraits<typename GemmTraits_::Epilogue::Scalar>::host_type(0)) {
  typedef cutlass::gemm::Gemm<GemmTraits_> Gemm;
  typename Gemm::Params params;

  test::GemmTestbed<
      typename test::GemmTestbedTraits<
          typename GemmTraits_::GemmConfig::ScalarA>::host_type,  // AType
      typename test::GemmTestbedTraits<
          typename GemmTraits_::GemmConfig::ScalarB>::host_type,  // BType
      typename test::GemmTestbedTraits<
          typename GemmTraits_::Epilogue::ScalarC>::host_type,  // CType
      typename test::GemmTestbedTraits<
          typename GemmTraits_::Epilogue::Accumulators::Element>::host_type,  // Accumulator
      typename test::GemmTestbedTraits<typename GemmTraits_::Epilogue::Scalar>::host_type  // Scalar
      >
      testbed(m,
              n,
              k,
              lda,
              ldb,
              ldc,
              cutlass::convert(GemmTraits_::kLayoutA),
              cutlass::convert(GemmTraits_::kLayoutB),
              alpha,
              beta);

  testbed.initialize();

  if (testbed.has_cublas_support()) {
    EXPECT_TRUE(testbed.verify_host_with_cublas());
  }

  params.initialize(testbed.M(),
                    testbed.N(),
                    testbed.K(),
                    testbed.alpha,
                    testbed.ptr_A(),
                    testbed.lda(),
                    testbed.ptr_B(),
                    testbed.ldb(),
                    testbed.beta,
                    testbed.ptr_C_initial(),
                    testbed.ldc(),
                    testbed.ptr_computed(),
                    testbed.ldc());

  Gemm::launch(params);

  cudaError_t result = cudaDeviceSynchronize();
  ASSERT_EQ(result, cudaSuccess) << "\nCUDA kernel launch error: " << cudaGetErrorString(result)
                                 << "\n";

  if (testbed.has_cublas_support()) {
    ASSERT_TRUE(testbed.verify_with_cublas());
  } else {
    ASSERT_TRUE(testbed.verify_with_host());
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename GemmTraits_>
static void run_gemm(
    int m,
    int n,
    int k,
    typename test::GemmTestbedTraits<typename GemmTraits_::Epilogue::Scalar>::host_type alpha =
        typename test::GemmTestbedTraits<typename GemmTraits_::Epilogue::Scalar>::host_type(1),
    typename test::GemmTestbedTraits<typename GemmTraits_::Epilogue::Scalar>::host_type beta =
        typename test::GemmTestbedTraits<typename GemmTraits_::Epilogue::Scalar>::host_type(0)) {
  int lda = GemmTraits_::kLayoutA == cutlass::MatrixLayout::kColumnMajor ? m : k;
  int ldb = GemmTraits_::kLayoutB == cutlass::MatrixLayout::kColumnMajor ? k : n;

  run_gemm<GemmTraits_>(m, n, k, lda, ldb, m, alpha, beta);
}


////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename GemmTraits_>
static void run_gemm_test_int_to_float(
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    /* typename test::GemmTestbedTraits<typename GemmTraits_::Epilogue::Scalar>::host_type alpha = */
    /*     typename test::GemmTestbedTraits<typename GemmTraits_::Epilogue::Scalar>::host_type(1), */
    /* typename test::GemmTestbedTraits<typename GemmTraits_::Epilogue::Scalar>::host_type beta = */
    /*     typename test::GemmTestbedTraits<typename GemmTraits_::Epilogue::Scalar>::host_type(0) */
    int alpha = int(1),
    int beta = int(0)
  ) {
  typedef cutlass::gemm::Gemm<GemmTraits_> Gemm;
  typename Gemm::Params params;
  std::cout << "m = " << m << " n = " << n << " k = " << k << " lda = " << lda << " ldb = " << ldb 
	    << " ldc = " << ldc;
  test::GemmTestbed_int_to_float<
      /* typename test::GemmTestbedTraits< */
      /*     typename GemmTraits_::GemmConfig::ScalarA>::host_type,  // AType */
      int8_t,
      /* typename test::GemmTestbedTraits< */
      /*     typename GemmTraits_::GemmConfig::ScalarB>::host_type,  // BType */
      int8_t,
      /* typename test::GemmTestbedTraits< */
      /*     typename GemmTraits_::Epilogue::ScalarC>::host_type,  // CType */
      int,
      /* typename test::GemmTestbedTraits< */
      /*     typename GemmTraits_::Epilogue::Accumulators::Element>::host_type,  // Accumulator */
      int,
      //typename test::GemmTestbedTraits<typename GemmTraits_::Epilogue::Scalar>::host_type  // Scalar
      int
      >
      testbed(m,
              n,
              k,
              lda,
              ldb,
              ldc,
              cutlass::convert(GemmTraits_::kLayoutA),
              cutlass::convert(GemmTraits_::kLayoutB),
              alpha,
              beta);

  testbed.initialize();

  if (testbed.has_cublas_support()) {
    EXPECT_TRUE(testbed.verify_host_with_cublas());
  }

  params.initialize(testbed.M(),
                    testbed.N(),
                    testbed.K(),
                    testbed.alpha,
                    testbed.ptr_A(),
                    testbed.lda(),
                    testbed.ptr_B(),
                    testbed.ldb(),
                    testbed.beta,
                    testbed.ptr_C_initial(),
                    testbed.ldc(),
                    testbed.ptr_computed(),
                    testbed.ldc());

  Gemm::launch(params);

  cudaError_t result = cudaDeviceSynchronize();
  ASSERT_EQ(result, cudaSuccess) << "\nCUDA kernel launch error: " << cudaGetErrorString(result)
                                 << "\n";

  if (testbed.has_cublas_support()) {
    ASSERT_TRUE(testbed.verify_with_cublas());
  } else {
    ASSERT_TRUE(testbed.verify_with_host());
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename GemmTraits_>
static void run_gemm_test_int_to_float(
    int m,
    int n,
    int k,
    //    typename test::GemmTestbedTraits<typename GemmTraits_::Epilogue::Scalar>::host_type alpha =
    //    typename test::GemmTestbedTraits<typename GemmTraits_::Epilogue::Scalar>::host_type(1),
    int alpha = int(1),
    //typename test::GemmTestbedTraits<typename GemmTraits_::Epilogue::Scalar>::host_type beta =
    //    typename test::GemmTestbedTraits<typename GemmTraits_::Epilogue::Scalar>::host_type(0)) {
    int beta = int(0)) {
  int lda = GemmTraits_::kLayoutA == cutlass::MatrixLayout::kColumnMajor ? m : k;
  int ldb = GemmTraits_::kLayoutB == cutlass::MatrixLayout::kColumnMajor ? k : n;

  run_gemm_test_int_to_float<GemmTraits_>(m, n, k, lda, ldb, m, alpha, beta);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
static void set_memory(T* C, size_t length){
    T* temp = new T[length];
    for(int i = 0; i < length; i++){
      temp[i] = 1;
    }
    cudaMemcpy(C, temp, length*sizeof(T), cudaMemcpyHostToDevice);
    delete(temp);
}

template <typename T>
static void print_memory(T* C, size_t length){
    T* temp = new T[length];
    cudaMemcpy(temp, C, length*sizeof(T), cudaMemcpyDeviceToHost);
    for(int i = 0; i < length; i++){
      std::cout << temp[i] << ",";
    }
    delete(temp);
}


////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename GemmTraits_>
static void run_gemm_test_int_to_float2(
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    /* typename test::GemmTestbedTraits<typename GemmTraits_::Epilogue::Scalar>::host_type alpha = */
    /*     typename test::GemmTestbedTraits<typename GemmTraits_::Epilogue::Scalar>::host_type(1), */
    /* typename test::GemmTestbedTraits<typename GemmTraits_::Epilogue::Scalar>::host_type beta = */
    /*     typename test::GemmTestbedTraits<typename GemmTraits_::Epilogue::Scalar>::host_type(0) */
    int alpha = int(1),
    int beta = int(0)
  ) {
  typedef cutlass::gemm::Gemm<GemmTraits_> Gemm;
  typename Gemm::Params params;
  std::cout << "m = " << m << " n = " << n << " k = " << k << " lda = " << lda << " ldb = " << ldb 
	    << " ldc = " << ldc;
  /* test::GemmTestbed_int_to_float< */
  /*     /\* typename test::GemmTestbedTraits< *\/ */
  /*     /\*     typename GemmTraits_::GemmConfig::ScalarA>::host_type,  // AType *\/ */
  /*     int8_t, */
  /*     /\* typename test::GemmTestbedTraits< *\/ */
  /*     /\*     typename GemmTraits_::GemmConfig::ScalarB>::host_type,  // BType *\/ */
  /*     int8_t, */
  /*     /\* typename test::GemmTestbedTraits< *\/ */
  /*     /\*     typename GemmTraits_::Epilogue::ScalarC>::host_type,  // CType *\/ */
  /*     int, */
  /*     /\* typename test::GemmTestbedTraits< *\/ */
  /*     /\*     typename GemmTraits_::Epilogue::Accumulators::Element>::host_type,  // Accumulator *\/ */
  /*     int, */
  /*     //typename test::GemmTestbedTraits<typename GemmTraits_::Epilogue::Scalar>::host_type  // Scalar */
  /*     int */
  /*     > */
  /*     testbed(m, */
  /*             n, */
  /*             k, */
  /*             lda, */
  /*             ldb, */
  /*             ldc, */
  /*             cutlass::convert(GemmTraits_::kLayoutA), */
  /*             cutlass::convert(GemmTraits_::kLayoutB), */
  /*             alpha, */
  /*             beta); */

  /* testbed.initialize(); */

  /* if (testbed.has_cublas_support()) { */
  /*   EXPECT_TRUE(testbed.verify_host_with_cublas()); */
  /* } */

  int8_t *ptr_A, *ptr_B;
  int *ptr_C_initial, *ptr_computed;

  cudaMalloc(&ptr_A, m*k*sizeof(int8_t));
  cudaMalloc(&ptr_B, n*k*sizeof(int8_t));
  cudaMalloc(&ptr_C_initial, m*n*sizeof(int));
  cudaMalloc(&ptr_computed, m*n*sizeof(int));

  //initialization
  set_memory(ptr_A, m*k);
  set_memory(ptr_B, n*k);
  set_memory(ptr_C_initial, m*n);
  set_memory(ptr_computed, m*n);

  params.initialize(m,
                    n,
                    k,
                    alpha,
                    ptr_A,
                    lda,
                    ptr_B,
                    ldb,
                    beta,
                    ptr_C_initial,
                    ldc,
                    ptr_computed,
                    ldc);

  Gemm::launch(params);

  cudaError_t result = cudaDeviceSynchronize();
  ASSERT_EQ(result, cudaSuccess) << "\nCUDA kernel launch error: " << cudaGetErrorString(result)
                                 << "\n";

  /* if (testbed.has_cublas_support()) { */
  /*   ASSERT_TRUE(testbed.verify_with_cublas()); */
  /* } else { */
  /*   ASSERT_TRUE(testbed.verify_with_host()); */
  /* } */
  print_memory(ptr_computed, m*n);

  cudaFree(ptr_A);
  cudaFree(ptr_B);
  cudaFree(ptr_C_initial);
  cudaFree(ptr_computed);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename GemmTraits_>
static void run_gemm_test_int_to_float2(
    int m,
    int n,
    int k,
    //    typename test::GemmTestbedTraits<typename GemmTraits_::Epilogue::Scalar>::host_type alpha =
    //    typename test::GemmTestbedTraits<typename GemmTraits_::Epilogue::Scalar>::host_type(1),
    int alpha = int(1),
    //typename test::GemmTestbedTraits<typename GemmTraits_::Epilogue::Scalar>::host_type beta =
    //    typename test::GemmTestbedTraits<typename GemmTraits_::Epilogue::Scalar>::host_type(0)) {
    int beta = int(0)) {
  int lda = GemmTraits_::kLayoutA == cutlass::MatrixLayout::kColumnMajor ? m : k;
  int ldb = GemmTraits_::kLayoutB == cutlass::MatrixLayout::kColumnMajor ? k : n;

  run_gemm_test_int_to_float2<GemmTraits_>(m, n, k, lda, ldb, m, alpha, beta);
}

////////////////////////////////////////////////////////////////////////////////////////////////////


/***************************************************************************************************
 * Copyright (c) 2017, NVIDIA CORPORATION.  All rights reserved.
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
/*! \file
    \brief Defines abstractions for efficiently loading and storing vectors to memory.
 */
#pragma once

#include <cutlass/vector.h>

namespace cutlass {

////////////////////////////////////////////////////////////////////////////////////////////////////

/**
* @brief Enum to specify which memory space data resides in.
*/
struct MemorySpace {
  enum Kind {
    kGeneric,  // Data accessed through pointer dereferencing
    kShared,   // Data resides in shared memory
    kGlobal    // Data resides in global memory
  };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Scalar_,
          int Lanes_,
          MemorySpace::Kind Memory_,
          bool = (Lanes_ > 1),
          size_t = (sizeof(Scalar_) * Lanes_)>
struct Load {
  /// The output type.
  typedef typename Vectorize<Scalar_, Lanes_>::Type AccessType;

  /// The load function.
  static CUTLASS_DEVICE void load(AccessType& dst, Scalar_ const* pointer, int offset) {
    dst = reinterpret_cast<AccessType const*>(&pointer[offset])[0];
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Scalar_, int Lanes_, MemorySpace::Kind Memory_>
struct Load<Scalar_, Lanes_, Memory_, true, 4> {
  /// The output type.
  typedef typename Vectorize<Scalar_, Lanes_>::Type AccessType;

  /// The store function.
  static CUTLASS_DEVICE void load(AccessType& dst, Scalar_ const* pointer, int offset) {
    dst.registers[0] = reinterpret_cast<uint32_t const*>(&pointer[offset])[0];
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Scalar_, int Lanes_, MemorySpace::Kind Memory_>
struct Load<Scalar_, Lanes_, Memory_, true, 8> {
  /// The output type.
  typedef typename Vectorize<Scalar_, Lanes_>::Type AccessType;

  /// The store function.
  static CUTLASS_DEVICE void load(AccessType& dst, Scalar_ const* pointer, int offset) {
    uint2 tmp = reinterpret_cast<uint2 const*>(&pointer[offset])[0];
    dst.registers[0] = tmp.x;
    dst.registers[1] = tmp.y;
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <MemorySpace::Kind Memory_>
struct Load<double, 2, Memory_, true, 16> {
  /// The output type.
  typedef typename Vectorize<double, 2>::Type AccessType;

  /// The store function.
  static CUTLASS_DEVICE void load(AccessType& dst, double const* pointer, int offset) {
    double2 tmp = reinterpret_cast<double2 const*>(&pointer[offset])[0];
    dst[0] = tmp.x;
    dst[1] = tmp.y;
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(__CUDACC_VERSION_MAJOR) && __CUDACC_VERSION_MAJOR < 10
// WAR bug in NVCC where the upper and lower half of the register end up being the same
template <MemorySpace::Kind Memory_>
struct Load<half, 8, Memory_, true, 16> {
  /// The output type.
  typedef typename Vectorize<half, 8>::Type AccessType;

  /// The store function.
  static CUTLASS_DEVICE void load(AccessType& dst, half const* pointer, int offset) {
    int2 tmp = reinterpret_cast<int2 const*>(&pointer[offset])[0];
    dst.registers[0] = tmp.x;
    dst.registers[1] = tmp.y;

    tmp = reinterpret_cast<int2 const*>(&pointer[offset + 4])[0];
    dst.registers[2] = tmp.x;
    dst.registers[3] = tmp.y;
  }
};

#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Scalar_, int Lanes_, MemorySpace::Kind Memory_>
struct Load<Scalar_, Lanes_, Memory_, true, 16> {
  /// The output type.
  typedef typename Vectorize<Scalar_, Lanes_>::Type AccessType;

  /// The store function.
  static CUTLASS_DEVICE void load(AccessType& dst, Scalar_ const* pointer, int offset) {
    uint4 tmp = reinterpret_cast<uint4 const*>(&pointer[offset])[0];
    dst.registers[0] = tmp.x;
    dst.registers[1] = tmp.y;
    dst.registers[2] = tmp.z;
    dst.registers[3] = tmp.w;
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Scalar_,
          int Lanes_,
          MemorySpace::Kind Memory_,
          bool = (Lanes_ > 1),
          size_t = (sizeof(Scalar_) * Lanes_)>
struct Store {
  /// The output type.
  typedef typename Vectorize<Scalar_, Lanes_>::Type AccessType;

  /// The store function.
  static CUTLASS_DEVICE void store(AccessType const& src, Scalar_* pointer, int offset) {
    extern const float *start_pos;
    extern int COLUMN_NUM;
    extern int ROW_NUM;
    extern const float *d_scalerA;
    extern const float *d_scalerB;

    if(d_scalerA != NULL)
    {
        float *add = reinterpret_cast<float*>(&pointer[offset]);
        int off = (add - start_pos);
        int row = off / COLUMN_NUM;
        int col = off % COLUMN_NUM;
        pointer[offset] = src * __ldg(d_scalerA + row) * __ldg(d_scalerB + col);
    }
    else
        pointer[offset] = src;
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Scalar_, int Lanes_, MemorySpace::Kind Memory_>
struct Store<Scalar_, Lanes_, Memory_, true, 4> {
  /// The output type.
  typedef typename Vectorize<Scalar_, Lanes_>::Type AccessType;

  /// The store function.
  static CUTLASS_DEVICE void store(AccessType const& src, Scalar_* pointer, int offset) {
    uint32_t* addr = reinterpret_cast<uint32_t*>(&pointer[offset]);
    addr[0] = src.registers[0];
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Scalar_, int Lanes_, MemorySpace::Kind Memory_>
struct Store<Scalar_, Lanes_, Memory_, true, 8> {
  /// The output type.
  typedef typename Vectorize<Scalar_, Lanes_>::Type AccessType;

  /// The store function.
  static CUTLASS_DEVICE void store(AccessType const& src, Scalar_* pointer, int offset) {
    uint2* addr = reinterpret_cast<uint2*>(&pointer[offset]);
    addr[0] = make_uint2(src.registers[0], src.registers[1]);
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <MemorySpace::Kind Memory_>
struct Store<double, 2, Memory_, true, 16> {
  /// The output type.
  typedef typename Vectorize<double, 2>::Type AccessType;

  /// The store function.
  static CUTLASS_DEVICE void store(AccessType const& src, double* pointer, int offset) {
    double2* addr = reinterpret_cast<double2*>(&pointer[offset]);
    addr[0] = make_double2(src[0], src[1]);
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Scalar_, int Lanes_, MemorySpace::Kind Memory_>
struct Store<Scalar_, Lanes_, Memory_, true, 16> {
  /// The output type.
  typedef typename Vectorize<Scalar_, Lanes_>::Type AccessType;

  /// The store function.
  static CUTLASS_DEVICE void store(AccessType const& src, Scalar_* pointer, int offset) {
    uint4* addr = reinterpret_cast<uint4*>(&pointer[offset]);
    addr[0] = make_uint4(src.registers[0], src.registers[1], src.registers[2], src.registers[3]);
  }
};


////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace cutlass

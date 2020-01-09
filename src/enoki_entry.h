#pragma once

#pragma warning(push, 0)
#define _USE_MATH_DEFINES // since enoki requires M_PI
#include <cstdint>
#include "math.h"
#include <enoki/array.h>
#include <enoki/autodiff.h>
#include <enoki/cuda.h>
#include <enoki/transform.h>
#include <enoki/matrix.h>
#include <enoki/random.h>
#include <enoki/sh.h>
#include <enoki/stl.h>
#pragma warning(pop)

const size_t CudaPacketSize = 100;

// math and vectors types
using Int8 = int8_t;
using Uint8 = uint8_t;
using Int16 = int16_t;
using Uint16 = uint16_t;
using Int32 = int32_t;
using Uint32 = uint32_t;
using Int64 = int64_t;
using Uint64 = uint64_t;

using namespace enoki;

#ifdef USE_DOUBLE_PRECISION
using Real = double;
#else
using Real = float;
#endif
using Int = int;
using Uint = unsigned int;

using RealP = Packet<Real, 4>;
using RealC = CUDAArray<Real>;
using RealPC = Packet<RealC, CudaPacketSize>;
using IntC = CUDAArray<Int>;
using IntPC = Packet<IntC, CudaPacketSize>;
using UintC = CUDAArray<Uint>;
using BoolC = CUDAArray<bool>;

// neat trick from https://github.com/yuanming-hu/taichi
// in which he learned from https://github.com/hi2p-perim/lightmetrica-v2)
constexpr Real operator"" _f(long double v)
{
	return Real(v);
}

constexpr Real operator"" _f(unsigned long long v)
{
	return Real(v);
}

using Real2 = Array<Real, 2, false>;
using Real3 = Array<Real, 3, false>;
using Real4 = Array<Real, 4, false>;
using Int2 = Array<Int, 2, false>;
using Int3 = Array<Int, 3, false>;
using Int4 = Array<Int, 4, false>;
using Uint2 = Array<Uint, 2, false>;
using Uint3 = Array<Uint, 3, false>;
using Uint4 = Array<Uint, 4, false>;
using Mat4 = Matrix<Real, 4, false>;

using Real2P = Array<RealP, 2, false>;
using Real3P = Array<RealP, 3, false>;
using Real4P = Array<RealP, 4, false>;

using Real2C = Array<RealC, 2, false>;
using Real3C = Array<RealC, 3, false>;
using Real4C = Array<RealC, 4, false>;
using Int2C = Array<IntC, 2, false>;
using Int3C = Array<IntC, 3, false>;

/*
using Ivec2P = Array<Packet<int>, 2, false>;
using Ivec3P = Array<Packet<int>, 3, false>;
using Ivec4P = Array<Packet<int>, 4, false>;
using Lvec2P = Array<Packet<int64>, 2, false>;
using Lvec3P = Array<Packet<int64>, 3, false>;
using Lvec4P = Array<Packet<int64>, 4, false>;
using Lvec5P = Array<Packet<int64>, 5, false>;
using Uvec2P = Array<Packet<unsigned int>, 2, false>;
using Uvec3P = Array<Packet<unsigned int>, 3, false>;
using Uvec4P = Array<Packet<unsigned int>, 4, false>;
using Mat4P = Matrix<Packet<Real>, 4, false>;
*/

// note: changing this does not make the renderer automatically switch from right to left hand.
const Real3 UP = Real3(0.0_f, 1.0_f, 0.0_f);

const Real M_PI_f = Real(M_PI);
const Real M_1_PI_f = Real(M_1_PI);

#ifdef USE_DOUBLE_PRECISION
const Real SmallValue = 1e-12;
#else
const Real SmallValue = 1e-4_f;
#endif

template <typename Value_, size_t Size_, bool Approx_, RoundingMode Mode_>
struct SpectrumT : enoki::StaticArrayImpl<Value_, Size_, Approx_, Mode_, false, SpectrumT<Value_, Size_, Approx_, Mode_>>
{
    using Base = enoki::StaticArrayImpl<Value_, Size_, Approx_, Mode_, false, SpectrumT<Value_, Size_, Approx_, Mode_>>;
    using ArrayType = SpectrumT;
    using MaskType = Mask<Value_, Size_, Approx_, Mode_>;
    template <typename T>
    using ReplaceValue = SpectrumT<T, Size_,
        is_std_float_v<scalar_t<T>> && is_std_float_v<scalar_t<Value_>>
        ? Approx_ : array_approx_v<scalar_t<T>>,
        is_std_float_v<scalar_t<T>> && is_std_float_v<scalar_t<Value_>>
        ? Mode_ : RoundingMode::Default>;
    ENOKI_ARRAY_IMPORT(Base, SpectrumT)
};
using SpectrumC = SpectrumT<RealC, 3, false, RoundingMode::Default>;

#pragma once

#include "enoki_entry.h"

template <typename Real_>
struct Frame3
{
	using RealT		= Real_;
	using Real3T	= Array<RealT, 3, false>;
	using Mat3T		= Matrix<RealT, 3, false>;
	using MaskT		= mask_t<RealT>;

	Frame3(): m_world_from_local(empty<Mat3T>())
	{
	}

	Frame3(const Real3T & normal)
	{
		const RealT sign = copysign(1.0_f, normal.y());
		const RealT a = -1.0_f / (sign + normal.y());
		const RealT b = normal.z() + normal.x() * a;
		const Real3T basis_x = Real3T(sign + normal[0] * normal[0] * a, -normal[0], b);
		const Real3T basis_y = normal;
		const Real3T basis_z = Real3T(sign * b, -sign * normal[2], 1.0_f + sign * normal[2] * normal[2] * a);

		m_world_from_local = Mat3T(basis_x.x(), basis_y.x(), basis_z.x(),
								   basis_x.y(), basis_y.y(), basis_z.y(),
								   basis_x.z(), basis_y.z(), basis_z.z());
	}

	Frame3(const Real3T & basis_x, const Real3T & basis_y, const Real3T & basis_z)
	{
		m_world_from_local = Mat3T(basis_x.x(), basis_y.x(), basis_z.x(),
								   basis_x.y(), basis_y.y(), basis_z.y(),
								   basis_x.z(), basis_y.z(), basis_z.z());
	}

	template <typename Real3TT>
	Real3TT to_local(const Real3TT & world) const
	{
		return transpose(m_world_from_local) * world;
	}

	template <typename Real3TT>
	Real3TT to_world(const Real3TT & local) const
	{
		return m_world_from_local * local;
	}

	Mat3T m_world_from_local;
};

using Frame3C = Frame3<RealC>;
using Frame3S = Frame3<Real>;
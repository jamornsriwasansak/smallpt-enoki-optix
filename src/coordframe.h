#include "enoki_entry.h"

template <typename Real_>
struct CoordFrame3
{
	using RealT		= Real_;
	using Real3T	= Array<RealT, 3, false>;
	using Mat3T		= Matrix<RealT, 3, false>;

	CoordFrame3(const Real3T & normal)
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

	Real3T to_local(const Real3T & world) const
	{
		return transpose(m_world_from_local) * world;
	}

	Real3T to_world(const Real3T & local) const
	{
		return m_world_from_local * local;
	}

	Mat3T m_world_from_local;
};

using CoordFrame3C = CoordFrame3<RealC>;
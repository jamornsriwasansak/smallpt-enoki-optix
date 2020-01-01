#include "enoki_entry.h"

template<typename T, typename Real2T>
inline T barycentric_interpolate(const T & a, const T & b, const T & c, const Real2T & bc)
{
	return (1.0_f - bc[0] - bc[1]) * a + bc[0] * b + bc[1] * c;
}

template<typename Real3T>
inline Real3T compute_geometry_normal(const Real3T & p1, const Real3T & p2, const Real3T & p3)
{
	const Real3T e1 = p2 - p1;
	const Real3T e2 = p3 - p1;
	const Real3T un = cross(e1, e2);
	auto len = enoki::norm(un);
	return un / len;
}
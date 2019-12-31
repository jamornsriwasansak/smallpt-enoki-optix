#include "enoki_entry.h"

struct LambertBsdf
{
	LambertBsdf(const Vec3 & m_reflectance)
	{
	}

	std::tie<Vec3, Vec3> sample(const RealC & sample0, const RealC & sample1)
	{
		sample0;
	}


	Vec3 m_reflectance;
};
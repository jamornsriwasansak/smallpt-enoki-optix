#include "enoki_entry.h"

inline Real3C cosine_weighted_hemisphere_from_square(const RealC & sample0, const RealC & sample1)
{
	const RealC sin_phi = sqrt(1.0_f - sample0);
	const RealC theta = M_PI_f * 2.0_f * sample1;
	auto [sin_theta, cos_theta] = sincos(theta);
	return Real3C(cos_theta * sin_phi, sqrt(sample0), sin_theta * sin_phi);
}
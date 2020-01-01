#pragma once

#include "enoki_entry.h"

inline Real3C cosine_weighted_hemisphere_from_square(const Real2C & sample)
{
	const RealC sin_phi = sqrt(1.0_f - sample.x());
	const RealC theta = M_PI_f * 2.0_f * sample.y();
	auto [sin_theta, cos_theta] = sincos(theta);
	return Real3C(cos_theta * sin_phi, sqrt(sample.x()), sin_theta * sin_phi);
}

// taken from Dave Cline from Peter Shirley from http://psgraphics.blogspot.jp/2011/01/improved-code-for-concentric-map.html
// and https://github.com/mmp/pbrt-v3/blob/7095acfd8331cdefb03fe0bcae32c2fc9bd95980/src/core/sampling.cpp
inline Real2C disk_from_square(const Real2C & sample)
{
	const Real2C ab = sample * 2.0_f - 1.0_f;
	const Real2C ab2 = sqr(ab);
	const RealC r = select(ab2.x() > ab2.y(),
						   ab.x(),
						   ab.y());
	const RealC phi = select(ab2.x() > ab2.y(),
							 (M_PI_f / 4.0_f) * (ab.y() / ab.x()),
							 (M_PI_f / 2.0_f) - (M_PI_f / 4.0_f) * (ab.x() / ab.y()));
	auto [sin_phi, cos_phi] = sincos(phi);
	return r * Real2C(cos_phi, sin_phi);
}

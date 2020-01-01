#pragma once

#include "enoki_entry.h"

struct ThinlensCamera
{
	ThinlensCamera(const Real3 & look_from, const Real3 & look_at, const Real3 & up,
				   //const Real lens_radius, const Real focal_dist,
				   const Real fov_y,
				   const Real film_size_y = 0.035_f):
		m_film_size_y(film_size_y),

		/*
		m_focal_dist(focal_dist),
		m_lens_radius(lens_radius),
		m_lens_area(M_PI * lens_radius * lens_radius),*/
		m_fov_y(fov_y)
	{
		m_origin = look_from;

		m_z = normalize(look_at - look_from);
		m_x = normalize(cross(up, m_z));
		m_y = normalize(cross(m_z, m_x));
	}

	Real3C sample_pos(const RealC & sample0, const RealC & sample1) const
	{

	}

	Real3C sample_dir(const Int2C & pixel, const Int2 & image_resolution, const RealC & sample0, const RealC & sample1) const
	{
		Real3C position(sample0 * 0, sample0 * 0, sample0 * 0);
		position += m_origin;

		// compute distance from film to camera
		const Real distance_lens_to_film = m_film_size_y * 0.5_f / tan(m_fov_y * 0.5_f);
		const Real ratio_x_to_y = Real(image_resolution.x()) / Real(image_resolution.y());
		const Real2 half_film_size(ratio_x_to_y * m_film_size_y * 0.5_f, m_film_size_y * 0.5_f);

		// sample position on the film
		const Real2C ndc = (Real2C(pixel) + Real2C(sample0, sample1)) / Real2(image_resolution);
		const Real2C film_position_xy = (Real2(0.5_f) - ndc) * half_film_size;
		const Real3C film_position(film_position_xy.x(), film_position_xy.y(), distance_lens_to_film);

		// TODO:: transform film_position before exit
		Matrix<float, 3, false> test_matrix(m_x.x(), m_y.x(), m_z.x(), m_x.y(), m_y.y(), m_z.y(), m_x.z(), m_y.z(), m_z.z());
		const Real3C direction = normalize(film_position);
		return test_matrix * direction;
	}

	// film params
	Real	m_film_size_y;

	Real3	m_origin;
	Real3	m_x;
	Real3	m_y;
	Real3	m_z;

	Real	m_focal_dist;
	Real	m_lens_radius;
	Real	m_lens_area;
	Real	m_fov_y;
};

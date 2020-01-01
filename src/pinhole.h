#pragma once

#include "enoki_entry.h"
#include "mapping.h"
#include "coordframe.h"

struct ThinlensCamera
{
	ThinlensCamera(const Real3 & look_from, const Real3 & look_at, const Real3 & up,
				   const Real lens_radius, const Real focal_dist, const Real fov_y,
				   const Real film_size_y = 0.035_f):
		m_film_size_y(film_size_y),
		m_focal_dist(focal_dist),
		m_fov_y(fov_y),
		m_lens_area(M_PI_f * lens_radius * lens_radius),
		m_lens_radius(lens_radius)
	{
		m_origin = look_from;

		m_z = normalize(look_at - look_from);
		m_x = normalize(cross(up, m_z));
		m_y = normalize(cross(m_z, m_x));
		
		m_frame = Frame3S(m_x, m_y, m_z);
	}

	Real3C sample_pos(const Real2C & sample) const
	{
		const Real2C lens_position_xy = disk_from_square(sample) * m_lens_radius;
		const Real3C lens_position = Real3C(lens_position_xy.x(), 0.0_f, lens_position_xy.y());
		return m_frame.to_world(lens_position) + m_origin;
	}

	Real3C sample_dir(const Int2C & pixel, const Real3C & position, const Int2 & image_resolution, const Real2C & sample) const
	{
		// compute lens position
		const Real3C lens_position = m_frame.to_local(position - m_origin);

		// compute distance from film to camera
		const Real distance_lens_to_film = m_film_size_y * 0.5_f / tan(m_fov_y * 0.5_f);
		const Real ratio_x_to_y = Real(image_resolution.x()) / Real(image_resolution.y());
		const Real2 half_film_size(ratio_x_to_y * m_film_size_y * 0.5_f, m_film_size_y * 0.5_f);

		// sample position on the film
		const Real2C ndc = (Real2C(pixel) + Real2C(sample.x(), sample.y())) / Real2(image_resolution);
		const Real2C film_position_xy = (Real2(0.5_f) - ndc) * half_film_size;
		const Real3C film_position(film_position_xy.x(), film_position_xy.y(), distance_lens_to_film);

		// scale the film to focal distance
		const Real3C focal_dist_position = m_focal_dist * film_position / film_position.z();

		// compute direction
		const Real3C direction = normalize(focal_dist_position - lens_position);

		return m_frame.to_world(direction);
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

	Frame3S	m_frame;
};

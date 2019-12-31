template<typename T, typename Real2T>
inline T barycentric_interpolate(const T & a, const T & b, const T & c, const Real2T & bc)
{
	return (1.0_f - bc[0] - bc[1]) * a + bc[0] * b + bc[1] * c;
}
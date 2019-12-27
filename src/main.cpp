#include <iostream>
#include "enoki/cuda.h"
#include "enoki/array.h"
#include "enoki/autodiff.h"
#include "enoki/dynamic.h"
#include "enoki/stl.h"

template <typename Value> Value srgb_gamma(Value x) {
	return enoki::select(
		x <= 0.0031308f,
		x * 12.92f,
		enoki::pow(x * 1.055f, 1.f / 2.4f) - 0.055f
	);
}

int main()
{
	using FloatC = enoki::CUDAArray<float>;
	using FloatD = enoki::DiffArray<FloatC>;
	using Color3fD = enoki::Array<FloatD, 3>;

	Color3fD input = Color3fD(0.5f, 1.0f, 2.0f);
	enoki::set_requires_gradient(input);

	Color3fD output = srgb_gamma(input);

	std::cout << "test1" << std::endl;
	FloatD loss = enoki::norm(output - Color3fD(.1f, .2f, .3f));
	enoki::backward(loss);
	std::cout << "test2" << std::endl;
	std::cout << enoki::gradient(input) << std::endl;
	std::cout << "test3" << std::endl;
	return 0;
}
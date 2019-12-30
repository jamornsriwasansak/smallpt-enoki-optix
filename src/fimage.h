#include "enoki_entry.h"
#include <filesystem>
#include <fstream>

struct Fimage
{
	static void save_pfm(const float * image, const size_t width, const size_t height, const std::filesystem::path & filepath)
	{
		// write PFM header
		std::ofstream os(filepath, std::ios::binary);
		if (!os.is_open())
		{
			std::cout << "wtf" << std::endl;
		}
		os << "PF" << std::endl;
		os << width << " " << height << std::endl;
		os << "-1" << std::endl;

		// write data flip row
		for (int y = height; y >= 1; y--)
			for (int x = 0; x < width; x++)
			{
				for (int c = 0; c < 3; c++)
				{
					float vf = image[y * width + x];
					os.write((char *)(&vf), sizeof(float));
				}
			}

		os.close();
	}
};

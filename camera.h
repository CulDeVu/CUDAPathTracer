#pragma once

#include "color.h"

struct ray
{
	vec3 o;
	vec3 dir;
};

struct camera
{
	vec3 pos;

	float distFromFilm;
	float focalLength;
	float radius;

	int pxlWidth, pxlHeight;

	__host__ __device__ vec3 pxlToFilm(uint16_t x, uint16_t y)
	{
		vec3 ret;
		ret.x = (float)x / pxlWidth - 0.5f;
		ret.y = (float)y / pxlHeight - 0.5f;
		ret.z = 0;
		return ret;
	}

	// scanline indexing
	__host__ __device__ void scanlineItoPxl(uint16_t* x, uint16_t* y, uint32_t index)
	{
		*y = index / pxlWidth;
		*x = index % pxlWidth;
	}
	__host__ __device__ int scanlinePxltoI(uint16_t x, uint16_t y)
	{
		return y * pxlWidth + x;
	}

	// morton z-curve indexing
	__host__ __device__ void mortonItoPxl(uint16_t* x, uint16_t* y, uint32_t index)
	{
		*x = *y = 0;
		for (int i = 0; i < 16; ++i)
		{
			*x |= ((index >> (2 * i + 0)) & 0x1) << i;
			*y |= ((index >> (2 * i + 1)) & 0x1) << i;
		}
	}
	__host__ __device__ uint32_t mortonPxltoI(uint16_t x, uint16_t y)
	{
		uint32_t index = 0;
		for (int i = 0; i < 16; ++i)
		{
			index |= ((x >> i) & 0x1) << (2 * i + 0);
			index |= ((y >> i) & 0x1) << (2 * i + 1);
		}
		return index;
	}

	__device__ ray cameraRay(int idx, float u1, float u2)
	{
		vec3 posRelFilm;
		{
			uint16_t x, y;
			mortonItoPxl(&x, &y, idx);
			posRelFilm = pxlToFilm(x, y);
		}

		float r = radius * sqrt(u1);
		float theta = 2 * 3.14159 * u2;
		vec3 o = vec3(r * cos(theta), r * sin(theta), 0);

		posRelFilm.z = distFromFilm;
		posRelFilm = posRelFilm * -focalLength / distFromFilm;
		ray ret;
		ret.o = o + pos;
		ret.dir = normalized(posRelFilm - o);

		return ret;
	}
};

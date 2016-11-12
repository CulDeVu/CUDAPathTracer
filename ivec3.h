#pragma once

struct ivec3
{
	int x, y, z;

	__host__ __device__ ivec3(int a = 0, int b = 0, int c = 0)
	{
		x = a;
		y = b;
		z = c;
	}

	__host__ __device__ ivec3 operator+(const ivec3& a)
	{
		return ivec3(x + a.x, y + a.y, z + a.z);
	}
	__host__ __device__ ivec3 operator-(const ivec3& a)
	{
		return ivec3(x - a.x, y - a.y, z - a.z);
	}
	__host__ __device__ ivec3 operator*(const float& a)
	{
		return ivec3(x * a, y * a, z * a);
	}
	__host__ __device__ ivec3 operator/(const float& a)
	{
		return ivec3(x / a, y / a, z / a);
	}
	__host__ __device__ ivec3 operator/(const int& a)
	{
		return ivec3(x / a, y / a, z / a);
	}

	__host__ __device__ int operator[](int i) const
	{
		if (i == 0)
			return x;
		if (i == 1)
			return y;
		if (i == 2)
			return z;
		return 0;
	}
	__host__ __device__ int& operator[](int i)
	{
		if (i == 0)
			return x;
		if (i == 1)
			return y;
		if (i == 2)
			return z;
		return x;
	}
};

__host__ __device__ int dot(ivec3 a, ivec3 b)
{
	return a.x*b.x + a.y*b.y + a.z*b.z;
}

ivec3 min(ivec3 a, ivec3 b)
{
	return ivec3(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z));
}
ivec3 max(ivec3 a, ivec3 b)
{
	return ivec3(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z));
}

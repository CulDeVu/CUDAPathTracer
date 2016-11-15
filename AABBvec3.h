#pragma once

struct AABBvec3
{
	float x, y, z;

	__host__ __device__ AABBvec3()
	{
	}

	__host__ __device__ AABBvec3(const vec3& a)
	{
		x = a.x;
		y = a.y;
		z = a.z;
	}

	__host__ __device__ AABBvec3(float a, float b, float c)
	{
		x = a;
		y = b;
		z = c;
	}

	__host__ __device__ AABBvec3 operator+(const AABBvec3& a)
	{
		return AABBvec3(x + a.x, y + a.y, z + a.z);
	}
	__host__ __device__ AABBvec3 operator-(const AABBvec3& a)
	{
		return AABBvec3(x - a.x, y - a.y, z - a.z);
	}
	__host__ __device__ AABBvec3 operator*(const float& a)
	{
		return AABBvec3(x * a, y * a, z * a);
	}
	__host__ __device__ AABBvec3 operator/(const float& a)
	{
		return AABBvec3(x / a, y / a, z / a);
	}
	__host__ __device__ AABBvec3 operator/(const int& a)
	{
		return AABBvec3(x / (float)a, y / (float)a, z / (float)a);
	}
	__host__ __device__ AABBvec3& operator=(const vec3& a)
	{
		x = a.x;
		y = a.y;
		z = a.z;
		return *this;
	}

	vec3 toVec3()
	{
		return vec3(x, y, z);
	}

	__host__ __device__ float operator[](int i) const
	{
		if (i == 0)
			return x;
		if (i == 1)
			return y;
		if (i == 2)
			return z;
		return 0;
	}
	__host__ __device__ float& operator[](int i)
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

__host__ __device__ AABBvec3 normalized(AABBvec3 v)
{
	float len = sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
	return AABBvec3(v.x / len, v.y / len, v.z / len);
}
__host__ __device__ float dot(AABBvec3 a, AABBvec3 b)
{
	return a.x*b.x + a.y*b.y + a.z*b.z;
}
__host__ __device__ AABBvec3 cross(AABBvec3 v1, AABBvec3 v2)
{
	return AABBvec3(v1.y*v2.z - v1.z*v2.y, v1.z*v2.x - v1.x*v2.z, v1.x*v2.y - v1.y*v2.x);
}

__host__ __device__ float length(AABBvec3 v)
{
	return sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
}

AABBvec3 min(AABBvec3 a, AABBvec3 b)
{
	return AABBvec3(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z));
}
AABBvec3 max(AABBvec3 a, AABBvec3 b)
{
	return AABBvec3(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z));
}

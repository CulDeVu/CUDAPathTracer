#ifndef VEC3_H
#define VEC3_H

struct vec3
{
	float x, y, z;

	__host__ __device__ vec3(float a = 0, float b = 0, float c = 0)
	{
		x = a;
		y = b;
		z = c;
	}

	__host__ __device__ vec3 operator+(const vec3& a)
	{
		return vec3(x + a.x, y + a.y, z + a.z);
	}
	__host__ __device__ vec3 operator-(const vec3& a)
	{
		return vec3(x - a.x, y - a.y, z - a.z);
	}
	__host__ __device__ vec3 operator*(const float& a)
	{
		return vec3(x * a, y * a, z * a);
	}
	__host__ __device__ vec3 operator/(const float& a)
	{
		return vec3(x / a, y / a, z / a);
	}
};

__host__ __device__ vec3 normalized(vec3 v)
{
	float len = sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
	return vec3(v.x / len, v.y / len, v.z / len);
}
__host__ __device__ float dot(vec3 a, vec3 b)
{
	return a.x*b.x + a.y*b.y + a.z*b.z;
}
__host__ __device__ vec3 cross(vec3 v1, vec3 v2)
{
	return vec3(v1.y*v2.z - v1.z*v2.y, v1.z*v2.x - v1.x*v2.z, v1.x*v2.y - v1.y*v2.x);
}

__host__ __device__ float length(vec3 v)
{
	return sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
}

#endif
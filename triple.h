#pragma once

#include "SIUnits.h"

#define __global __host__ __device__

template<int m, int kg, int s, int sr>
struct siTriple
{
	#define curUnits siUnits<m, kg, s, sr>;
	#define curTriple siTriple<m, kg, s, sr>;

	siUnits<m,kg,s,sr> x, y, z;

	__global siTriple()
		: x(0.0f), y(0.0f), z(0.0f)
	{}

	__global siTriple(
		siUnits<m,kg,s,sr> a, 
		siUnits<m,kg,s,sr> b,
		siUnits<m,kg,s,sr> c)
		: x(a), y(b), z(c)
	{}

	// arithmatic operators
	__global siTriple<m,kg,s,sr> operator+(siTriple<m,kg,s,sr> a)
	{
		return curTriple(x + a.x, y + a.y, z + a.z);
	}
	__global siTriple<m,kg,s,sr> operator-(siTriple<m,kg,s,sr> a)
	{
		return curTriple(x - a.x, y - a.y, z - a.z);
	}

	template<int a, int b, int c, int d>
	__global siTriple<m+a, kg+b, s+c, sr+d> operator*(siUnits<a,b,c,d> a)
	{
		return siTriple<m+a, kg+b, s+c, sr+d>(x * a, y * a, z * a);
	}
	template<int a, int b, int c, int d>
	__global siTriple<m-a, kg-b, s-c, sr-d> operator/(siUnits<a,b,c,d> a)
	{
		return siTriple<m-a, kg-b, s-c, sr-d>(x / a, y / a, z / a);
	}

	// index access operators
	__global siUnits<m,kg,s,sr> operator[](int i) const
	{
		if (i == 0)
			return x;
		if (i == 1)
			return y;
		if (i == 2)
			return z;
		return 0;
	}
	__global siUnits<m,kg,s,sr>& operator[](int i)
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

template<int m, int kg, int s, int sr>
__global siTriple<m,kg,s,sr> normalized(siTriple<m,kg,s,sr> v)
{
	siUnits<m,kg,s,sr> len = sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
	return siTriple(v.x / len, v.y / len, v.z / len);
}
template<typename T>
__global T dot(triple<T> a, triple<T> b)
{
	return a.x*b.x + a.y*b.y + a.z*b.z;
}
template<typename T>
__global triple<T> cross(triple<T> v1, triple<T> v2)
{
	return triple(v1.y*v2.z - v1.z*v2.y, v1.z*v2.x - v1.x*v2.z, v1.x*v2.y - v1.y*v2.x);
}

template<int m, int kg, int s, int sr>
__global siUnits<m.kg,s,sr> length(siTriple<m,kg,s,sr> v)
{
	return sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
}

template<typename T>
triple<T> min(triple<T> a, triple<T> b)
{
	return triple(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z));
}
template<typename T>
triple<T> max(triple<T> a, triple<T> b)
{
	return triple(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z));
}

#define vec3 triple<1,0,0,0>
#define color triple<float>

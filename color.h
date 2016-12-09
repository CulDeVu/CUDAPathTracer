#ifndef COLOR_H
#define COLOR_H

struct color
{
	double r, g, b;

	__host__ __device__ color(double x, double y, double z)
	{
		r = x;
		g = y;
		b = z;
	}

	__host__ __device__ color()
		: color(0, 0, 0)
	{}

	__host__ __device__ color operator+(const color& a)
	{
		return color(r + a.r, g + a.g, b + a.b);
	}
	__host__ __device__ color operator-(const color& a)
	{
		return color(r - a.r, g - a.g, b - a.b);
	}
	__host__ __device__ color operator*(const float& a)
	{
		return color(r * a, g * a, b * a);
	}
	__host__ __device__ color operator/(const float& a)
	{
		return color(r / a, g / a, b / a);
	}
	__host__ __device__ color operator/(const int& a)
	{
		return color(r / (float)a, g / (float)a, b / (float)a);
	}

	__host__ __device__ color operator*(const color& a)
	{
		return color(r * a.r, g * a.g, b * a.b);
	}
};

__device__ color add(color a, color b)
{
	return color(a.r + b.r, a.g + b.g, a.b + b.b);
}
__device__ color mul(color c, float f)
{
	return color(c.r * f, c.g * f, c.b * f);
}

__device__ color mul(color c, color d)
{
	return color(c.r * d.r, c.g * d.g, c.b * d.b);
}
__host__ __device__ color normalized(color c)
{
	return color(c.r / (c.r + 1), c.g / (c.g + 1), c.b / (c.b + 1));
}
__device__ color denormalized(color c)
{
	return color(c.r / (1 - c.r), c.g / (1 - c.g), c.b / (1 - c.b));
}

color gammaCorrect(color c, float a)
{
	return color(pow(c.r, a), pow(c.g, a), pow(c.b, a));
}


#endif
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
__device__ color normalized(color c)
{
	return color(c.r / (c.r + 1), c.g / (c.g + 1), c.b / (c.b + 1));
}
__device__ color denormalized(color c)
{
	return color(c.r / (1 - c.r), c.g / (1 - c.g), c.b / (1 - c.b));
}

#endif
#pragma once

#define _global __host__ __device__

struct allUnits
{
	int len;
	int mass;
	int time;
	int sr;

	allUnits(int a, int b, int c, int d)
		: len(a), mass(b), time(c), sr(d)
	{}

	allUnits operator+(allUnits a)
	{
		return allUnits(len + a.len, mass + a.mass, time + a.time, sr + a.sr);
	}

	allUnits operator*(allUnits a)
	{
		return allUnits(len * a.len, mass * a.mass, time * a.time, sr * a.sr);
	}
};

template<int len, int mass, int time, int sr>
class siUnits
{
public:
	float raw;

	_global siUnits(float n)
	{
		raw = n;
	}

	siUnits<len, mass, time, sr>
	operator+(siUnits<len, mass, time, sr> other)
	{
		siUnits<len, mass, time, sr> ret(raw + other.raw);
		return ret;
	}

	template<int l, int m, int t, int s> 
	siUnits<len + l, mass + m, time + t, sr + s>
	operator*(siUnits<l, m, t, s> other)
	{
		siUnits<len + l, mass + m, time + t, sr + s> ret(raw * other.raw);
		return ret;
	}

	template<int l, int m, int t, int s> 
	siUnits<len - l, mass - m, time - t, sr - s>
	operator/(siUnits<l, m, t, s> other)
	{
		siUnits<len - l, mass - m, time - t, sr - s> ret(raw / other.raw);
		return ret;
	}

	siUnits<len, mass, time, sr>
	operator*(float a)
	{
		siUnits<len, mass, time, sr> ret(raw * a);
		return ret;
	}
};

#define siScalar siUnits<0,0,0,0>

#define siLength   siUnits<1,0,0,0>
#define siArea     siUnits<2,0,0,0>
#define siVolume   siUnits<3,0,0,0>

#define siMass siUnits<0,1,0,0>

#define siTime siUnits<0,0,1,0>
#define siFrequency siUnits<0,0,-1,0>

#define siVelocity siUnits<1,0,-1,0>
#define siAcceleration siUnits<1,0,-2,0>

#define siNewtons  siUnits<1,1,-2,0>
#define siJoules   siUnits<2,1,-2,0>
#define siWatts    siUnits<2,1,-3,0>

#define siIrradiance siUnits<0,1,-3,0>
#define siRadiance siUnits<0,1,-3,-1>

#define siSrInv siUnits<0,0,0,-1>
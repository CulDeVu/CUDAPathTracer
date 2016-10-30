#ifndef SPHERE_H
#define SPHERE_H

#include "vec3.h"
#include "color.h"

struct sphere
{
	vec3 pos;
	float rad;
	color diffuse;
	color emm;
};

#endif
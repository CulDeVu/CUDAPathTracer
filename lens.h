#pragma once

struct lens
{
	float focalLength;
	float radius;
};

struct camera
{
	vec3 pos;

	float distFromFilm;
	float focalLength;
	float radius;
};

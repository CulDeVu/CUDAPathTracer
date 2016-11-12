#pragma once

#include <string>
#include <vector>

#include "color.h"
#include "limits.h"
#include "tiny_obj_loader.h"
#include "vec3.h"

using std::string;
using std::vector;

struct triangle
{
	int32_t v0, v1, v2;
	vec3 norm;
	int32_t mat;
};

struct materialDesc
{
	color albedo;
	color emmision;
};

// number of triangles is restricted to the number of signed positive 32bit integers
// so indexing can be done in 32bit
struct sceneDesc
{
	uint32_t numVerts;
	uint32_t numTris;
	uint32_t numMats;
	uint32_t numLights;

	vec3* verts;
	triangle* tris;
	materialDesc* mats;
	uint32_t* lights;
	float totalLightArea;
};

vector<vec3> verts;
vector<triangle> tris;
vector<materialDesc> mats;
vector<int32_t> lights;
float totalLightArea = 0;

__device__ float triIntersect(vec3 o, vec3 ray, vec3* verts_device, triangle* tris_device)
{
	vec3 v0 = verts_device[tris_device->v0];
	vec3 v1 = verts_device[tris_device->v1];
	vec3 v2 = verts_device[tris_device->v2];

	//vec3 N = tris_device[triID].norm;

	vec3 e1 = v1 - v0;
	vec3 e2 = v2 - v0;

	vec3 q = cross(ray, e2);
	float a = dot(e1, q);

	// nearly parallel
	if (abs(a) < 0.00001)
		return MAX_FLOAT;

	vec3 s = (o - v0) / a;
	vec3 r = cross(s, e1);

	float b0 = dot(s, q);
	float b1 = dot(r, ray);
	float b2 = 1.0f - b0 - b1;

	if (b0 < 0.0f)
		return MAX_FLOAT;
	if (b1 < 0.0f)
		return MAX_FLOAT;
	if (b2 < 0.0f)
		return MAX_FLOAT;

	float t = dot(e2, r);
	return t;

	/*// rays parallel?
	float NdotRayDirection = dot(N, ray);
	if (fabs(NdotRayDirection) < 0.0001)
		return MAX_FLOAT;

	float d = dot(N, v0);
	vec3 dist = sub(v0, o);
	float t = dot(dist, N) / NdotRayDirection;

	//float t = (dot(N, o) + d) / NdotRayDirection;
	if (t < 0) // triangle behind test
		return MAX_FLOAT;

	vec3 P = add(mul(ray, t), o);

	vec3 C;

	// edge 0
	vec3 edge0 = sub(v1, v0);
	vec3 vp0 = sub(P, v0);
	C = cross(edge0, vp0);
	if (dot(N, C) < 0)
		return MAX_FLOAT; // P is on right side

	// edge1
	vec3 edge1 = sub(v2, v1);
	vp0 = sub(P, v1);
	C = cross(edge1, vp0);
	if (dot(N, C) < 0)
		return MAX_FLOAT; // P is on left side

	// edge2
	vec3 edge2 = sub(v0, v2);
	vp0 = sub(P, v2);
	C = cross(edge2, vp0);
	if (dot(N, C) < 0)
		return 100; // P is on left side

	return t;*/
}

void loadOBJ(string filename, vec3 origin, float scale, bool flipNormals = false)
{
	printf("Loading .obj file: %s\n", filename.c_str());

	vector<tinyobj::shape_t> shapes;
	vector<tinyobj::material_t> materials;

	string err = tinyobj::LoadObj(shapes, materials, filename.c_str(), "models/");

	if (!err.empty())
		printf("\n\nTINYOBJ ERROR: %s \n\n", err.c_str());

	printf("Loading materials\n");
	for (int i = 0; i < materials.size(); ++i)
	{
		materialDesc m;

		m.albedo = color(materials[i].diffuse[0], materials[i].diffuse[1], materials[i].diffuse[2]);
		m.emmision = color(materials[i].emission[0], materials[i].emission[1], materials[i].emission[2]);

		mats.push_back(m);
	}

	uint32_t matsOffset = (int32_t)mats.size();

	printf("Loading materials\n");
	for (int i = 0; i < materials.size(); ++i)
	{
		materialDesc m;

		m.albedo = color(materials[i].diffuse[0], materials[i].diffuse[1], materials[i].diffuse[2]);
		m.emmision = color(materials[i].emission[0], materials[i].emission[1], materials[i].emission[2]);

		mats.push_back(m);
	}

	printf("Loaded .obj file. Loading models into RAM.\n");
	for (int i = 0; i < shapes.size(); ++i)
	{
		int32_t indexBufferOffset = (int32_t)verts.size();

		// vertex buffer
		for (int v = 0; v < shapes[i].mesh.positions.size() / 3; ++v)
		{
			verts.push_back(vec3());
			size_t ind = verts.size() - 1;
			verts[ind].x = shapes[i].mesh.positions[3 * v + 0] * scale + origin.x;
			verts[ind].y = shapes[i].mesh.positions[3 * v + 1] * scale + origin.y;
			verts[ind].z = shapes[i].mesh.positions[3 * v + 2] * scale + origin.z;
		}

		// index buffer
		for (int v = 0; v < shapes[i].mesh.indices.size() / 3; ++v)
		{
			tris.push_back(triangle());
			uint32_t ind = (uint32_t)tris.size() - 1;
			tris[ind].v0 = shapes[i].mesh.indices[3 * v + 0] + indexBufferOffset;
			tris[ind].v1 = shapes[i].mesh.indices[3 * v + 1] + indexBufferOffset;
			tris[ind].v2 = shapes[i].mesh.indices[3 * v + 2] + indexBufferOffset;

			vec3 v0 = verts[tris[ind].v0];
			vec3 v1 = verts[tris[ind].v1];
			vec3 v2 = verts[tris[ind].v2];

			tris[ind].mat = shapes[i].mesh.material_ids[0] + matsOffset;

			if (mats[tris[ind].mat].emmision.r != 0)
			{
				lights.push_back(ind);
				vec3 p1 = v1 - v0;
				vec3 p2 = v2 - v0;
				float area = length(cross(p1, p2)) / 2;
				totalLightArea += area;
				printf("got emmissive something: %f\n", area);
			}

			tris[ind].norm = normalized(cross(v1 - v0, v2 - v0));
			if (flipNormals)
				tris[ind].norm = tris[ind].norm * -1;
			//printf("v0: %f, %f, %f\n", (double)tris[ind].norm.x, (double)tris[ind].norm.y, (double)tris[ind].norm.z);
		}
	}

	printf("%zd verts added \n", verts.size());
	printf("%zd tris added\n", tris.size());
}

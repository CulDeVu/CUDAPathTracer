
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <chrono>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <fstream>
#include <math.h>

#include "modelLoader.h"

#include "BVH.h"
#include "BVHTest.h"
#include "camera.h"
#include "color.h"
#include "sphere.h"
#include "vec3.h"

// global constants
#define IMAGE_WIDTH 512
#define IMAGE_HEIGHT 512
#define IMAGE_SIZE (IMAGE_WIDTH*IMAGE_HEIGHT)
#define TILE_SIZE (IMAGE_SIZE)
#define NUM_SAMPLES 100
#define NUM_BOUNCES 3

#define MAX_BVH_DEPTH 64

void checkError()
{
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err)
		printf("error! ID: %d, \"%s\"\n", err, cudaGetErrorString(err));
}

__device__ vec3 getTangent(vec3 norm)
{
	vec3 tangent;
	vec3 c1 = cross(norm, vec3(0, 0, 1));
	vec3 c2 = cross(norm, vec3(0, 1, 0));
	if (dot(c1, c1) > dot(c2, c2))
		tangent = c1;
	else
		tangent = c2;
	return tangent;
}

__device__ float nrand(curandState* crs)
{
	return curand_uniform(crs);
}
__device__ vec3 randRay(vec3 norm, curandState* crs)
{
	float u1 = nrand(crs),
		u2 = nrand(crs);

	float r = sqrt(1.0f - u1 * u1);
	float phi = 2 * 3.14159 * u2;

	vec3 castRay = vec3(r * cos(phi), u1, r * sin(phi));
	vec3 tangent = getTangent(norm);
	vec3 bitangent = cross(norm, tangent);
	castRay = norm * castRay.y + 
			tangent * castRay.x + 
			bitangent * castRay.z;
	castRay = normalized(castRay);

	return castRay;
}
__device__ vec3 cosineWeightedRay(vec3 norm, curandState* crs)
{
	float u1 = nrand(crs),
		u2 = nrand(crs);

	float r = sqrt(u1);
	float theta = 2 * 3.14159 * u2;

	float x = r * cos(theta);
	float z = r * sin(theta);
	float y = sqrt(max(0.0f, 1.0f - u1));

	vec3 castRay = vec3(x, y, z);
	vec3 tangent = getTangent(norm);
	vec3 bitangent = cross(norm, tangent);
	castRay = norm * castRay.y + 
			tangent * castRay.x + 
			bitangent * castRay.z;
	castRay = normalized(castRay);

	return castRay;
}

__device__ color BRDF(materialDesc m, vec3 vDir, vec3 lDir)
{
	return mul(m.albedo, 1 / 3.14159);
}

//__constant__ AABBvec3 bvhArray[300000];
struct triIntersection
{
	int32_t triIndex;
	float t;
};
__device__ triIntersection trace(ray r, sceneDesc scene, BVH_array bvh, uint32_t* test)
{
	uint32_t stack[MAX_BVH_DEPTH];
	stack[0] = 0;

	float closestT = MAX_FLOAT;
	int trisID = -1;

	int i = 0;
	int counter = 0;
	while (i >= 0)
	{
		if (stack[i] & BVH_LEAF_FLAG)
		{
			float t = triIntersect(r.o, r.dir, scene.verts, scene.tris + (stack[i] ^ BVH_LEAF_FLAG));
			if (0 < t && t < closestT)
			{
				closestT = t;
				trisID = stack[i] ^ BVH_LEAF_FLAG;
			}
			++counter;
			test[stack[i] ^ BVH_LEAF_FLAG] += 1;

			--i;
		}
		else
		{
			BVH_array_node* cur = &bvh.root[stack[i]];

			bool b = rayAABBIntersect(r.o, r.dir, cur->box);
			if (b)
			{
				stack[i] = cur->right;
				stack[i + 1] = cur->left;
				++i;
			}
			else
			{
				--i;
			}
			++counter;
		}
	}
	//printf("%d\n", counter);

	triIntersection ret;
	ret.triIndex = trisID;
	ret.t = closestT;
	return ret;
}

__device__ triIntersection trace_shared(ray r, sceneDesc scene, BVH_array bvh, uint32_t* test)
{
	uint32_t warpID = MAX_BVH_DEPTH * (threadIdx.x / 32);
	__shared__ uint32_t stack[MAX_BVH_DEPTH * 16];
	stack[warpID + 0] = 2;
	stack[warpID + 1] = 1;
	stack[warpID + 2] = 0;

	float closestT = MAX_FLOAT;
	int trisID = -1;

	int i = 1;
	while (i >= 0)
	{
		if (stack[warpID + i] & BVH_LEAF_FLAG)
		{
			float t = triIntersect(r.o, r.dir, scene.verts, scene.tris + (stack[warpID + i] ^ BVH_LEAF_FLAG));
			if (0 < t && t < closestT)
			{
				closestT = t;
				trisID = stack[warpID + i] ^ BVH_LEAF_FLAG;
			}
		}
		else
		{
			BVH_array_node* cur = &bvh.root[stack[warpID + i]];

			bool b = rayAABBIntersect(r.o, r.dir, cur->box);
			if (b)
			{
				stack[warpID + i + 1] = cur->left;
				stack[warpID + i + 2] = 0;
			}
		}

		__syncthreads();
		if (stack[warpID + i + 1] != 0)
		{
			stack[warpID + i] = bvh.root[stack[warpID + i]].right;
			++i;
		}
		else
		{
			stack[warpID + i] = 0;
			--i;
		}
	}

	triIntersection ret;
	ret.triIndex = trisID;
	ret.t = closestT;
	return ret;
}

__device__ color radianceAlongSingleStep(ray vDir, sceneDesc scene, BVH_array bvh, curandState* crs, uint32_t* test)
{
	const int LIGHT_PATH_SIZE = 2;
	const int CAMERA_PATH_SIZE = 3;
	const int PATH_SIZE = LIGHT_PATH_SIZE + CAMERA_PATH_SIZE;
	const int camInd = PATH_SIZE - 1;

	vec3 x[PATH_SIZE];
	int32_t mat[PATH_SIZE];
	vec3 norm[PATH_SIZE];
	float prob[PATH_SIZE];

	// light path
	{
		float randArea = scene.totalLightArea * nrand(crs);
			int selectedTri = 0;
		for (int j = 0; j < scene.numLights; ++j)
		{
			vec3 v0 = scene.verts[scene.tris[scene.lights[j]].v0];
			vec3 v1 = scene.verts[scene.tris[scene.lights[j]].v1];
			vec3 v2 = scene.verts[scene.tris[scene.lights[j]].v2];
			vec3 a1 = v1 - v0;
			vec3 a2 = v2 - v0;
			float area = length(cross(a1, a2)) / 2;
			if (randArea < area && randArea > 0)
				selectedTri = scene.lights[j];
			randArea -= area;
		}

		float u = nrand(crs);
		float v = nrand(crs);
		vec3 v0 = scene.verts[scene.tris[selectedTri].v0];
		vec3 v1 = scene.verts[scene.tris[selectedTri].v1];
		vec3 v2 = scene.verts[scene.tris[selectedTri].v2];
		vec3 a1 = v1 - v0;
		vec3 a2 = v2 - v0;

		if (u + v > 1.0)
		{
			u += 2 * (0.5 - u);
			v += 2 * (0.5 - v);
		}

		vec3 normal = scene.tris[selectedTri].norm;
		vec3 pos1 = v0 + a1 * u + a2 * v + normal * 0.001f;

		x[0] = pos1;
		norm[0] = normal;
		mat[0] = scene.tris[selectedTri].mat;
		prob[0] = 1 / scene.totalLightArea;
	}
	{
		vec3 oDir = randRay(norm[0], crs);
		ray vDir2;
		vDir2.o = x[0];
		vDir2.dir = oDir;
		triIntersection intersect = trace(vDir2, scene, bvh, test);

		int trisID = intersect.triIndex;
		float closestT = intersect.t;

		closestT -= 0.001;
		if (closestT > MAX_FLOAT - 1)
		{
			trisID = 0;
			closestT = 0;
		}

		triangle curTris = scene.tris[trisID];
		materialDesc curMat = scene.mats[curTris.mat];
		vec3 normal2 = scene.tris[trisID].norm;

		vec3 iDir2 = vDir2.dir * -1;
		vec3 pos = vDir2.o + vDir2.dir * closestT;

		float G = abs(dot(normal2, oDir)) / max(0.001f, closestT * closestT);

		x[1] = pos;
		norm[1] = normal2;
		mat[1] = curTris.mat;
		prob[1] = max(0.001f, 1 / (2 * 3.14159) * G);
	}

	// camera path
	{
		x[camInd] = vDir.o;
		norm[camInd] = vDir.dir;
		prob[camInd] = 1;
	}
	{
		triIntersection intersect = trace(vDir, scene, bvh, test);
		int trisID = intersect.triIndex;
		float closestT = intersect.t;

		closestT -= 0.001;
		if (closestT > MAX_FLOAT - 1)
		{
			trisID = 0;
			closestT = 0;
		}

		triangle curTris = scene.tris[trisID];
		materialDesc curMat = scene.mats[curTris.mat];
		vec3 normal = scene.tris[trisID].norm;

		vec3 pos = vDir.o + vDir.dir * closestT;

		x[camInd - 1] = pos;
		norm[camInd - 1] = normal;
		mat[camInd - 1] = curTris.mat;
		prob[camInd - 1] = 1;
	}
	{
		ray vDir;
		vDir.o = x[camInd - 1];
		vDir.dir = cosineWeightedRay(norm[camInd - 1], crs);
		triIntersection intersect = trace(vDir, scene, bvh, test);
		intersect.t -= 0.001;

		vec3 normal = scene.tris[intersect.triIndex].norm;

		float G = abs(dot(normal, vDir.dir) * dot(norm[camInd - 1], vDir.dir)) / max(0.001f, intersect.t * intersect.t);

		x[camInd - 2] = vDir.o + vDir.dir * intersect.t;
		norm[camInd - 2] = scene.tris[intersect.triIndex].norm;
		mat[camInd - 2] = scene.tris[intersect.triIndex].mat;
		prob[camInd - 2] = max(0.001f, G / 3.14159);
	}

	color accum = color(0, 0, 0);
	color L_e = scene.mats[mat[0]].emmision;
	for (int i = 0; i < LIGHT_PATH_SIZE; ++i)
	//int i = LIGHT_PATH_SIZE - 1;
	{
		for (int j = LIGHT_PATH_SIZE; j < PATH_SIZE - 1; ++j)
		//int j = LIGHT_PATH_SIZE;
		{
			color weight = L_e / prob[0];

			// light path first
			for (int k = 1; k <= i; ++k)
			{
				vec3 seg = x[k] - x[k - 1];
				vec3 ray = normalized(seg);
				float G = abs(dot(ray, norm[k]) * dot(ray, norm[k - 1])) / max(0.001f, dot(seg, seg));
				if (G != G)
					G = 0;
				color f_s = scene.mats[mat[k]].albedo / 3.14159f;
				weight = mul(weight, f_s) * G / max(0.001f, prob[k]);
			}

			for (int k = j + 1; k < PATH_SIZE - 1; ++k)
			{
				vec3 seg = x[k] - x[k - 1];
				vec3 ray = normalized(seg);
				float G = abs(dot(ray, norm[k]) * dot(ray, norm[k - 1])) / max(0.001f, dot(seg, seg));
				if (G != G)
					G = 0;
				color f_s = scene.mats[mat[k]].albedo / 3.14159f;
				weight = mul(weight, f_s) * G / max(0.001f, prob[k]);
			}

			// the middle link
			{
				vec3 seg = x[j] - x[i];
				float len = length(seg);
				vec3 ray = normalized(seg);
				float G = abs(dot(ray, norm[j]) * dot(ray, norm[i])) / max(0.001f, dot(seg, seg));
				if (G != G)
					G = 0;
				color f_s = scene.mats[mat[j]].albedo / 3.14159f;
				weight = mul(weight, f_s) * G / max(0.001f, prob[j]);
				float m = max(weight.r, max(weight.g, weight.b));

				float V = 0;
				if (m > 0.01)
				{
					//ray vDir;
					vDir.o = x[i];
					vDir.dir = ray;
					triIntersection intersect = trace(vDir, scene, bvh, test);
					if (abs(intersect.t - len) <= 0.01)
						V = 1;
				}
				weight = weight * V;
			}

			accum = add(accum, weight);
			accum = add(accum, scene.mats[mat[PATH_SIZE - 2]].emmision);
		}
	}

	// calc final contribution
	/*color accum = color(0, 0, 0);
	{
		color weight = color(1, 1, 1);

		color L_e = scene.mats[mat[0]].emmision;

		vec3 seg0_1 = x[1] - x[0];
		vec3 ray0_1 = normalized(seg0_1);
		float G0_1 = abs(dot(ray0_1, norm[0]) * dot(ray0_1, norm[1])) / max(0.001f, dot(seg0_1, seg0_1));
		if (G0_1 != G0_1)
			G0_1 = 0;
		color f0_1_2 = mul(scene.mats[mat[1]].albedo, 1 / 3.14159);

		vec3 seg1_2 = x[2] - x[1];
		float len1_2 = length(x[2] - x[1]);
		vec3 ray1_2 = normalized(seg1_2);
		float V1_2 = 1;
		{
			ray vDir;
			vDir.o = x[1];
			vDir.dir = ray1_2;
			triIntersection intersect = trace(vDir, scene, bvh, test);
			if (abs(intersect.t - len1_2) > 0.01)
				V1_2 = 0;
		}
		float G1_2 = V1_2 * abs(max(0.0f, dot(ray1_2, norm[1])) * max(0.0f, dot(ray1_2 * -1, norm[2]))) / max(0.001f, dot(seg1_2, seg1_2));
		color f1_2_3 = mul(scene.mats[mat[2]].albedo, 1 / 3.14159);

		vec3 seg2_3 = x[2] - x[3];
		vec3 ray2_3 = normalized(seg2_3);
		float G2_3 = abs(dot(ray2_3, norm[2]) * dot(ray2_3, norm[3])) / max(0.001f, dot(seg2_3, seg2_3));
		if (G2_3 != G2_3)
			G2_3 = 0;

		accum = mul(mul(L_e, f0_1_2), f1_2_3) * G0_1 * G1_2 / max(0.001f, prob[0] * prob[1] * prob[2]);

		// direct lighting
		vec3 seg0_2 = x[2] - x[0];
		float len0_2 = length(seg0_2);
		vec3 ray0_2 = normalized(seg0_2);
		float V0_2 = 1;
		{
			ray vDir;
			vDir.o = x[0];
			vDir.dir = ray0_2;
			triIntersection intersect = trace(vDir, scene, bvh, test);
			if (abs(intersect.t - len0_2) > 0.01)
				V0_2 = 0;
		}
		float G0_2 = V0_2 * abs(dot(ray0_2, norm[0]) * dot(ray0_2, norm[2])) / max(0.001f, dot(seg0_2, seg0_2));
		if (G0_2 != G0_2)
			G0_2 = 0;
		color f0_2_3 = mul(scene.mats[mat[2]].albedo, 1 / 3.14159);

		accum = add(accum, mul(L_e, f0_2_3) * G0_2 / max(0.001f, prob[0] * prob[2]));

		// directly intersecting light
		accum = add(accum, scene.mats[mat[2]].emmision);

		if (accum.r != accum.r)
			accum = color(5, 0, 0);
	}*/

	return accum;
}

__device__ color radianceAlongSingleStep2(ray vDir, sceneDesc scene, BVH_array bvh, curandState* crs, uint32_t* test)
{
	color accum = color(0, 0, 0);
	color weight = color(1, 1, 1);

	for (int i = 0; i < NUM_BOUNCES; ++i)
	{
		// intersection routine
		triIntersection intersect = trace(vDir, scene, bvh, test);
		int trisID = intersect.triIndex;
		float closestT = intersect.t;

		closestT -= 0.001;
		if (closestT < 0.001)
		{
			weight = color(0, 0, 0);
		}
		if (closestT > MAX_FLOAT - 1)
		{
			weight = color(0, 0, 0);
			trisID = 0;
			closestT = 0;
		}

		triangle curTris = scene.tris[trisID];
		materialDesc curMat = scene.mats[curTris.mat];
		vec3 normal = scene.tris[trisID].norm;

		vec3 oDir = vDir.dir * -1;
		vec3 lDir;
		vec3 pos = vDir.o + vDir.dir * closestT;

		float cosODir = abs(dot(oDir, normal));

		if (curMat.emmision.r != 0)
		{
			accum = add(accum, mul(weight, curMat.emmision));
			weight = color(0, 0, 0);
		}

		float a = nrand(crs);
		if (a < 0.5)
		{
			lDir = cosineWeightedRay(normal, crs);
			color curWeight = BRDF(curMat, oDir, lDir) * 3.14159;
			weight = mul(weight, curWeight);
		}
		else
		{
			float randArea = scene.totalLightArea * nrand(crs);
			int selectedTri = 0;
			for (int j = 0; j < scene.numLights; ++j)
			{
				vec3 v0 = scene.verts[scene.tris[scene.lights[j]].v0];
				vec3 v1 = scene.verts[scene.tris[scene.lights[j]].v1];
				vec3 v2 = scene.verts[scene.tris[scene.lights[j]].v2];
				vec3 a1 = v1 - v0;
				vec3 a2 = v2 - v0;
				float area = length(cross(a1, a2)) / 2;
				if (randArea < area && randArea > 0)
					selectedTri = scene.lights[j];
				randArea -= area;
			}

			float u = nrand(crs);
			float v = nrand(crs);
			vec3 v0 = scene.verts[scene.tris[selectedTri].v0];
			vec3 v1 = scene.verts[scene.tris[selectedTri].v1];
			vec3 v2 = scene.verts[scene.tris[selectedTri].v2];
			vec3 a1 = v1 - v0;
			vec3 a2 = v2 - v0;

			if (u + v > 1.0)
			{
				u += 2 * (0.5 - u);
				v += 2 * (0.5 - v);
			}

			vec3 pos1 = v0 + a1 * u + a2 * v;
			vec3 d = pos1 - pos;
			lDir = normalized(d);

			float invProb = scene.totalLightArea;
			float cosLDir = max(0.0f, dot(lDir, normal));
			float cosO1Dir = max(0.0f, dot(vec3(0, -1, 0), lDir * -1));
			float G = cosLDir * cosO1Dir / dot(d, d);

			weight = mul(weight, BRDF(curMat, oDir, lDir) * G * invProb);
			i = max(i, NUM_BOUNCES - 2);
		}

		vDir.o = pos;
		vDir.dir = lDir;
	}

	return accum;
}

//-----------------------------------------------------------------------------
// global functions
//-----------------------------------------------------------------------------
__global__ void setupImgBuffer(color* imgBuff)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx >= IMAGE_SIZE)
		return;
	imgBuff[idx] = color(0, 0, 0);
}
__global__ void setupCurand(curandState *state)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx >= IMAGE_SIZE)
		return;
	curand_init(1234, idx, 0, &state[idx]);
}

__global__ void drawPixel(
	color* imgBuff,
	sceneDesc scene,
	camera* cam,
	BVH_array bvh,
	int curSampleNum,
	curandState* randState, uint32_t* test)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx >= IMAGE_SIZE)
		return;

	ray r = cam->cameraRay(idx, nrand(randState), nrand(randState));

	color result = radianceAlongSingleStep(r, scene, bvh, &randState[idx], test);

	color prevSum = imgBuff[idx];
	imgBuff[idx] = add(mul(prevSum, (float)(curSampleNum - 1) / curSampleNum), mul(result, 1.0f / curSampleNum));
}

int main()
{
	// print out CUDA properties
	cudaDeviceProp p;
	cudaGetDeviceProperties(&p, 0);
	printf("Device name: %s\n", p.name);
	printf("Compute ver: %d.%d\n", p.major, p.minor);
	printf("Memory:\n");
	{
		printf("  Global: %zd\n", p.totalGlobalMem);
		printf("  Shared (per block): %zd\n", p.sharedMemPerBlock);
		printf("  Constant: %zd\n", p.totalConstMem);
		printf("  Registers (per block): %d\n", p.regsPerBlock);
	}
	printf("Dimensions:\n");
	{
		printf("  Threads per block: %d, %d, %d (%d max total)\n", p.maxThreadsDim[0], p.maxThreadsDim[1], p.maxThreadsDim[2], p.maxThreadsPerBlock);
		printf("  Blocks: %d, %d, %d\n", p.maxGridSize[0], p.maxGridSize[1], p.maxGridSize[2]);
	}
	printf("\n");

	// load shit
	//loadOBJ("models/sponza_light.obj", vec3(), 1);
	loadOBJ("models/CornellBox-Original.obj", vec3(), 1);
	loadOBJ("models/teapot.obj", vec3(0.35, 0.6, 0.3), 0.75);
	//loadOBJ("models/dragon_simple.obj", vec3(0.3, 0.6, 0.5), 1);
	//loadOBJ("models/cube.obj", vec3(0, 0, 0), 0.5);
	//loadOBJ("models/my_cornell.obj", vec3(), 1);
	//loadOBJ("models/CornellBox-Sphere.obj", vec3(), 1);
	/*for (int i = 0; i < 200; ++i)
		loadOBJ("models/cube.obj", vec3(0, 0, 0), 0.25);
	loadOBJ("models/cube.obj", vec3(0, 1, 0), 0.5);*/
	//getchar();
	BVH_array bvh = buildBVH();
	printf("MAX DEPTH OF TREE: %d\n", bvh.depth);
	/*getchar();*
/	/*printf("---\n");
	for (int i = 0; i < bvh.size; ++i)
	{
		printf("(%f,%f,%f), (%f,%f,%f), %d,%d\n", bvh.root[i].box.lo.x, bvh.root[i].box.lo.y, bvh.root[i].box.lo.z, 
			bvh.root[i].box.hi.x, bvh.root[i].box.hi.y, bvh.root[i].box.hi.z, 
			bvh.root[i].left & MAX_BVH_INDEX, bvh.root[i].right & MAX_BVH_INDEX);
	}
	printf("---\n");*/

	// output DOT file
	/*for (int i = 0; i < bvh.size; ++i)
	{
		if (bvh.root[i].left & BVH_LEAF_FLAG)
			printf("\t%d -- -%zd\n", i, bvh.root[i].left ^ BVH_LEAF_FLAG);
		else
			printf("\t%d -- %d\n", i, bvh.root[i].left);
		
		if (bvh.root[i].right & BVH_LEAF_FLAG)
			printf("\t%d -- -%zd\n", i, bvh.root[i.]newtfl1A1right ^ BVH_LEAF_FLAG);
		else
			printf("\t%d -- %d\n", i, bvh.root[i].right);
	}*/

	if (bvh.depth >= MAX_BVH_DEPTH)
	{
		printf("Critical Error: BVH depth is too big\n");
		return 0;
	}

	int nThreads = 512;
	int nblocks = ceil((float)IMAGE_SIZE / nThreads);

	// setup the random number generator
	curandState* randState_device;
	cudaMalloc((void**)&randState_device, IMAGE_SIZE * sizeof(curandState));
	setupCurand <<< nblocks, nThreads >>>(randState_device);

	// set up the camera
	camera cam;
	cam.pos = vec3(0, 1, 3);
	cam.distFromFilm = 1;
	cam.focalLength = 3;
	cam.radius = 0.0;
	cam.pxlWidth = IMAGE_WIDTH;
	cam.pxlHeight = IMAGE_HEIGHT;
	camera* cam_device;
	cudaMalloc((void**)&cam_device, sizeof(camera));
	cudaMemcpy(cam_device, &cam, sizeof(camera), cudaMemcpyHostToDevice);

	cudaDeviceSynchronize();
	checkError();
	
	// setup host image buffer
	color* imgBuffer_host = (color*)malloc(IMAGE_SIZE * sizeof(color));

	// setup device image buffer
	color* imgBuffer_device;
	cudaMalloc((void**)&imgBuffer_device, IMAGE_SIZE * sizeof(color));
	setupImgBuffer <<< nblocks, nThreads >>>(imgBuffer_device);

	// vertex buffer
	sceneDesc scene_device;
	cudaMalloc((void**)&scene_device.verts, verts.size() * sizeof(vec3));
	cudaMemcpy(scene_device.verts, &(verts[0]), verts.size() * sizeof(vec3), cudaMemcpyHostToDevice);
	scene_device.numVerts = verts.size();

	// triangle buffer
	cudaMalloc((void**)&scene_device.tris, tris.size() * sizeof(triangle));
	cudaMemcpy(scene_device.tris, &(tris[0]), tris.size() * sizeof(triangle), cudaMemcpyHostToDevice);
	scene_device.numTris = tris.size();

	// material buffer
	cudaMalloc((void**)&scene_device.mats, mats.size() * sizeof(materialDesc));
	cudaMemcpy(scene_device.mats, &(mats[0]), mats.size() * sizeof(materialDesc), cudaMemcpyHostToDevice);
	scene_device.numMats = mats.size();

	// light buffer
	cudaMalloc((void**)&scene_device.lights, lights.size() * sizeof(int32_t));
	cudaMemcpy(scene_device.lights, &(lights[0]), lights.size() * sizeof(int32_t), cudaMemcpyHostToDevice);
	scene_device.numLights = lights.size();
	scene_device.totalLightArea = totalLightArea;

	// BVH
	BVH_array bvh_device;
	cudaMalloc((void**)&bvh_device.root, bvh.size * sizeof(BVH_array_node));
	cudaMemcpy(bvh_device.root, bvh.root, bvh.size * sizeof(BVH_array_node), cudaMemcpyHostToDevice);
	bvh_device.size = bvh.size;
	bvh_device.depth = bvh.depth;
	//cudaMemcpyToSymbol(bvhArray, bvh.root, bvh.size * sizeof(BVH_array_node));

	// bvh intersection counter
	uint32_t* bvhIntersection_device;
	cudaMalloc((void**)&bvhIntersection_device, bvh.size * sizeof(uint32_t));
	cudaMemset(bvhIntersection_device, 0, bvh.size * sizeof(uint32_t));

	cudaDeviceSynchronize();
	checkError();

	// render loop
	printf("\nEntering render loop...\n\n");
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	{
		int sampleNum = 0;
		while (sampleNum < NUM_SAMPLES)
		{
			auto start = std::chrono::steady_clock::now();

			drawPixel <<< nblocks, nThreads >>>(imgBuffer_device, scene_device, cam_device, bvh_device, sampleNum + 1, randState_device, bvhIntersection_device);

			if ((sampleNum + 1) % 10 == 0 && sampleNum)
				printf("sample %d finished\n", sampleNum);
			cudaDeviceSynchronize();

			auto end = std::chrono::steady_clock::now();
			double diff = std::chrono::duration<double>(end - start).count();
			if (diff > 0.5)
				printf("-----Possible too long execution of %f seconds-----\n", diff);

			checkError();
			++sampleNum;
		}
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop); // shouldnt need this but fuck CUDA
	printf("exiting render loop!\n\n");

	{
		uint32_t* a = new uint32_t[bvh.size];
		cudaMemcpy(a, bvhIntersection_device, bvh.size * sizeof(uint32_t), cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		std::ofstream f("out.csv");
		for (int i = 0; i < bvh.size; ++i)
			f << a[i] << ",\n";
		delete[] a;
	}

	// report time taken
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Render took %f ms (%f s)\n", milliseconds, milliseconds / 1000);
	printf("%f ms per loop \n", milliseconds / NUM_SAMPLES);
	printf("%f Mrays/s\n", IMAGE_SIZE * NUM_SAMPLES * (NUM_BOUNCES + 1) / (milliseconds * 1000));

	// build imgBuffer_host
	cudaMemcpy(imgBuffer_host, imgBuffer_device, IMAGE_SIZE * sizeof(color), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	// save the file
	FILE* fp = fopen("image.ppm", "w");
	fprintf(fp, "P3 %d %d 255\n", IMAGE_WIDTH, IMAGE_HEIGHT);
	for (int y = 0; y < IMAGE_HEIGHT; ++y)
	{
		for (int x = IMAGE_WIDTH - 1; x >= 0; --x)
		{
			//int idx = y * IMAGE_WIDTH + x;
			int idx = cam.mortonPxltoI(x, y);
			//scanlinePxltoI(&idx, x, y, IMAGE_WIDTH, IMAGE_HEIGHT);
			//mortonPxltoI(&idx, x, y, IMAGE_WIDTH, IMAGE_HEIGHT);
			color c = gammaCorrect(normalized(imgBuffer_host[idx]), 1 / 2.2);
			fprintf(fp, "%d %d %d ", (int)(c.r * 255), (int)(c.g * 255), (int)(c.b * 255));
		}
	}
	fclose(fp);

	printf("\nfinished");

	cudaFree(randState_device);
	cudaFree(cam_device);

	getchar();

	return 0;
}
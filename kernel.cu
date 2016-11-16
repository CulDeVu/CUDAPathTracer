
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <chrono>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>

#include "modelLoader.h"

#include "BVH.h"
#include "camera.h"
#include "color.h"
#include "sphere.h"
#include "vec3.h"

// global constants
#define IMAGE_WIDTH 512
#define IMAGE_HEIGHT 512
#define IMAGE_SIZE (IMAGE_WIDTH*IMAGE_HEIGHT)
#define TILE_SIZE (IMAGE_SIZE)
#define NUM_SAMPLES 1000

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
__device__ triIntersection trace(ray r, sceneDesc scene, BVH_array bvh)
{
	uint32_t stack[MAX_BVH_DEPTH];
	stack[0] = 0;

	float closestT = MAX_FLOAT;
	int trisID = -1;

	int i = 0;
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
		}
	}

	triIntersection ret;
	ret.triIndex = trisID;
	ret.t = closestT;
	return ret;
}

/*__device__ bool radianceAlongSingleStep(pathState* pathStatePtr, sceneDesc scene, BVH_array bvh, curandState* crs)
{
	bool doneBouncing = false;

	// intersection routine
	triIntersection intersect = trace(pathStatePtr->vDir, scene, bvh);
	int trisID = intersect.triIndex;
	float closestT = intersect.t;

	closestT -= 0.001;
	if (closestT < 0.001)
	{
		pathStatePtr->weight = color(0, 0, 0);
		doneBouncing = true;
	}
	if (closestT > MAX_FLOAT - 1)
	{
		pathStatePtr->accumulation = mul(pathStatePtr->weight, color(0, 0, 0));
		doneBouncing = true;
	}

	triangle curTris = scene.tris[trisID];
	materialDesc curMat = scene.mats[curTris.mat];
	vec3 normal = scene.tris[trisID].norm;

	vec3 oDir = pathStatePtr->vDir.dir * -1;
	vec3 lDir;
	vec3 pos = pathStatePtr->vDir.o + pathStatePtr->vDir.dir * closestT;

	float cosODir = dot(oDir, normal);

	if (curMat.emmision.r != 0 && cosODir > 0)
	{
		pathStatePtr->accumulation = mul(pathStatePtr->weight, curMat.emmision);
		pathStatePtr->weight = color(0, 0, 0);
		doneBouncing = true;
	}
	else if (pathStatePtr->bounceNum == 0)
	{
		pathStatePtr->weight = color(0, 0, 0);
		doneBouncing = true;
	}

	__shared__ int path;
	float a = nrand(crs);
	path = floor(a * 2);
	__syncthreads();

	/*int path;
	float a = nrand(crs);
	path = floor(a * 2);*

	// which sampling strat?
	if (path == 0)
	{
		lDir = cosineWeightedRay(normal, crs);
		color curWeight = curMat.albedo;
		pathStatePtr->weight = mul(pathStatePtr->weight, curWeight);
	}
	else
	{
		float randArea = scene.totalLightArea * nrand(crs);
		int selectedTri = 0;
		for (int i = 0; i < scene.numLights; ++i)
		{
			vec3 v0 = scene.verts[scene.tris[scene.lights[i]].v0];
			vec3 v1 = scene.verts[scene.tris[scene.lights[i]].v1];
			vec3 v2 = scene.verts[scene.tris[scene.lights[i]].v2];

			vec3 a1 = v1 - v0;
			vec3 a2 = v2 - v0;
			float area = length(cross(a1, a2)) / 2;

			if (randArea < area && randArea > 0)
				selectedTri = scene.lights[i];

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
		pathStatePtr->weight = mul(pathStatePtr->weight, BRDF(curMat, oDir, lDir) * G * invProb);
		pathStatePtr->bounceNum = 1;
	}

	pathStatePtr->vDir.o = pos;
	pathStatePtr->vDir.dir = lDir;
	pathStatePtr->bounceNum -= 1;
	//pathState->bounceNum = 0;
	return doneBouncing;
}*/

__device__ color radianceAlongSingleStep2(ray vDir, sceneDesc scene, BVH_array bvh, curandState* crs)
{
	color accum = color(0, 0, 0);
	color weight = color(1, 1, 1);

	for (int i = 0; i < 3; ++i)
	{
		// intersection routine
		triIntersection intersect = trace(vDir, scene, bvh);
		int trisID = intersect.triIndex;
		float closestT = intersect.t;

		closestT -= 0.001;
		if (closestT < 0.001)
		{
			weight = color(0, 0, 0);
		}
		if (closestT > MAX_FLOAT - 1)
		{
			accum = mul(weight, color(0, 0, 0));
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

		float cosODir = dot(oDir, normal);

		if (curMat.emmision.r != 0 && cosODir > 0)
		{
			accum = add(accum, mul(weight, curMat.emmision));
			weight = color(0, 0, 0);
		}

		lDir = cosineWeightedRay(normal, crs);
		color curWeight = BRDF(curMat, oDir, lDir) * 3.14159;
		weight = mul(weight, curWeight);

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
	curandState* randState)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx >= IMAGE_SIZE)
		return;

	ray r = cam->cameraRay(idx, nrand(randState), nrand(randState));

	color result = radianceAlongSingleStep2(r, scene, bvh, &randState[idx]);

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
	//loadOBJ("models/teapot.obj", vec3(0, 1, 0), 1);
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
	/*getchar();*/
	/*printf("---\n");
	for (int i = 0; i < bvh.size; ++i)
	{
		printf("(%f,%f,%f), (%f,%f,%f), %d,%d\n", bvh.root[i].box.lo.x, bvh.root[i].box.lo.y, bvh.root[i].box.lo.z, 
			bvh.root[i].box.hi.x, bvh.root[i].box.hi.y, bvh.root[i].box.hi.z, 
			bvh.root[i].left & MAX_BVH_INDEX, bvh.root[i].right & MAX_BVH_INDEX);
	}
	printf("---\n");*/

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

	cudaDeviceSynchronize();
	checkError();

	// render loop
	printf("\nEntering render loop...\n\n");
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	{
		int sampleNum = 1;
		while (sampleNum < NUM_SAMPLES)
		{
			auto start = std::chrono::steady_clock::now();

			drawPixel <<< nblocks, nThreads >>>(imgBuffer_device, scene_device, cam_device, bvh_device, sampleNum, randState_device);

			if (sampleNum % 10 == 0)
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

	// report time taken
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Render took %f ms (%f s)\n", milliseconds, milliseconds / 1000);
	printf("%f ms per loop \n", milliseconds / NUM_SAMPLES);

	// build imgBuffer_host
	cudaMemcpy(imgBuffer_host, imgBuffer_device, IMAGE_SIZE * sizeof(color), cudaMemcpyDeviceToHost);

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
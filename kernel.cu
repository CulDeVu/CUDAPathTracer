
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>

#include "modelLoader.h"

#include "BVH.h"
#include "lens.h"
#include "color.h"
#include "sphere.h"
#include "vec3.h"

// global constants
#define IMAGE_WIDTH 512
#define IMAGE_HEIGHT 512
#define IMAGE_SIZE (IMAGE_WIDTH*IMAGE_HEIGHT)
#define TILE_SIZE (IMAGE_SIZE)
#define NUM_SAMPLES 1

#define NUM_SPHERES 8

struct ray
{
	vec3 o;
	vec3 dir;
};
struct pathState
{
	ray vDir;
	color weight;
	int bounceNum;
	int sampleNum;
};

void checkError()
{
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err)
		printf("error! ID: %d, \"%s\"\n", err, cudaGetErrorString(err));
}

sceneDesc scene_device;

__device__ ray getCameraRay(int idx)
{
	float y = (int)(idx / IMAGE_WIDTH) - IMAGE_HEIGHT / 2;
	float x = idx % IMAGE_WIDTH - IMAGE_WIDTH / 2;

	ray ret;
	ret.o = vec3(0, 1.f, 5);
	ret.dir = normalized(vec3(x / 2 / IMAGE_WIDTH, y / 2 / IMAGE_HEIGHT, -0.5));
	return ret;
}
__device__ ray randCameraRay(camera* cam, vec3 posRelFilm, curandState* crs)
{
	float r = cam->radius * sqrt(curand_uniform(crs));
	float theta = 2 * 3.14159 * curand_uniform(crs);
	vec3 o = vec3(r * cos(theta), r * sin(theta), 0);

	posRelFilm.z = cam->distFromFilm;
	posRelFilm = posRelFilm * -cam->focalLength / cam->distFromFilm;
	ray ret;
	ret.o = o + cam->pos;
	ret.dir = normalized(posRelFilm - o);

	return ret;
}
__device__ float intersectSphere(vec3 o, vec3 ray, vec3 cpos, float radius)
{
	vec3 nray = normalized(ray);
	float a = dot(ray, ray);
	vec3 oRelCpos = o - cpos;
	float b = 2.0 * dot(ray, oRelCpos);
	float c = dot(oRelCpos, oRelCpos) - radius*radius;
	if (b * b - 4.0 * a * c < 0.0)
		return 0.0;
	float t0 = (-b - sqrt(b * b - 4.0 * a * c)) / (2.0 * a),
		t1 = (-b + sqrt(b * b - 4.0 * a * c)) / (2.0 * a);

	if (t0 <= 0.0 && t1 >= 0.0)
		return t1;
	if (t1 <= 0.0 && t0 >= 0.0)
		return t0;
	return min(t0, t1);
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
__device__ vec3 cosineWeightedRay(vec3 norm, curandState* crs) {
	float u1 = nrand(crs),
		u2 = nrand(crs);

	float r_sqr = 1.0 - u1 * u1;
	if (r_sqr < 0.0)
		r_sqr = 0.0;
	float r = sqrt(r_sqr);
	float theta = 2.0 * 3.14159 * u2;

	float x = r * cos(theta);
	float z = r * sin(theta);
	float y = u1;

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

__device__ bool radianceAlongSingleStep(pathState* pathState, sceneDesc scene, curandState* crs)
{
	if (pathState->bounceNum > 5)
	{
		pathState->weight = color(0, 0, 0);
		return true;
	}

	// intersect
	float closestT = MAX_FLOAT;
	int trisID = -1;
	for (int i = 0; i < scene.numTris; ++i)
	{
		float t = triIntersect(pathState->vDir.o, pathState->vDir.dir, scene.verts, scene.tris, i);
		if (0 < t && t < closestT)
		{
			closestT = t;
			trisID = i;
		}
	}
	if (closestT > MAX_FLOAT - 1)
	{
		pathState->weight = color(0, 0, 0);
		return true;
	}

	triangle curTris = scene.tris[trisID];
	materialDesc curMat = scene.mats[curTris.mat];

	pathState->vDir.o = pathState->vDir.o + pathState->vDir.dir * closestT;
	vec3 normal = scene.tris[trisID].norm;

	if (curMat.emmision.r != 0)
	{
		pathState->weight = mul(pathState->weight, curMat.emmision);
		return true;
	}

	vec3 vDirReverse = pathState->vDir.dir * -1;
	vec3 lDir = cosineWeightedRay(normal, crs);

	color curWeight = curMat.albedo;
	pathState->weight = mul(pathState->weight, curWeight);

	pathState->vDir.dir = lDir;
	pathState->bounceNum += 1;
	return false;
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
__global__ void setupPathStateBuffer(pathState* pathStateBuffer, camera* cam, curandState* randState)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx >= IMAGE_SIZE)
		return;

	float y = (float)((int)(idx / IMAGE_WIDTH)) / IMAGE_HEIGHT - 0.5;
	float x = (float)(idx % IMAGE_WIDTH) / IMAGE_WIDTH - 0.5;

	pathStateBuffer[idx].vDir = randCameraRay(cam, vec3(x, y, 0), &randState[idx]);;
	pathStateBuffer[idx].weight = color(1, 1, 1);
	pathStateBuffer[idx].bounceNum = 0;
	pathStateBuffer[idx].sampleNum = 1;
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
	pathState* pathStateBuffer,
	sceneDesc scene,
	camera* cam,
	curandState* randState)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx >= IMAGE_SIZE)
		return;

	bool result = radianceAlongSingleStep(&pathStateBuffer[idx], scene, &randState[idx]);
	if (result)
	{
		int curSampleNum = pathStateBuffer[idx].sampleNum;
		color prevSum = denormalized(imgBuff[idx]);
		imgBuff[idx] = add(mul(prevSum, (float)(curSampleNum - 1) / curSampleNum), mul(pathStateBuffer[idx].weight, 1.0f / curSampleNum));

		float y = (float)((int)(idx / IMAGE_WIDTH)) / IMAGE_HEIGHT - 0.5;
		float x = (float)(idx % IMAGE_WIDTH) / IMAGE_WIDTH - 0.5;

		imgBuff[idx] = normalized(imgBuff[idx]);
		pathStateBuffer[idx].vDir = randCameraRay(cam, vec3(x, y, 0), &randState[idx]);
		//pathStateBuffer[idx].vDir = getCameraRay(idx);
		pathStateBuffer[idx].weight = color(1, 1, 1);
		pathStateBuffer[idx].bounceNum = 0;
		pathStateBuffer[idx].sampleNum += 1;
	}
}

/*__host__ sphere* setupScene()
{
	sphere* scene = (sphere*)malloc(NUM_SPHERES * sizeof(sphere));

	scene[0].pos = newvec3(0, -3 - 1000, 0); // bottom
	scene[0].rad = 1000;
	scene[0].diffuse = newcolor(0.25, 0.25, 0.25);
	scene[0].emm = newcolor(0, 0, 0);

	scene[1].pos = newvec3(0, 0, -20 - 1000); // front
	scene[1].rad = 1000;
	scene[1].diffuse = newcolor(0.75, 0.75, 0.75);
	scene[1].emm = newcolor(0, 0, 0);

	scene[2].pos = newvec3(-5 - 1000, 0, 0); // left
	scene[2].rad = 1000;
	scene[2].diffuse = newcolor(0.75, 0.25, 0.25);
	scene[2].emm = newcolor(0, 0, 0);

	scene[3].pos = newvec3(5 + 1000, 0, 0); // right
	scene[3].rad = 1000;
	scene[3].diffuse = newcolor(0.25, 0.25, 0.75);
	scene[3].emm = newcolor(0, 0, 0);

	scene[4].pos = newvec3(0, 5 + 1000, 0); // top
	scene[4].rad = 1000;
	scene[4].diffuse = newcolor(0.75, 0.75, 0.75);
	scene[4].emm = newcolor(0, 0, 0);

	scene[5].pos = newvec3(-3, -1.6f, -15); // ball 1
	scene[5].rad = 1.5;
	scene[5].diffuse = newcolor(0.15, 0.15, 0.4);
	scene[5].emm = newcolor(0, 0, 0);

	scene[6].pos = newvec3(3, -1.6f, -11); // ball 2
	scene[6].rad = 1.5;
	scene[6].diffuse = newcolor(0.8, 0.8, 0.8);
	scene[6].emm = newcolor(0, 0, 0);

	scene[7].pos = newvec3(0, 19, -11); // light
	scene[7].rad = 14.2;
	scene[7].diffuse = newcolor(0, 0, 0);
	scene[7].emm = newcolor(60, 60, 60);

	return scene;
}*/

int main()
{
	// load shit
	loadOBJ("models/CornellBox-Original.obj", vec3(), 1);
	//loadOBJ("models/my_cornell.obj", vec3(0, 0, 2), 1);
	buildBVH();

	//int blocksize = 512;
	//int nblocks = TILE_SIZE / blocksize + (TILE_SIZE % blocksize == 0 ? 0 : 1);
	int blocksize = 512;
	int nblocks = 4;

	// setup the random number generator
	curandState* randState_device;
	cudaMalloc(&randState_device, IMAGE_SIZE * sizeof(curandState));
	setupCurand <<< nblocks, blocksize >>>(randState_device);

	cudaDeviceSynchronize();

	cudaFree(randState_device);

	// set up the camera
	/*camera cam;
	cam.pos = vec3(0, 1, 3);
	cam.distFromFilm = 1;
	cam.focalLength = 3;
	cam.radius = 0.0;
	camera* cam_device;
	printf("camera size: %zd", sizeof(camera));
	cudaMalloc((void**)&cam_device, sizeof(camera));
	cudaMemcpy(cam_device, &cam, sizeof(camera), cudaMemcpyHostToDevice);

	cudaDeviceSynchronize();

	/*cudaDeviceSynchronize();
	checkError();
	printf("here?\n");
	
	// setup host image buffer
	color* imgBuffer_host = (color*)malloc(IMAGE_SIZE * sizeof(color));

	// setup device image buffer
	color* imgBuffer_device;
	cudaMalloc((void**)&imgBuffer_device, IMAGE_SIZE * sizeof(color));
	setupImgBuffer <<< nblocks, blocksize >>>(imgBuffer_device);

	// setup path state buffer
	pathState* pathStateBuffer_device;
	cudaMalloc((void**)&pathStateBuffer_device, IMAGE_SIZE * sizeof(pathState));
	setupPathStateBuffer <<< nblocks, blocksize >>>(pathStateBuffer_device, cam_device, randState_device);

	// vertex buffer
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
		while (sampleNum <= NUM_SAMPLES)
		{
			drawPixel << < nblocks, blocksize >> >(imgBuffer_device, pathStateBuffer_device, scene_device, cam_device, randState_device);

			if (sampleNum % 10 == 0)
				printf("sample %d finished\n", sampleNum);
			cudaDeviceSynchronize();

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
			int idx = y * IMAGE_WIDTH + x;
			color c = imgBuffer_host[idx];
			fprintf(fp, "%d %d %d ", (int)(c.r * 255), (int)(c.g * 255), (int)(c.b * 255));
		}
	}
	fclose(fp);*/

	printf("\nfinished");

	getchar();

	return 0;
}
#pragma once

#include <float.h>
#include <math.h>
#include <limits>
#include <queue>

#include "modelLoader.h"
#include "vec3.h"

struct AABB
{
	vec3 lo, hi;

	AABB()
		: lo(vec3()), hi(vec3())
	{ }

	void makeNegative()
	{
		lo = vec3(10000, 10000, 10000);
		hi = vec3(-10000, -10000, -10000);
	}

	float weight()
	{
		vec3 diff = hi - lo;
		return 2 * (diff.x * diff.y + diff.x * diff.z + diff.y * diff.z);
	}
};
void AABBUnion(AABB* ret, AABB* b1, AABB* b2)
{
	ret->lo = min(b1->lo, b2->lo);
	ret->hi = max(b1->hi, b2->hi);
}
float BVHweight(AABB b1, AABB b2)
{
	AABB u;
	AABBUnion(&u, &b1, &b2);
	vec3 diff = u.hi - u.lo;
	return 2 * (diff.x * diff.y + diff.x * diff.z + diff.y * diff.z);
}
__device__ void swap(float& a, float& b)
{
	float c = a;
	a = b;
	b = c;
}
__device__ float rayAABBIntersect(vec3 o, vec3 ray, AABB b)
{
	/*float tmin = -MAX_FLOAT;
	float tmax = MAX_FLOAT;

	if (ray.x != 0)
	{
		float tx1 = (b.lo.x - o.x) / ray.x;
		float tx2 = (b.hi.x - o.x) / ray.x;

		tmin = max(tmin, min(tx1, tx2));
		tmax = min(tmax, max(tx1, tx2));
	}

	if (ray.y != 0)
	{
		float ty1 = (b.lo.y - o.y) / ray.y;
		float ty2 = (b.hi.y - o.y) / ray.y;

		tmin = max(tmin, min(ty1, ty2));
		tmax = min(tmax, max(ty1, ty2));
	}

	if (ray.z != 0)
	{
		float tz1 = (b.lo.z - o.z) / ray.z;
		float tz2 = (b.hi.z - o.z) / ray.z;

		tmin = max(tmin, min(tz1, tz2));
		tmax = min(tmax, max(tz1, tz2));
	}

	if (tmax <= tmin)
		return MAX_FLOAT;

	return tmin;*/

	float tmin = (b.lo.x - o.x) / ray.x;
	float tmax = (b.hi.x - o.x) / ray.x;

	if (tmin > tmax) swap(tmin, tmax);

	float tymin = (b.lo.y - o.y) / ray.y;
	float tymax = (b.hi.y - o.y) / ray.y;

	if (tymin > tymax) swap(tymin, tymax);
	
	if ((tmin > tymax) || (tymin > tmax))
		return MAX_FLOAT;

	if (tymin > tmin)
		tmin = tymin;
	if (tymax < tmax)
		tmax = tymax;

	float tzmin = (b.lo.z - o.z) / ray.z;
	float tzmax = (b.hi.z - o.z) / ray.z;

	if (tzmin > tzmax) swap(tzmin, tzmax);

	if ((tmin > tzmax) || (tzmin > tmax))
		return MAX_FLOAT;

	if (tzmin > tmin)
		tmin = tzmin;
	if (tzmax < tmax)
		tmax = tzmax;

	return tmin;
}

struct BVH_node
{
	AABB box;
	
	BVH_node *left, *right;
	int numChildNodes;

	int target;
	int depth;

	BVH_node()
	{
		box = AABB();
		left = right = 0;
		numChildNodes = 0;
		target = -1;
		depth = 0;
	}
	~BVH_node()
	{
		if (left != 0)
			delete left;
		if (right != 0)
			delete right;
	}
};
struct BVH_array_node
{
	AABB box;
	int32_t left, right;
};
struct BVH_array
{
	BVH_array_node* root;
	int size;
	int depth;
};
BVH_node* createEmptyBVH_node()
{
	BVH_node* ret = new BVH_node();
	ret->box = AABB();
	ret->left = ret->right = 0;

	ret->target = -1;
	ret->depth = 0;

	return ret;
}
BVH_node* buildBVHRecurse(BVH_node* nodes, int* workingList, const int numNodes)
{
	// if it's just 1 or 2, just put them in a bvh
	if (numNodes == 2)
	{
		AABB total = AABB();
		total.makeNegative();
		AABBUnion(&total, &total, &(nodes[workingList[0]].box));
		AABBUnion(&total, &total, &(nodes[workingList[1]].box));

		BVH_node* ret = createEmptyBVH_node();
		ret->box = total;
		ret->left = &nodes[workingList[0]];
		ret->right = &nodes[workingList[1]];
		ret->numChildNodes = 2;
		ret->depth = 2;
		return ret;
	}
	if (numNodes == 1)
	{
		/*BVH_node* ret = createEmptyBVH_node();
		ret->left = &nodes[workingList[0]];
		return ret;*/
		return &nodes[workingList[0]];
	}

	// find the extents of the nodes given
	AABB total = AABB();
	total.makeNegative();
	for (int i = 0; i < numNodes; ++i)
	{
		AABBUnion(&total, &total, &(nodes[workingList[i]].box));
	}

	// build the grid
	const int gridDim = 8;
	AABB grid[gridDim][gridDim][gridDim];
	int countGrid[gridDim][gridDim][gridDim];
	for (int i = 0; i < gridDim; ++i)
	{
		for (int j = 0; j < gridDim; ++j)
		{
			for (int k = 0; k < gridDim; ++k)
			{
				grid[i][j][k].makeNegative();
				countGrid[i][j][k] = 0;
			}
		}
	}

	// fill in the grid
	vec3 dimUnits = (total.hi - total.lo) / gridDim;
	for (int i = 0; i < numNodes; ++i)
	{
		vec3 center = ((nodes[workingList[i]].box.hi + nodes[workingList[i]].box.lo) / 2 - total.lo);
		int cx = (int)min(gridDim - 1, max(0, (int)(center.x / dimUnits.x)));
		int cy = (int)min(gridDim - 1, max(0, (int)(center.y / dimUnits.y)));
		int cz = (int)min(gridDim - 1, max(0, (int)(center.z / dimUnits.z)));

		AABBUnion(&grid[cx][cy][cz], &grid[cx][cy][cz], &nodes[i].box);
		countGrid[cx][cy][cz] += 1;
	}

	// debug
	/*for (int j = 0; j < 8; ++j)
	{
		for (int i = 0; i < 8; ++i)
		{
			int total = 0;
			for (int k = 0; k < 8; ++k)
			{
				total += countGrid[i][j][k];
			}
			printf("%4d,", total);
		}
		printf("\n");
	}
	printf("\n");*/

	int bestSlice = 0;
	int bestAxis = 0;
	double bestScore = DBL_MAX;
	int bestLeftCount = 0;
	int bestRightCount = 0;

	for (int axis = 0; axis < 3; ++axis)
	{
		for (int slice = 0; slice < gridDim; ++slice)
		{
			vec3 leftHi = vec3(gridDim, gridDim, gridDim);
			leftHi[axis] = slice;

			AABB left;
			left.makeNegative();
			int countLeft = 0;
			for (int i = 0; i < leftHi.x; ++i)
			{
				for (int j = 0; j < leftHi.y; ++j)
				{
					for (int k = 0; k < leftHi.z; ++k)
					{
						if (countGrid[i][j][k] > 0)
						{
							countLeft += countGrid[i][j][k];
							AABBUnion(&left, &left, &grid[i][j][k]);
						}
					}
				}
			}

			vec3 rightLo = vec3(0, 0, 0);
			rightLo[axis] = slice;

			AABB right;
			right.makeNegative();
			int countRight = 0;
			for (int i = rightLo.x; i < gridDim; ++i)
			{
				for (int j = rightLo.y; j < gridDim; ++j)
				{
					for (int k = rightLo.z; k < gridDim; ++k)
					{
						if (countGrid[i][j][k] > 0)
						{
							countRight += countGrid[i][j][k];
							AABBUnion(&right, &right, &grid[i][j][k]);
						}
					}
				}
			}

			double score = countLeft * left.weight() + countRight * right.weight();
			//float score = left.weight() + right.weight();
			if (score < bestScore)
			{
				bestSlice = slice;
				bestAxis = axis;
				bestScore = score;
				bestLeftCount = countLeft;
				bestRightCount = countRight;
			}
		}
	}

	// in certain cases, there will be no slice that cuts more optimally than not cutting them at all
	// well too fucking bad we're doing it anyways
	if (bestLeftCount == 0)
	{
		int leftCount = bestRightCount / 2;
		int rightCount = bestRightCount - leftCount;

		int* leftList = new int[leftCount];
		int* rightList = new int[rightCount];
		for (int i = 0; i < leftCount; ++i)
			leftList[i] = workingList[i];
		for (int i = 0; i < rightCount; ++i)
			rightList[i] = workingList[leftCount + i];

		BVH_node* leftNode = buildBVHRecurse(nodes, leftList, leftCount);
		BVH_node* rightNode = buildBVHRecurse(nodes, rightList, rightCount);

		delete[] leftList;
		delete[] rightList;

		BVH_node* ret = createEmptyBVH_node();
		ret->box = total;
		ret->left = leftNode;
		ret->right = rightNode;
		ret->numChildNodes = ret->left->numChildNodes + ret->right->numChildNodes + 2;
		ret->depth = max(leftNode->depth, rightNode->depth) + 1;
		return ret;
	}

	int* leftList = new int[bestLeftCount];
	int leftCount = 0;
	int* rightList = new int[bestRightCount];
	int rightCount = 0;

	for (int i = 0; i < numNodes; ++i)
	{
		vec3 center = ((nodes[workingList[i]].box.hi + nodes[workingList[i]].box.lo) / 2 - total.lo);
		center.x = min(gridDim - 1, max(0, (int)(center.x / dimUnits.x)));
		center.y = min(gridDim - 1, max(0, (int)(center.y / dimUnits.y)));
		center.z = min(gridDim - 1, max(0, (int)(center.z / dimUnits.z)));

		if (center[bestAxis] < bestSlice)
		{
			//leftList[leftCount] = workingList[i];
			leftList[leftCount] = workingList[i];
			++leftCount;
		}
		else
		{
			rightList[rightCount] = workingList[i];
			++rightCount;
		}
	}

	BVH_node* leftNode = buildBVHRecurse(nodes, leftList, bestLeftCount);
	BVH_node* rightNode = buildBVHRecurse(nodes, rightList, bestRightCount);

	delete[] leftList;
	delete[] rightList;

	BVH_node* ret = createEmptyBVH_node();
	ret->box = total;
	ret->left = leftNode;
	ret->right = rightNode;
	ret->numChildNodes = ret->left->numChildNodes + ret->right->numChildNodes + 2;
	ret->depth = max(leftNode->depth, rightNode->depth) + 1;
	return ret;
}

BVH_array BVHTreeToArray(BVH_node* root)
{
	std::queue<BVH_node*> line;
	line.push(root);

	int32_t arraySize = (root->numChildNodes + 1) / 2;
	//int arraySize = (root->numChildNodes + 1);

	BVH_array ret;
	ret.root = new BVH_array_node[arraySize];
	ret.size = arraySize;
	ret.depth = root->depth;

	int32_t counter = 0;

	while (!line.empty())
	{
		BVH_node* cur = line.front();
		line.pop();
		if (cur->left != 0)
		{
			if (cur->left->target == -1)
			{
				line.push(cur->left);
				ret.root[counter].left = counter + line.size();
			}
			else
				ret.root[counter].left = -cur->left->target - 1;
		}
		if (cur->right != 0)
		{
			if (cur->right->target == -1)
			{
				line.push(cur->right);
				ret.root[counter].right = counter + line.size();
			}
			else
				ret.root[counter].right = -cur->right->target - 1;
		}

		ret.root[counter].box = cur->box;
		++counter;
	}

	printf("array size: %d\n", root->numChildNodes + 1);
	printf("actual array size: %d\n", counter);
	return ret;
}

BVH_array buildBVH()
{
	printf("Adding triangles to BVH\n");

	BVH_node* allNodes = new BVH_node[tris.size()];
	int* workingList = new int[tris.size()];

	// add triangles to BVH
	for (int i = 0; i < tris.size(); ++i)
	{
		AABB b = AABB();
		b.lo = min(min(verts[tris[i].v0], verts[tris[i].v1]), verts[tris[i].v2]);
		b.hi = max(max(verts[tris[i].v0], verts[tris[i].v1]), verts[tris[i].v2]);

		allNodes[i] = BVH_node();
		allNodes[i].box = b;
		allNodes[i].target = i;

		workingList[i] = i;
	}

	printf("Building the BVH\n");
	BVH_node* root = buildBVHRecurse(allNodes, workingList, tris.size());

	BVH_array ret = BVHTreeToArray(root);

	/*delete root;
	delete[] allNodes;
	delete[] workingList;*/

	return ret;
}

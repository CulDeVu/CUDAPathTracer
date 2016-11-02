#pragma once

#include <math.h>
#include <limits>
#include <queue>

#include "modelLoader.h"
#include "vec3.h"

struct AABB
{
	/*float x1, x2,
		y1, y2,
		z1, z2;*/
	vec3 lo, hi;

	AABB()
		: lo(vec3()), hi(vec3())
	{ }

	void makeNegative()
	{
		lo = vec3(1, 1, 1);
		hi = vec3(-1, -1, -1);
	}

	float weight()
	{
		vec3 diff = hi - lo;
		return 2 * (diff.x * diff.y + diff.x * diff.z + diff.y * diff.z);
	}
};
void AABBUnion(AABB* ret, AABB* b1, AABB* b2)
{
	/*ret->x1 = min(b1->x1, b2->x1);
	ret->x2 = max(b1->x2, b2->x2);
	ret->y1 = min(b1->y1, b2->y1);
	ret->y2 = max(b1->y2, b2->y2);
	ret->z1 = min(b1->z1, b2->z1);
	ret->z2 = max(b1->z2, b2->z2);*/
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
__device__ float rayAABBIntersect(vec3 o, vec3 ray, AABB b)
{
	float tmin = -MAX_FLOAT;
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

	if (tmax >= tmin)
		return MAX_FLOAT;

	return tmin;
}

struct BVH_node
{
	AABB box;
	
	BVH_node *left, *right;
	int numChildNodes;

	int target;

	BVH_node()
	{
		box = AABB();
		left = right = 0;
		numChildNodes = 0;
		target = -1;
	}
};
struct BVH_array_node
{
	AABB box;
	int left, right;
	int target;
};
BVH_node* createEmptyBVH_node()
{
	BVH_node* ret = new BVH_node();
	ret->box = AABB();
	ret->left = ret->right = 0;

	ret->target = -1;

	return ret;
}
BVH_node* buildBVHRecurse(BVH_node* nodes, int* workingList, const int numNodes)
{
	// if it's just 1 or 2, just put them in a bvh
	if (numNodes == 2)
	{
		BVH_node* ret = createEmptyBVH_node();
		ret->left = &nodes[workingList[0]];
		ret->right = &nodes[workingList[1]];
		ret->numChildNodes = 2;
		return ret;
	}
	if (numNodes == 1)
	{
		/*BVH_node* ret = createEmptyBVH_node();
		ret->left = &nodes[workingList[0]];
		return ret;*/
		return &nodes[workingList[0]];
	}
	printf("\n\nnumfaces: %d\n", numNodes);

	// find the extents of the nodes given
	AABB total = AABB();
	total.makeNegative();
	for (int i = 0; i < numNodes; ++i)
	{
		AABBUnion(&total, &total, &(nodes[workingList[i]].box));
	}
	printf("width of everything: %f\n", total.hi.x - total.lo.x);

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
		int cx = min(gridDim - 1, max(0, (int)(center.x / dimUnits.x)));
		int cy = min(gridDim - 1, max(0, (int)(center.y / dimUnits.y)));
		int cz = min(gridDim - 1, max(0, (int)(center.z / dimUnits.z)));

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
	float bestScore = -1;
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

			float score = countLeft * left.weight() + countRight * right.weight();
			//float score = left.weight() + right.weight();
			if (score < bestScore || bestScore < 0)
			{
				bestSlice = slice;
				bestAxis = axis;
				bestScore = score;
				bestLeftCount = countLeft;
				bestRightCount = countRight;
				printf("t %d %d %f %d %d\n", bestSlice, bestAxis, bestScore, bestLeftCount, bestRightCount);
			}
		}
	}
	//printf("slice: %d %d\n", bestAxis, bestSlice);

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
			leftList[leftCount] = i;
			++leftCount;
		}
		else
		{
			rightList[rightCount] = i;
			++rightCount;
		}
	}

	BVH_node* leftNode = buildBVHRecurse(nodes, leftList, bestLeftCount);
	BVH_node* rightNode = buildBVHRecurse(nodes, rightList, bestRightCount);

	delete[] leftList;
	delete[] rightList;

	BVH_node* ret = createEmptyBVH_node();
	ret->left = leftNode;
	ret->right = rightNode;
	ret->numChildNodes = ret->left->numChildNodes + ret->right->numChildNodes + 2;
	return ret;
}

BVH_node* BVHTreeToArray(BVH_node* root)
{
	std::queue<BVH_node*> line;
	line.push(root);

	BVH_array_node* array = new BVH_array_node[root->numChildNodes + 1];
	int counter = 0;

	while (!line.empty())
	{
		BVH_node* cur = line.front();
		line.pop();
		if (cur->left != 0)
			line.push(cur->left);
		if (cur->right != 0)
			line.push(cur->right);

		array[counter].box = cur->box;
		array[counter].left = counter + line.size() - 1;
		array[counter].right = counter + line.size();
		array[counter].target = cur->target;
		++counter;
	}

	printf("array size: %d\n", root->numChildNodes + 1);
	return 0;
}

void buildBVH()
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

	BVHTreeToArray(root);
}

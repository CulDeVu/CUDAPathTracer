#pragma once

#include "BVH.h"

BVH_node* buildBVHRecurseTest(BVH_node* nodes, int* workingList, const int numNodes, uint32_t* test)
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
		return &nodes[workingList[0]];
	}

	// find the extents of the nodes given
	AABB total = AABB();
	total.makeNegative();
	for (int i = 0; i < numNodes; ++i)
	{
		AABBUnion(&total, &total, &(nodes[workingList[i]].box));
	}
	float totalWeight = total.weight();

	// build the grid
	const int gridDim = 3;
	uint32_t grid[gridDim][gridDim][gridDim];
	int countGrid[gridDim][gridDim][gridDim];
	for (int i = 0; i < gridDim; ++i)
	{
		for (int j = 0; j < gridDim; ++j)
		{
			for (int k = 0; k < gridDim; ++k)
			{
				grid[i][j][k] = 0;
				countGrid[i][j][k] = 0;
			}
		}
	}

	// fill in the grid
	vec3 dimUnits = (total.hi - total.lo).toVec3() / gridDim;
	for (int i = 0; i < numNodes; ++i)
	{
		vec3 center = ((nodes[workingList[i]].box.hi + nodes[workingList[i]].box.lo) / 2 - total.lo).toVec3();
		int cx = (int)min(gridDim - 1, max(0, (int)(center.x / dimUnits.x)));
		int cy = (int)min(gridDim - 1, max(0, (int)(center.y / dimUnits.y)));
		int cz = (int)min(gridDim - 1, max(0, (int)(center.z / dimUnits.z)));

		uint32_t count = min(MAX_UINT32_T - 1, test[workingList[i]]);
		uint32_t oldcount = grid[cx][cy][cz];
		uint32_t newcount = oldcount + count;
		if (newcount < oldcount)
			newcount = MAX_UINT32_T;

		grid[cx][cy][cz] = newcount;
		countGrid[cx][cy][cz] += 1;
	}

	int bestSlice = 0;
	int bestAxis = 0;
	double bestScore = DBL_MAX;
	int bestLeftCount = 0;
	int bestRightCount = 0;

	for (int axis = 0; axis < 3; ++axis)
	{
		for (int slice = 0; slice < gridDim; ++slice)
		{
			ivec3 leftHi = ivec3(gridDim, gridDim, gridDim);
			leftHi[axis] = slice;

			uint32_t left;
			int countLeft = 0;
			for (int i = 0; i < leftHi.x; ++i)
			{
				for (int j = 0; j < leftHi.y; ++j)
				{
					for (int k = 0; k < leftHi.z; ++k)
					{
						countLeft += countGrid[i][j][k];
						uint32_t sum = grid[i][j][k] + left;
						if (sum < grid[i][j][k])
							left = MAX_UINT32_T;
					}
				}
			}

			ivec3 rightLo = ivec3(0, 0, 0);
			rightLo[axis] = slice;

			uint32_t right;
			int countRight = 0;
			for (int i = rightLo.x; i < gridDim; ++i)
			{
				for (int j = rightLo.y; j < gridDim; ++j)
				{
					for (int k = rightLo.z; k < gridDim; ++k)
					{
						countRight += countGrid[i][j][k];
						uint32_t sum = grid[i][j][k] + right;
						if (sum < grid[i][j][k])
							right = MAX_UINT32_T;
					}
				}
			}

			double leftProb = left;
			double rightProb = right;
			double score = countLeft * leftProb + countRight * rightProb;

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
	if (bestLeftCount == 0 || bestRightCount == 0)
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
		vec3 center = ((nodes[workingList[i]].box.hi + nodes[workingList[i]].box.lo) / 2 - total.lo).toVec3();
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

BVH_array buildBVHTest(uint32_t* test)
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
	BVH_node* root = buildBVHRecurseTest(allNodes, workingList, tris.size(), test);

	BVH_array ret = BVHTreeToArrayBreadthFirst(root, (uint32_t)tris.size());

	/*delete root;
	delete[] allNodes;
	delete[] workingList;*/

	return ret;
}
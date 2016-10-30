#pragma once

#include <math.h>

#include "vec3.h"

struct AABB
{
	float x1, x2,
		y1, y2,
		z1, z2;

	AABB()
		: x1(0), x2(0), y1(0), y2(0), z1(0), z2(0)
	{ }
};
void AABBUnion(AABB* ret, AABB* b1, AABB* b2)
{
	ret->x1 = min(b1->x1, b2->x1);
	ret->x2 = max(b1->x2, b2->x2);
	ret->y1 = min(b1->y1, b2->y1);
	ret->y2 = max(b1->y2, b2->y2);
	ret->z1 = min(b1->z1, b2->z1);
	ret->z2 = max(b1->z2, b2->z2);
}
float BVHweight(AABB b1, AABB b2)
{
	AABB u;
	AABBUnion(&u, &b1, &b2);
	float l = u.x2 - u.x1;
	float w = u.y2 - u.y1;
	float h = u.z2 - u.z1;
	return l * w * h;
}
float rayAABBIntersect(vec3 o, vec3 ray, AABB b)
{
	float tmin = -MAX_FLOAT;
	float tmax = MAX_FLOAT;

	if (ray.x != 0)
	{
		float tx1 = (b.x1 - o.x) / ray.x;
		float tx2 = (b.x2 - o.x) / ray.x;

		tmin = max(tmin, min(tx1, tx2));
		tmax = min(tmax, max(tx1, tx2));
	}

	if (ray.y != 0)
	{
		float ty1 = (b.y1 - o.y) / ray.y;
		float ty2 = (b.y2 - o.y) / ray.y;

		tmin = max(tmin, min(ty1, ty2));
		tmax = min(tmax, max(ty1, ty2));
	}

	if (ray.z != 0)
	{
		float tz1 = (b.z1 - o.z) / ray.z;
		float tz2 = (b.z2 - o.z) / ray.z;

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
	
	BVH_node* parent;
	BVH_node *left, *right;

	int target;
};
BVH_node* createEmptyBVH_node()
{
	BVH_node* ret = new BVH_node();
	ret->box = AABB();
	ret->parent = ret->left = ret->right = 0;

	ret->target = -1;

	return ret;
}
void addToBVH(BVH_node* root, BVH_node* n)
{
	if (root->left == 0)
	{
		root->left = n;
		n->parent = root;
		return;
	}
	if (root->right == 0)
	{
		root->right = n;
		n->parent = root;
		return;
	}

	BVH_node* cur = root;
	float w1, w2;
	while (cur->target == -1)
	{
		w1 = BVHweight(n->box, cur->left->box);
		w2 = BVHweight(n->box, cur->right->box);

		if (w1 > w2)
		{
			cur = cur->left;
		}
		else
		{
			cur = cur->right;
		}
	}

	BVH_node* mid = createEmptyBVH_node();
	mid->parent = cur->parent;
	mid->left = cur;
	mid->right = n;
	if (w1 > w2)
		cur->parent->left = mid;
	else
		cur->parent->right = mid;
	cur->parent = mid;
	n->parent = mid;

	// fix AABBs
	cur = cur->parent;
	while (cur != 0)
	{
		if (cur->left == 0)
			cur->box = cur->right->box;
		else if (cur->right == 0)
			cur->box = cur->left->box;
		else
			AABBUnion(&(cur->box), &(cur->left->box), &(cur->right->box));

		cur = cur->parent;
	}
}

/*void addToBVH(BVH_node* root, BVH_node* n)
{
	BVH_node* cur = root;
	while (cur->left != 0)
		cur = cur->left;

	cur->left = n;
}

void addToBVH_recursive(BVH_node* cur, BVH_node* n)
{
	BVH_node** addTo = 0;

	float r = rand() % 2;
	if (r == 0)
	{
		if (cur->left == 0)
			cur->left = n;
		else
			addToBVH_recursive(cur->left, n);
	}
	else
	{
		if (cur->right == 0)
			cur->right = n;
		else
			addToBVH_recursive(cur->right, n);
	}

	// fix the AABBs
	if (cur->right == 0)
	{
		cur->x1 = cur->left->x1;
		cur->x2 = cur->left->x2;
		cur->y1 = cur->left->y1;
		cur->y2 = cur->left->y2;
		cur->z1 = cur->left->z1;
		cur->z2 = cur->left->z2;
	}
	else if (cur->left == 0)
	{
		cur->x1 = cur->right->x1;
		cur->x2 = cur->right->x2;
		cur->y1 = cur->right->y1;
		cur->y2 = cur->right->y2;
		cur->z1 = cur->right->z1;
		cur->z2 = cur->right->z2;
	}
	else
	{
		cur->x1 = min(cur->left->x1, cur->right->x1);
		cur->x2 = max(cur->left->x2, cur->right->x2);
		cur->y1 = min(cur->left->y1, cur->right->y1);
		cur->y2 = max(cur->left->y2, cur->right->y2);
		cur->z1 = min(cur->left->z1, cur->right->z1);
		cur->z2 = max(cur->left->z2, cur->right->z2);
	}
}*/
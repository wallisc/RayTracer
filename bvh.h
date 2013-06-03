#ifndef BVH_H
#define BVH_H
#include "Geometry.h"
#include "Plane.h"
#include "Box.h"
#include "GeometryUtil.h"
#include "Util.h"
#include "kernel.h"

typedef struct BVHNode {
   BVHNode *left, *right;
   Geometry *geom;
   BoundingBox bb;

   __device__ BVHNode() : left(NULL), right(NULL), geom(NULL) {}
   __device__ BVHNode(Geometry *object) : left(NULL), right(NULL), geom(object) {
      bb = geom->getBoundingBox();
   }
} BVHNode;

typedef struct BVHTree {
   BVHNode *root;
   Plane **planeList;
   int planeListSize;
} BVHTree;

void formBVH(Geometry *dGeomList[], int geomCount, Plane *planeList[], 
      int planeCount, BVHTree *dTree);

#endif //BVH_H

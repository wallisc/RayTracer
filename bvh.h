#include "Geometry.h"
#include "Plane.h"
#include "Box.h"
#include "GeometryUtil.h"
#include "Util.h"




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

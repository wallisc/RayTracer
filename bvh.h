#include "Geometry.h"
#include "Plane.h"
#include "Box.h"
#include "GeometryUtil.h"
#include "Util.h"


const int kXAxis = 0, kYAxis = 1, kZAxis = 2;
const int kAxisNum = 3;


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

typedef struct BVHStackEntry {
   BVHNode *cursor;
   Geometry **arr;
   int listSize;
   int axis;
   __device__ BVHStackEntry() {}
   __device__ BVHStackEntry(Geometry **nArr, BVHNode *nCursor, int nListSize, int nAxis) : 
      arr(nArr), cursor(nCursor), listSize(nListSize), axis(nAxis) {}
} BVHStackEntry;


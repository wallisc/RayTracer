#include "Geometry.h"
#include "Box.h"
#include "Util.h"


const int kXAxis = 0, kYAxis = 1, kZAxis = 2;
const int kAxisNum = 3;

__device__ inline void cudaSort(Geometry *list[], int end, int axis) {
   bool swapMade = false;
   for (int i = 0; i < end; i++) {
      for (int j = 0; j < end - i; j++) {
         if (list[j]->getCenter()[axis] > list[j+1]->getCenter()[axis]) {
            SWAP(list[j], list[j+1]);
            swapMade = true;
         }
      }
      if (!swapMade) break;
      swapMade = false;
   }
}

typedef struct BVHNode {
   BVHNode *left, *right;
   Geometry *geom;
   __device__ BVHNode() : left(NULL), right(NULL) {}
   __device__ BVHNode(Geometry *object) : left(NULL), right(NULL), geom(object) {}
} BVHNode;

typedef struct BVHStackEntry {
   Geometry **arr;
   BVHNode *cursor;
   int listSize;
   int axis;
   __device__ BVHStackEntry() {}
   __device__ BVHStackEntry(Geometry **nArr, BVHNode *nCursor, int nListSize, int nAxis) : 
      arr(nArr), cursor(nCursor), listSize(nListSize), axis(nAxis) {}
} BVHStackEntry;


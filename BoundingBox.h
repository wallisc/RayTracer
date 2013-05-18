#ifndef BOUNDING_BOX_H
#define BOUNDING_BOX_H

#include "glm/glm.hpp"
#include "GeometryUtil.h"
#include <cfloat>

const int kPointsPerBox = 8;

typedef struct BoundingBox {
   __device__ BoundingBox(glm::vec3 nMin = glm::vec3(0.0f), glm::vec3 nMax = glm::vec3(0.0f)) 
      : min(nMin), max(nMax) {
         min -= EPSILON;
         max += EPSILON;
      }

   __device__ float getIntersection(const Ray &ray) {
      return boxIntersect(ray, min, max);
   }

   glm::vec3 min, max;
} BoundingBox;

__device__ inline BoundingBox generateBoundingBox(const glm::vec3 &min, const glm::vec3 &max, 
                           const glm::mat4 &trans) {
   glm::vec4 points[kPointsPerBox];
   glm::vec3 transMin(FLT_MAX);
   glm::vec3 transMax(-FLT_MAX);
   points[0] = glm::vec4(min.x, min.y, min.z, 1.0f);
   points[1] = glm::vec4(max.x, min.y, min.z, 1.0f);
   points[2] = glm::vec4(min.x, max.y, min.z, 1.0f);
   points[3] = glm::vec4(max.x, max.y, min.z, 1.0f);
   points[4] = glm::vec4(min.x, min.y, max.z, 1.0f);
   points[5] = glm::vec4(max.x, min.y, max.z, 1.0f);
   points[6] = glm::vec4(min.x, max.y, max.z, 1.0f);
   points[7] = glm::vec4(max.x, max.y, max.z, 1.0f);

   for (int i = 0; i < kPointsPerBox; i++) {
      points[i] = trans * points[i]; 
      for (int axis = 0; axis < 3; axis++) {
         if (points[i][axis] < transMin[axis]) transMin[axis] = points[i][axis];
         if (points[i][axis] > transMax[axis]) transMax[axis] = points[i][axis];
      }
   }
   return BoundingBox(transMin, transMax);
}

__device__ inline BoundingBox combineBoundingBox(const BoundingBox &bb1, 
                                                 const BoundingBox &bb2) {
   return BoundingBox(getSmallestBoxCorner(bb1.min, bb2.min),
                      getBiggestBoxCorner(bb1.max, bb2.max));
}

#endif //BOUNDING_BOX_H

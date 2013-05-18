#ifndef GEOM_UTIL_H
#define GEOM_UTIL_H

#include "Ray.h" 
#include "Util.h" 
#include "glm/glm.hpp"
#include <cfloat> 

__device__ inline float boxIntersect(const Ray &ray, const glm::vec3 &minP, const glm::vec3 &maxP) {
   float tXmin, tYmin, tZmin;
   float tXmax, tYmax, tZmax;

   tXmin = tYmin = tZmin = -FLT_MAX;
   tXmax = tYmax = tZmax = FLT_MAX;

   if (!isFloatZero(ray.d.x)) {
      tXmin = (minP.x - ray.o.x) / ray.d.x;
      tXmax = (maxP.x - ray.o.x) / ray.d.x;
   } else if (ray.o.x < minP.x || ray.o.x > maxP.x) {
      return -1.0f;
   }

   if (!isFloatZero(ray.d.y)) {
      tYmin = (minP.y - ray.o.y) / ray.d.y;
      tYmax = (maxP.y - ray.o.y) / ray.d.y;
   } else if (ray.o.y < minP.y || ray.o.y > maxP.y) {
      return -1.0f;
   }

   if (!isFloatZero(ray.d.z)) {
      tZmin = (minP.z - ray.o.z) / ray.d.z;
      tZmax = (maxP.z - ray.o.z) / ray.d.z;
   } else if (ray.o.z < minP.z || ray.o.z > maxP.z) {
      return -1.0f;
   }

   if (tXmin > tXmax) SWAP(tXmin, tXmax);
   if (tYmin > tYmax) SWAP(tYmin, tYmax);
   if (tZmin > tZmax) SWAP(tZmin, tZmax);

   float tBiggestMin = mymax(mymax(tXmin, tYmin), tZmin);
   float tSmallestMax = mymin(mymin(tXmax, tYmax), tZmax);

   // If the ray is in the box
   if (ray.o.x > minP.x && ray.o.x < maxP.x && ray.o.y > minP.y 
       && ray.o.y < maxP.y && ray.o.z > minP.z && ray.o.z < maxP.z) {
      return tSmallestMax;
   } else {
      return tBiggestMin < tSmallestMax ? tBiggestMin : -1.0f;
   }
}

__device__ inline glm::vec3 getSmallestBoxCorner(const glm::vec3 &p1, const glm::vec3 &p2) {
   return glm::vec3(mymin(p1.x, p2.x), mymin(p1.y, p2.y), mymin(p1.z, p2.z));
}

__device__ inline glm::vec3 getBiggestBoxCorner(const glm::vec3 &p1, const glm::vec3 &p2) {
   return glm::vec3(mymax(p1.x, p2.x), mymax(p1.y, p2.y), mymax(p1.z, p2.z));
}

#endif //GEOM_UTIL_H

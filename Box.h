#ifndef BOX_H
#define BOX_H

#include "Geometry.h"
#include "Util.h"
#include "Material.h"
#include "glm/glm.hpp"
#include <cfloat>

class Box : public Geometry {
public:
   __device__ Box(const glm::vec3 p1, const glm::vec3 p2, 
         const Material &mat, const glm::mat4 trans, 
         const glm::mat4 & invTrans) :
         Geometry(mat, trans, invTrans) {
      xMin = min(p1.x, p2.x); 
      xMax = max(p1.x, p2.x);
      yMin = min(p1.y, p2.y);
      yMax = max(p1.y, p2.y);
      zMin = min(p1.z, p2.z);
      zMax = max(p1.z, p2.z);
   }

   __device__ virtual glm::vec3 getCenter() const { 
      glm::vec4 objSpaceCenter((xMin + xMax) / 2.0f, (yMin + yMax) / 2.0f, (zMin + zMax) / 2.0f, 1.0f);
      return glm::vec3(trans * objSpaceCenter);
   }

private:
   __device__ virtual float intersects(const Ray &ray) const {
      float tXmin, tYmin, tZmin;
      float tXmax, tYmax, tZmax;

      tXmin = tYmin = tZmin = -FLT_MAX;
      tXmax = tYmax = tZmax = FLT_MAX;

      //TODO handle if the ray is in the box
      if (!isFloatZero(ray.d.x)) {
         tXmin = (xMin - ray.o.x) / ray.d.x;
         tXmax = (xMax - ray.o.x) / ray.d.x;
      } else if (ray.o.x < xMin || ray.o.x > xMax) {
         return -1.0f;
      }

      if (!isFloatZero(ray.d.y)) {
         tYmin = (yMin - ray.o.y) / ray.d.y;
         tYmax = (yMax - ray.o.y) / ray.d.y;
      } else if (ray.o.y < yMin || ray.o.y > yMax) {
         return -1.0f;
      }

      if (!isFloatZero(ray.d.z)) {
         tZmin = (zMin - ray.o.z) / ray.d.z;
         tZmax = (zMax - ray.o.z) / ray.d.z;
      } else if (ray.o.z < zMin || ray.o.z > zMax) {
         return -1.0f;
      }

      if (tXmin > tXmax) SWAP(tXmin, tXmax);
      if (tYmin > tYmax) SWAP(tYmin, tYmax);
      if (tZmin > tZmax) SWAP(tZmin, tZmax);

      float tBiggestMin = max(max(tXmin, tYmin), tZmin);
      float tSmallestMax = min(min(tXmax, tYmax), tZmax);
      
      return tBiggestMin < tSmallestMax ? tBiggestMin : -1.0f;
   }

   // Precondition: The given position is on the box 
   __device__ virtual glm::vec3 getNormalAt(const Ray &ray, float param) const {
      glm::vec3 intersectPoint = ray.transform(invTrans).getPoint(param);
      glm::vec3 norm;

      if (isFloatEqual(intersectPoint.x, xMin)) {
         norm = glm::vec3(-1.0f, 0.0f, 0.0f);
      } else if (isFloatEqual(intersectPoint.x, xMax)) {
         norm = glm::vec3(1.0f, 0.0f, 0.0f);
      } else if (isFloatEqual(intersectPoint.y, yMin)) {
         norm = glm::vec3(0.0f, -1.0f, 0.0f);
      } else if (isFloatEqual(intersectPoint.y, yMax)) {
         norm = glm::vec3(0.0f, 1.0f, 0.0f);
      } else if (isFloatEqual(intersectPoint.z, zMin)) {
         norm = glm::vec3(0.0f, 0.0f, -1.0f);
      } else {
         norm = glm::vec3(0.0f, 0.0f, 1.0f);
      }
      return glm::vec3(trans * glm::vec4(norm.x, norm.y, norm.z, 0.0f));
   }

   float xMin, yMin, zMin;
   float xMax, yMax, zMax;
};

#endif //BOX_H

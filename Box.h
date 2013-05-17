#ifndef BOX_H
#define BOX_H

#include "Geometry.h"
#include "GeometryUtil.h"
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
      minP = getSmallestBoxCorner(p1, p2);
      maxP = getBiggestBoxCorner(p1, p2);
   }

   __device__ virtual  BoundingBox getBoundingBox() const {
      generateBoundingBox(minP, maxP, trans);
   }

   __device__ virtual glm::vec3 getCenter() const { 
      glm::vec3 c = (minP + maxP) / 2.0f;
      glm::vec4 objSpaceCenter(c.x, c.y, c.z, 1.0f);
      return glm::vec3(trans * objSpaceCenter);
   }

private:
   __device__ virtual float intersects(const Ray &ray) const {
      return boxIntersect(ray, minP, maxP);
   }

   // Precondition: The given position is on the box 
   __device__ virtual glm::vec3 getNormalAt(const Ray &ray, float param) const {
      glm::vec3 intersectPoint = ray.transform(invTrans).getPoint(param);
      glm::vec3 norm;

      if (isFloatEqual(intersectPoint.x, minP.x)) {
         norm = glm::vec3(-1.0f, 0.0f, 0.0f);
      } else if (isFloatEqual(intersectPoint.x, maxP.x)) {
         norm = glm::vec3(1.0f, 0.0f, 0.0f);
      } else if (isFloatEqual(intersectPoint.y, minP.y)) {
         norm = glm::vec3(0.0f, -1.0f, 0.0f);
      } else if (isFloatEqual(intersectPoint.y, maxP.y)) {
         norm = glm::vec3(0.0f, 1.0f, 0.0f);
      } else if (isFloatEqual(intersectPoint.z, minP.z)) {
         norm = glm::vec3(0.0f, 0.0f, -1.0f);
      } else {
         norm = glm::vec3(0.0f, 0.0f, 1.0f);
      }
      return glm::vec3(trans * glm::vec4(norm.x, norm.y, norm.z, 0.0f));
   }

   glm::vec3 minP, maxP;
};

#endif //BOX_H

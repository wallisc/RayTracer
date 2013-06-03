#ifndef PLANE_H
#define PLANE_H

#include "Geometry.h"
#include "Material.h"
#include "glm/glm.hpp"
#include "Util.h"

class Plane : public Geometry {
public:
   __device__ Plane(float distance, const glm::vec3 &normal, 
         const Material &mat, const glm::mat4 trans,
         const glm::mat4 &invTrans) :
      Geometry(mat, trans, invTrans), n(normal), d(distance) {
         glm::vec4 wn = trans * glm::vec4(n.x, n.y, n.z, 0.0f);
         worldSpaceN = glm::vec3(wn.x, wn.y, wn.z);
      }

   // Precondition: The given position is on the sphere
   __device__ virtual glm::vec3 getNormalAt(const Ray &ray, float param) const {
      return worldSpaceN;
   }

   __device__ virtual glm::vec3 getCenter() const { 
      return d * worldSpaceN;
   }

   __device__ virtual  BoundingBox getBoundingBox() const {
      return BoundingBox(glm::vec3(-FLT_MAX / 5), glm::vec3(FLT_MAX / 5));
   }

   __device__ virtual glm::vec2 UVAt(const Ray &r, float param) const {
      printf("Planes do not currently support textures\n");
      return glm::vec2(0.0f);
   }

protected:
   
   __device__ virtual float intersects(const Ray &r) const {
      glm::vec3 c = n * d;
      float numer = glm::dot(-n,r.o - c);
      float denom = glm::dot(n, r.d);
      float t;

      if (isFloatZero(numer) || isFloatZero(denom) || 
            isFloatLessThan(t = numer / denom, 0.0f))
         return -1.0f;
      else 
         return t;   
   }

   glm::vec3 worldSpaceN;
   glm::vec3 n;
   float d;
};

#endif //SPHERE_H

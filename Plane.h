#ifndef PLANE_H
#define PLANE_H

#include "Geometry.h"
#include "Material.h"
#include "glm/glm.hpp"
#include "Util.h"

class Plane : public Geometry {
public:
   __device__ Plane(float distance, glm::vec3 normal, Material mat) :
      Geometry(mat), n(normal), d(distance) {}

   __device__ virtual float getIntersection(Ray r) {
      glm::vec3 c = -n * d;
      float numer = glm::dot(-n,r.o - c);
      float denom = glm::dot(n, r.d);
      float t;

      if (isFloatZero(numer) || isFloatZero(denom) || 
            isFloatLessThan(t = numer / denom, 0.0f))
         return -1.0f;
      else 
         return t;   
   }

   // Precondition: The given position is on the sphere
   __device__ virtual glm::vec3 getNormalAt(Ray ray) {
      return n;
   }

private:
   glm::vec3 n;
   float d;

};

#endif //SPHERE_H

#ifndef SPHERE_H
#define SPHERE_H

#include "Geometry.h"
#include "Material.h"
#include "glm/glm.hpp"

class Sphere : public Geometry {
public:
   __device__ Sphere(const glm::vec3 &center, float radius, const Material &mat, 
         const glm::mat4 trans, const glm::mat4 &invTrans) :
      Geometry(mat, trans, invTrans), c(center), r(radius) {
         glm::vec4 wc = trans * glm::vec4(c.x, c.y, c.z, 1.0f);
         worldSpaceCenter = glm::vec3(wc.x, wc.y, wc.z); 
      }


   // Precondition: The given position is on the sphere
   __device__ virtual glm::vec3 getNormalAt(const Ray &ray, float param) const {
      glm::vec3 pos = ray.getPoint(param);

      //TODO investigate a way to get around using glm::normalize
      return glm::normalize(pos - worldSpaceCenter);
   }

private:
   __device__ virtual float intersects(const Ray &ray) const {

      glm::vec3 eMinusC = ray.o - c;
      float dDotD = glm::dot(ray.d, ray.d);

      float discriminant = glm::dot(ray.d, (eMinusC)) * glm::dot(ray.d, (eMinusC))
         - dDotD * (glm::dot(eMinusC, eMinusC) - r * r);

      // If the ray doesn't intersect
      if (discriminant < 0.0f) 
         return -1.0f;


      float firstIntersect = (glm::dot(-ray.d, eMinusC) - sqrt(discriminant))
             / dDotD;

      // If the ray is outside the sphere
      if (isFloatLessThan(0.0, firstIntersect)) {
         return firstIntersect;
      // If the ray is inside the sphere
      } else {
         return (glm::dot(-ray.d, eMinusC) + sqrt(discriminant))
             / dDotD;
      }
   }

   glm::vec3 worldSpaceCenter;
   glm::vec3 c;
   float r;
};

#endif //SPHERE_H

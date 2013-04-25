#ifndef TRIANGLE_H
#define TRIANGLE_H

#include "Geometry.h"
#include "Material.h"
#include "glm/glm.hpp"
#include "Util.h"

class Triangle : public Geometry {
public:
   __device__ Triangle(const glm::vec3 &point1, const glm::vec3 &point2, 
              const glm::vec3 &point3, const Material &mat, const glm::mat4 &trans,
              const glm::mat4 &invTrans) : Geometry(mat, trans, invTrans),
      p1(point1), p2(point2), p3(point3)
   {
      n = glm::normalize(glm::cross(point2 - point1, point3 - point1));
      c = point1;
      glm::vec4 wsn = glm::vec4(n.x, n.y, n.z, 0.0f) * invTrans;
      worldSpaceNormal = glm::vec3(wsn.x, wsn.y, wsn.z);
   }

   __device__ virtual glm::vec3 getNormalAt(const Ray &ray, float param) const {
      return worldSpaceNormal;
   }

private:

   __device__ virtual float intersects(const Ray &r) const {
      float numer = glm::dot(-n,r.o - c);
      float denom = glm::dot(n, r.d);
      float t;

      if (isFloatZero(numer) || isFloatZero(denom) || 
            isFloatLessThan(t = numer / denom, 0.0f))
         return -1.0f;

      // intersection point
      glm::vec3 q(r.d * t + r.o);

      float area = glm::dot(glm::cross(p2 - p1, p3 - p1), n);
      if (isFloatZero(area)) return -1.0f;

      float beta = glm::dot(glm::cross(p1 - p3, q - p3), n) / area;
      if (isFloatLessThan(1.0f, beta) || isFloatLessThan(beta, 0.0f)) return -1.0f;

      float gamma = glm::dot(glm::cross(p2 - p1, q - p1), n) / area;
      if (isFloatLessThan(1.0f,gamma + beta) || isFloatLessThan(gamma, 0.0f)) return -1.0f;

      return t;
   }

   glm::vec3 worldSpaceNormal;
   // Normal and center of the plane the triangle is sitting on
   // TODO c is just p1, kept for syntax readability
   glm::vec3 n, c;

   glm::vec3 p1, p2, p3;


};

#endif //TRIANGLE_H

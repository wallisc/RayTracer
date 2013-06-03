#ifndef SMOOTH_TRIANGLE_H
#define SMOOTH_TRIANGLE_H

#include "Geometry.h"
#include "Material.h"
#include "glm/glm.hpp"
#include "Util.h"
#include "Triangle.h"

class SmoothTriangle : public Triangle {
public:
   __device__ SmoothTriangle(const glm::vec3 &point1, const glm::vec3 &point2, 
           const glm::vec3 &point3, const glm::vec3 &norm1, 
           const glm::vec3 &norm2, const glm::vec3 &norm3, 
           const Material &mat, const glm::mat4 &trans,
           const glm::mat4 &invTrans, glm::vec2 vt1, 
           glm::vec2 vt2, glm::vec2 vt3)
      : Triangle(point1, point2, point3, mat, trans, invTrans, 
         vt1, vt2, vt3) {

      glm::vec4 wsNorm1 = glm::vec4(norm1.x, norm1.y, norm1.z, 0.0f) * invTrans;
      glm::vec4 wsNorm2 = glm::vec4(norm2.x, norm2.y, norm2.z, 0.0f) * invTrans;
      glm::vec4 wsNorm3 = glm::vec4(norm3.x, norm3.y, norm3.z, 0.0f) * invTrans;

      wsN1 = glm::vec3(wsNorm1.x, wsNorm1.y, wsNorm1.z);
      wsN2 = glm::vec3(wsNorm2.x, wsNorm2.y, wsNorm2.z);
      wsN3 = glm::vec3(wsNorm3.x, wsNorm3.y, wsNorm3.z);
   }

   // Precondition: the ray intersects with the triangle
   __device__ virtual glm::vec3 getNormalAt(const Ray &ray, float param) const {
      glm::vec3 q = ray.getPoint(param);
      float area = glm::dot(glm::cross(p2 - p1, p3 - p1), n);
      float beta = glm::dot(glm::cross(p1 - p3, q - p3), n) / area;
      float gamma = glm::dot(glm::cross(p2 - p1, q - p1), n) / area;
      float alpha = 1.0 - beta - gamma;
      return glm::normalize(alpha * wsN1 + beta * wsN2 + gamma * wsN3);
   }

private:
   glm::vec3 wsN1, wsN2, wsN3;

};

#endif //SMOOTH_TRIANGLE_H

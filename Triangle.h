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
         const glm::mat4 &invTrans, glm::vec2 nVt1 = glm::vec2(0.0f), 
         glm::vec2 nVt2 = glm::vec2(0.0f), glm::vec2 nVt3 = glm::vec2(0.0f)) 
      : Geometry(mat, trans, invTrans), p1(point1), p2(point2), p3(point3),
        vt1(nVt1), vt2(nVt2), vt3(nVt3)
   {
      n = glm::normalize(glm::cross(point2 - point1, point3 - point1));
      c = point1;
      glm::vec4 wsn = glm::vec4(n.x, n.y, n.z, 0.0f) * invTrans;
      worldSpaceNormal = glm::vec3(wsn.x, wsn.y, wsn.z);
   }

   __device__ virtual glm::vec3 getNormalAt(const Ray &ray, float param) const {
      return worldSpaceNormal;
   }

   __device__ virtual glm::vec3 getCenter() const { 
      glm::vec3 objSpaceCenter = (p1 + p2 + p3) / 3.0f;
      return glm::vec3(glm::vec4(objSpaceCenter.x, objSpaceCenter.y, objSpaceCenter.z, 1.0f) * trans);
   }

   __device__ virtual  BoundingBox getBoundingBox() const {
      glm::vec3 transP1 = glm::vec3(trans*glm::vec4(p1.x, p1.y, p1.z, 1.0f));
      glm::vec3 transP2 = glm::vec3(trans*glm::vec4(p2.x, p2.y, p2.z, 1.0f));
      glm::vec3 transP3 = glm::vec3(trans*glm::vec4(p3.x, p3.y, p3.z, 1.0f));

      glm::vec3 min = getSmallestBoxCorner(getSmallestBoxCorner(transP1, transP2), transP3);
      glm::vec3 max = getBiggestBoxCorner(getBiggestBoxCorner(transP1, transP2), transP3);

      return BoundingBox(min, max);
   }

   __device__ virtual glm::vec2 UVAt(const Ray &r, float param) const {
      glm::vec3 q = r.getPoint(param);
      float area = glm::dot(glm::cross(p2 - p1, p3 - p1), n);
      float beta = glm::dot(glm::cross(p1 - p3, q - p3), n) / area;
      float gamma = glm::dot(glm::cross(p2 - p1, q - p1), n) / area;
      float alpha = 1.0 - beta - gamma;
      
      return alpha * vt1 + beta * vt2 + gamma * vt3;
   }
protected:

   __device__ virtual float intersects(const Ray &r) const {
      float numer = glm::dot(-n,r.o - c);
      float denom = glm::dot(n, r.d);
      float t;

      if (isFloatZero(numer) || isFloatZero(denom) || 
            isFloatLessThan(t = numer / denom, 0.0f))
         return -1.0f;

      // intersection point
      glm::vec3 P = r.getPoint(t);

      // TODO this only needs to be calculated once per triangle
      glm::vec3 A = p1, B = p2, C = p3;
      glm::vec3 AB = B - A;
      glm::vec3 AC = C - A;
      glm::vec3 N = glm::cross(AB, AC);

      glm::vec3 AP = P - A;
      glm::vec3 ABxAP = glm::cross(AB, AP);
      float v_num = glm::dot(N, ABxAP);
      if (v_num < 0.0f) return -1.0f;

      
      // edge 2
      glm::vec3 BP = P - B;
      glm::vec3 BC = C - B;
      glm::vec3 BCxBP = glm::cross(BC, BP);
      if (glm::dot(N, BCxBP) < 0.0f)
         return -1.0f; // P is on the left side

      // edge 3, needed to compute u
      glm::vec3 CP = P - C;
      // we have computed AC already so we can avoid computing CA by
      // inverting the vectors in the cross product:
      // Cross(CA, CP) = cross(CP, AC);
      glm::vec3 CAxCP = glm::cross(CP, AC);
      float u_num = glm::dot(N, CAxCP);
      if (u_num < 0.0f)
         return -1.0f; // P is on the left side;
      /*
      // compute barycentric coordinates
      float den = glm::dot(N, N); // ABxAC.N where N = ABxAC
      float u = u_num / den;
      float v = v_num / den;
      */

      return t;
   }

   glm::vec3 worldSpaceNormal;
   // Normal and center of the plane the triangle is sitting on
   // TODO c is just p1, kept for syntax readability
   glm::vec3 n, c;
   glm::vec3 p1, p2, p3;
   glm::vec2 vt1, vt2, vt3;
};

#endif //TRIANGLE_H

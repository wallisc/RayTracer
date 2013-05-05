#ifndef RAY_H
#define RAY_H
#include "glm/glm.hpp"

typedef struct Ray {
   __device__ Ray(const glm::vec3 &origin, const glm::vec3 &direction) :o(origin), d(direction) {}
   __device__ glm::vec3 getPoint(float param) const { return o + d * param; }
   __device__ Ray transform(const glm::mat4 &trans) const { 
      glm::vec4 newO = trans * glm::vec4(o.x, o.y, o.z, 1.0f);
      glm::vec4 newD = trans * glm::vec4(d.x, d.y, d.z, 0.0f);
      return Ray(glm::vec3(newO.x, newO.y, newO.z), glm::vec3(newD.x, newD.y, newD.z));
   }

   glm::vec3 o, d;

} Ray;

#endif //RAY_H

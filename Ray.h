#ifndef RAY_H
#define RAY_H
#include "glm/glm.hpp"

typedef struct Ray {
   __device__ Ray(const glm::vec3 &origin, const glm::vec3 &direction) :o(origin), d(direction) {}
   __device__ glm::vec3 getPoint(float param) { return o + d * param; }

   glm::vec3 o, d;
} Ray;

#endif RAY_H

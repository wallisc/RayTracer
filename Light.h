#ifndef LIGHT_H
#define LIGHT_H

#include "glm/glm.hpp"
#include "float.h"
#include "Util.h"
#include "Geometry.h"
#include "Ray.h"

class Light {
public:
   __device__ virtual glm::vec3 getLightAtPoint(glm::vec3 point) const = 0;

   __device__ virtual glm::vec3 getLightDir(glm::vec3 point) const = 0;
   
   __device__ virtual Ray getShadowFeeler(glm::vec3 point) const = 0;
};

#endif //LIGHT_H

#ifndef POINT_LIGHT_H
#define POINT_LIGHT_H

#include "Light.h"

class PointLight : public Light {
public:
   __device__ PointLight(glm::vec3 position, glm::vec3 color) : p(position), c(color) {}

   __device__ virtual glm::vec3 getLightAtPoint(Geometry *geomList[], int geomCount, 
         int objIdx, glm::vec3 point) const {
      return c;
   }

   __device__ virtual glm::vec3 getLightDir(glm::vec3 point) const {
      return glm::normalize(p - point);
   }

   __device__ virtual Ray getShadowFeeler(glm::vec3 point) const {
      return Ray(p, point - p);
   }

private:
   glm::vec3 p, c;
};

#endif //POINT_LIGHT_H

#ifndef POINT_LIGHT_H
#define POINT_LIGHT_H

#include "Light.h"

class PointLight : Light {
public:
   __device__ PointLight(glm::vec3 position, glm::vec3 color) : p(position), c(color) {}

   __device__ virtual glm::vec3 getLightAtPoint(Geometry *geomList[], int geomCount, 
         int objIdx, glm::vec3 point) {
      Ray shadow(point, p - point);
      if (!isInShadow(shadow, geomList, geomCount, objIdx)) {
         return c; 
      } else { 
         return glm::vec3(0.0f);
      }
   }

private:
   glm::vec3 p, c;
};

#endif //POINT_LIGHT_H

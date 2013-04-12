#ifndef LIGHT_H
#define LIGHT_H

#include "glm/glm.hpp"
#include "float.h"
#include "Util.h"
#include "Ray.h"

class Light {
public:
   __device__ virtual glm::vec3 getLightAtPoint(Geometry *geomList[], int geomCount, 
         int objIdx, glm::vec3 point) = 0;

   __device__ virtual glm::vec3 getLightDir(glm::vec3 point) = 0;

protected:
   __device__ static bool isInShadow(Ray shadow, Geometry *geomList[], int geomCount, 
         int objIdx) {
      float t = FLT_MAX;

      for (int i = 0; i < geomCount; i++) {
         if (i == objIdx) continue;

         float dist = geomList[i]->getIntersection(shadow);

         if (dist > 0.0f) return true;
      }
      return false;
   }
};

#endif //LIGHT_H

#ifndef SHADER_H
#define SHADER_H

#include "glm/glm.hpp"
#include "Util.h"
#include "math.h"

class Shader {
public:
   __device__ virtual glm::vec3 shade(glm::vec3 matColor, float amb, float dif, float spec, float roughness,
         glm::vec3 eyeVec, glm::vec3 lightDir, glm::vec3 lightColor, 
         glm::vec3 normal, bool inShadow) = 0;
};

#endif //SHADER_H

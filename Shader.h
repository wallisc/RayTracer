#ifndef SHADER_H
#define SHADER_H

#include "glm/glm.hpp"
#include "Util.h"
#include "math.h"

class Shader {
public:
   __device__ static inline glm::vec3 shade(float amb, float dif, float spec, float roughness,
         glm::vec3 eyeVec, glm::vec3 lightDir, glm::vec3 lightColor, 
         glm::vec3 normal) {

      glm::vec3 light(0.0f);
      
      // Ambient lighting
      light += amb * lightColor;

      // Diffuse lighting
      light += dif * clamp(glm::dot(normal, lightDir), 0.0f, 1.0f) * lightColor;

      // Specular lighting
      glm::vec3 reflect = 2.0f * glm::dot(lightDir, normal) * normal - lightDir;
      light += spec * pow(clamp(glm::dot(reflect, eyeVec), 0.0f, 1.0f), 1.0f / roughness) * lightColor;

      return light; 
   }
};

#endif //SHADER_H

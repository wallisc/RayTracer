#ifndef PHONG_SHADER_H
#define PHONG_SHADER_H
#include "Shader.h"

class PhongShader : public Shader {
public:
   __device__ virtual glm::vec3 shade(glm::vec3 matColor, float amb, float dif, 
         float spec, float roughness, glm::vec3 eyeVec, glm::vec3 lightDir, 
         glm::vec3 lightColor, glm::vec3 normal, bool inShadow) {

      glm::vec3 light(0.0f);
      
      // Ambient lighting
      light += amb * lightColor;
      if (inShadow) return light * matColor;

      // Diffuse lighting
      light += dif * clamp(glm::dot(normal, lightDir), 0.0f, 1.0f) * lightColor;

      // Specular lighting
      glm::vec3 reflect = 2.0f * glm::dot(lightDir, normal) * normal - lightDir;
      light += spec * pow(clamp(glm::dot(reflect, eyeVec), 0.0f, 1.0f), 1.0f / roughness) * lightColor;
      
      return light * matColor; 
   }

};
#endif //PHONG_SHADER_H

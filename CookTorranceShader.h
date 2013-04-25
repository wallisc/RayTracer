#ifndef COOK_TORRANCE_SHADER_H 
#define COOK_TORRANCE_SHADER_H 
#include "Shader.h"

class CookTorranceShader : public Shader {
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
      //Calculate the fresnel term
      glm::vec3 h = glm::normalize((lightDir + eyeVec));
      float ior = 2.0f;
      float r0 = (1.0f - ior) * (1.0f - ior) / ((1.0f + ior) * (1.0f + ior));
      float f = r0 + (1.0f - r0)* pow(1.0f - glm::dot(lightDir, normal), 5.0f);

      // Calculate the beckman distribution
      float nDotH = glm::dot(normal, h);
      float mSqr = roughness * roughness;
      float d = exp((nDotH * nDotH - 1.0f)/(mSqr * nDotH *nDotH)) /
                (mSqr * nDotH * nDotH * nDotH * nDotH);


      // Calculate geometric attenuation
      float gStart = 2.0f * nDotH / (glm::dot(eyeVec, h));
      float g1 = gStart * glm::dot(eyeVec, normal);
      float g2 = gStart * glm::dot(lightDir, normal);
      float g = g1 < g2 ? g1 : g2;
      g = g < 1.0f ? g : 1.0f;

      float kSpec = d * f * g
         / (4.0f * glm::dot(eyeVec, normal) * glm::dot(normal, lightDir));
      light += spec * kSpec * lightColor;
      
      return light * matColor; 
   }
};

#endif //COOK_TORRANCE_SHADER_H 

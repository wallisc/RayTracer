#ifndef COOK_TORRANCE_SHADER_H 
#define COOK_TORRANCE_SHADER_H 
#include "Shader.h"

class CookTorranceShader : public Shader {
public:
   __device__ virtual glm::vec3 shade(float amb, float dif, float spec, float roughness,
         glm::vec3 eyeVec, glm::vec3 lightDir, glm::vec3 lightColor, 
         glm::vec3 normal, bool inShadow) {

      glm::vec3 light(0.0f);
      
      // Ambient lighting
      light += amb * lightColor;
      if (inShadow) return light;

      // Diffuse lighting
      light += dif * clamp(glm::dot(normal, lightDir), 0.0f, 1.0f) * lightColor;

      // Specular lighting
      glm::vec3 h = glm::normalize((lightDir + eyeVec));
      //TODO roughness shouldn't be the exponent here
      float f = pow(1.0f + glm::dot(eyeVec, normal), 1.0f / roughness);

      // Calculate the beckman distribution
      float alp = acos(glm::dot(normal, h));
      float mSqr = roughness * roughness;
      float d = exp(-pow(tan(alp)/(mSqr), 2.0f))/ 
         (M_PI * mSqr * pow(cos(alp), 4.0f));

      // Calculate geometric attenuation
      float gStart = 2.0f * glm::dot(h, normal) / (glm::dot(eyeVec, h));
      float g1 = gStart * glm::dot(eyeVec, normal);
      float g2 = gStart * glm::dot(lightDir, normal);
      float g = g1 < g2 ? g1 : g2;
      g = g < 1.0f ? g : 1.0f;

      float kSpec = d * f * g
         / (4.0f * glm::dot(eyeVec, normal) * glm::dot(normal, lightDir));
      light += spec * kSpec * lightColor;
      
      return light; 
   }
};

#endif //COOK_TORRANCE_SHADER_H 

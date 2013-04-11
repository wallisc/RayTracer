#ifndef MATERIAL_H
#define MATERIAL_H

#include "glm/glm.hpp"

typedef struct Material {
   glm::vec3 clr;
   float amb, dif, spec, rough, refl, refr, ior;

   __device__ Material(glm::vec3 color, float ambiant, float diffuse, 
        float specular, float roughness, float reflection, float refraction, 
        float indexOfRefraction) :
     clr(color), amb(ambiant), dif(diffuse), spec(specular), rough(roughness), 
     refl(reflection), refr(refraction), ior(indexOfRefraction) {} 

} Material;
#endif //MATERIAL_H

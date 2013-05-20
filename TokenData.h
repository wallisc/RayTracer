#ifndef TOKEN_DATA_H
#define TOKEN_DATA_H

#include <vector>
#include "glm/glm.hpp"

typedef struct TKCamera {
   glm::vec3 pos, up, right, lookAt;

   TKCamera(glm::vec3 position, glm::vec3 upVec, glm::vec3 rightVec, 
         glm::vec3 look) : pos(position), up(upVec), right(rightVec), 
         lookAt(look) {}
} TKCamera;

typedef struct TKPigment {
   glm::vec3 clr;
   float f;

   TKPigment(glm::vec3 color = glm::vec3(0.0f), float filter = 0.0f) :
      clr(color), f(filter) {}

} TKPigment;

typedef struct TKFinish {
   float amb, dif, spec, rough, refl, refr, ior;

   // Assign default values to each field
   TKFinish() :
     amb(0.1f), dif(0.6f), spec(0.0), rough(0.05f), 
     refl(0.0f), refr(0.0f), ior(1.0f) {} 

   TKFinish(float ambiant, float diffuse, float specular, float roughness,
        float reflection, float refraction, float indexOfRefraction) :
     amb(ambiant), dif(diffuse), spec(specular), rough(roughness), 
     refl(reflection), refr(refraction), ior(indexOfRefraction) {} 
} TKFinish;

typedef struct TKModifier {
   TKPigment pig;
   TKFinish fin;
   glm::mat4 invTrans;
   glm::mat4 trans;
} TKModifier;

typedef struct TKPointLight {
   glm::vec3 pos, clr;

   TKPointLight(glm::vec3 position, glm::vec3 color) :
      pos(position), clr(color) {}
} TKPointLight;

typedef struct TKBox {
   glm::vec3 p1, p2;
   TKModifier mod;
} TKBox;

typedef struct TKSphere {
   glm::vec3 p;
   float r;
   TKModifier mod;
} TKSphere;

typedef struct TKPlane {
   glm::vec3 n;
   float d;
   TKModifier mod;
} TKPlane;

typedef struct TKCone{
   glm::vec3 p1, p2;
   float r1, r2;
   TKModifier mod;
} TKCone;

typedef struct TKTriangle {
   glm::vec3 p1, p2, p3;
   TKModifier mod;
} TKTriangle;

typedef struct TKSmoothTriangle {
   glm::vec3 p1, p2, p3;
   glm::vec3 n1, n2, n3;
   TKModifier mod;
} TKSmoothTriangle;

typedef struct TKSceneData {
   TKCamera *camera;

   std::vector<TKPointLight> pointLights;

   std::vector<TKBox> boxes;
   std::vector<TKTriangle> triangles;
   std::vector<TKSmoothTriangle> smoothTriangles;
   std::vector<TKSphere> spheres;
   std::vector<TKPlane> planes;

   TKSceneData() : camera(NULL) {}
} TKSceneData;

#endif //TOKEN_DATA_H

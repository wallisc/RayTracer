#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"

using glm::vec3;

vec3 cosineWeightedSample(vec3 normal, float rand1, float rand2) {
   static int lo = 0;
   static int mid = 0;
   static int hi = 0;

   float distFromCenter = 1.0f - sqrt(rand1);
   float theta = 2.0f * M_PI * rand2;
   float phi = M_PI / 2.0f - acos(distFromCenter);

   float phiDeg = phi * 180.0f / M_PI;
   float thetaDeg = theta * 180.0f / M_PI;

   vec3 outV = normal.x < .99f ? glm::cross(normal, vec3(1.0f, 0.0, 0.0)) : vec3(0.0f, 1.0f, 0.0f); 
   glm::mat4 rot1 = glm::rotate(glm::mat4(1.0f), phiDeg, outV);
   glm::mat4 rot2 = glm::rotate(glm::mat4(1.0f), thetaDeg, normal);
   glm::vec4 norm(normal.x, normal.y, normal.z, 0.0f);

   vec3 ret = vec3(rot2 * rot1 * norm);

   //if (phiDeg > 60) hi++; 
   //else if (phiDeg < 30) lo++; 
   //else mid++;
   if (ret.y > .33f) hi++; 
   else if (ret.y < -.33f) lo++; 
   else mid++;

   printf("%d, %d, %d\n", lo, mid, hi);
   //printf("%f, %f, %f\n", ret.x, ret.y, ret.z);
   
   return ret;
}

int main() {
   for (int i = 0; i < 10000; i++) {
      cosineWeightedSample(vec3(1.0f, 0.0f, 0.0f), rand() / (float)RAND_MAX, rand() / (float)RAND_MAX);
   }
}

#include "glm/glm.hpp"
#include <stdio.h>
#include <math.h>

using glm::vec3;

int main() {
   float n1 = 1.0f;
   float n2 = 1.33;
   vec3 n = vec3(0, 1, 0); //normal
   vec3 d = -glm::normalize(vec3(0, 0.1, 1.0)); //ray dir

   float nr = n1 / n2;
   float dDotN = glm::dot(d,n);
   float discrim = 1.0f - nr * nr * (1.0f - dDotN * dDotN);
   vec3 t = nr * (d - n*dDotN) - n * (float)sqrt(discrim);

   if (discrim < 0.0f) {
      printf("Discriminant is imaginary\n");
   } else {
      printf("Angle using dot product is %f\n", acos(glm::dot(-n, glm::normalize(t)))  * 180 / M_PI);
      printf("Angle using snells law is %f\n", asin(sin(acos(-dDotN)) * nr) * 180 / M_PI);
   }
}

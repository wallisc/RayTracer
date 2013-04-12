#ifndef GEOMETRY_H
#define GEOMETRY_H

#include <cuda.h>
#include "Ray.h"
#include "Material.h"
#include "glm/glm.hpp"

class Geometry {
public:
   __device__ Geometry(Material material) : mat(material) {}
   __device__ virtual float getIntersection(Ray r) = 0;
   __device__ Material getMaterial() { return mat; };
   //TODO make this function more efficient (take in a param t also?)
   __device__ virtual glm::vec3 getNormalAt(Ray r) = 0;
   
private:
   Material mat;
};
#endif //GEOMETRY_H

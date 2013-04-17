#ifndef GEOMETRY_H
#define GEOMETRY_H

#include <cuda.h>
#include "Ray.h"
#include "Material.h"
#include "glm/glm.hpp"

class Geometry {
public:
   __device__ Geometry(const Material &material) : mat(material) {}
   __device__ virtual float getIntersection(const Ray &r) const = 0;
   __device__ Material getMaterial() const { return mat; };
   //TODO make this function more efficient (take in a param t also?)
   __device__ virtual glm::vec3 getNormalAt(const Ray &r, float param) const = 0;
   
private:
   Material mat;
};
#endif //GEOMETRY_H

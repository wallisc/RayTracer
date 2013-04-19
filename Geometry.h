#ifndef GEOMETRY_H
#define GEOMETRY_H

#include <cuda.h>
#include "Ray.h"
#include "Material.h"
#include "glm/glm.hpp"

class Geometry {
public:
   __device__ Geometry(const Material &material, const glm::mat4 &transform,
         const glm::mat4 &inverseTransform) : mat(material), trans(transform),
         invTrans(inverseTransform) {}

   __device__ float getIntersection(const Ray &r) {
      return intersects(r.transform(invTrans));
   }

   __device__ Material getMaterial() const { return mat; };
   __device__ virtual glm::vec3 getNormalAt(const Ray &r, float param) const = 0;
   
private:
   __device__ virtual float intersects(const Ray &transformedRay) const = 0;

   Material mat;
   glm::mat4 invTrans;
   glm::mat4 trans;
};
#endif //GEOMETRY_H

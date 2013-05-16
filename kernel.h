#ifndef KERNEL_H
#define KERNEL_H
#include <stdio.h>
#include <cuda.h>
#include <vector_types.h>
#include "TokenData.h"
#include "Geometry.h"
#include "Light.h"
#include "Shader.h"

typedef enum {PHONG, COOK_TORRANCE} ShadingType;
const int kMaxRecurse = 6;
const float kAirIOR = 1.0f;

extern "C" void launch_kernel(TKSceneData *data, ShadingType stype, int width, 
                              int height, uchar4 *output, int sampleCount);

template<int t>
__device__ glm::vec3 shadeObject(Geometry *geomList[], int geomCount, 
      Light *lights[], int lightCount, int objIdx, 
      float intParam, Ray ray, Shader **shader);

#endif //KERNEL_H

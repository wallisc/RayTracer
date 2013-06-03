#ifndef KERNEL_H
#define KERNEL_H
#include <stdio.h>
#include <cuda.h>
#include <vector_types.h>

#include "curand_kernel.h"
#include "TokenData.h"
#include "Geometry.h"
#include "Light.h"
#include "Shader.h"
#include "bvh.h"

const int kMaxStackSize = 100;
const int kGimmeLotsOfMemory = 1000000 * 256;
const int kBlockWidth = 16;
const int kNumStreams = 4;
const int kMonteCarloSamples = 256;
const int kMonteCarloSamplesRoot = 16;
const int kMaxTextures= 10;

typedef enum {PHONG, COOK_TORRANCE} ShadingType;
const int kMaxRecurse = 6;
const float kAirIOR = 1.0f;

const int kXAxis = 0, kYAxis = 1, kZAxis = 2;
const int kAxisNum = 3;


extern "C" void launch_kernel(TKSceneData *data, ShadingType stype, int width, 
                              int height, uchar4 *output, int sampleCount);

template<int t>
__device__ glm::vec3 shadeObject(BVHTree *tree, 
      Light *lights[], int lightCount, Geometry *geom, 
      float intParam, Ray ray, Shader **shader, curandState randStates[]);

#endif //KERNEL_H

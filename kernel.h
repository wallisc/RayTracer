#ifndef KERNEL_H
#define KERNEL_H
#include <stdio.h>
#include <cuda.h>
#include <vector_types.h>
#include "TokenData.h"

typedef enum {PHONG, COOK_TORRANCE} ShadingType;
const int kMaxRecurse = 1;
const float kAirIOR = 1.0f;

extern "C" void launch_kernel(TKSceneData *data, ShadingType stype, int width, 
                              int height, uchar4 *output);

#endif //KERNEL_H

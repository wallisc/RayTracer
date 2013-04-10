#include <stdio.h>

#include "Geometry.h"
#include "Sphere.h"
#include "glm/glm.hpp"
#include "kernel.h"
#include "cudaError.h"

using glm::vec3;
const int kBlockWidth = 16;
const int kMaxGeometry = 1000;
const int kNoShapeFound = -1;
const float kMaxDist = 99999.0f;


__global__ void rayTrace(int resWidth, int resHeight, Geometry *geomList[], 
      int geomCount, TKCamera cam, TKSphere *sphereTks, int numSpheres, 
      uchar4 *output) {

   if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
      for (int i = 0; i < numSpheres; i++) {
         TKSphere s = sphereTks[i];
         TKFinish f = s.mod.fin;
         Material m(s.mod.pig.clr, f.amb, f.dif, f.spec, f.rough, f.refl, f.refr, f.ior);
         geomList[i] = new Sphere(s.p, s.r, m);
      }
   }
   syncthreads();

   int x = blockIdx.x * blockDim.x + threadIdx.x;
   int y = blockIdx.y * blockDim.y + threadIdx.y;

   if (x >= resWidth || y >= resHeight)
      return;

   int index = y * resWidth + x;
   uchar4 clr;
   
   // Generate rays
   float halfWidth = resWidth / 2.0;
   float halfHeight= resHeight / 2.0;
   vec3 rPos = vec3((halfWidth - x) / halfWidth, (halfHeight - y) / halfHeight, 0.0);
   rPos += cam.pos;

   //TODO remove this
   rPos.x *= 5;
   rPos.y *= 5;

   vec3 rDir = glm::normalize(cam.lookAt - cam.pos);
   Ray r(rPos, rDir);

   float t = kMaxDist;
   int closestShapeIdx = kNoShapeFound;
   for (int i = 0; i < geomCount; i++) {
      float dist = geomList[i]->getIntersection(r);
      if (dist > 0.0 && dist < t) {
         closestShapeIdx = i;
         t = dist;
      }
   }

   if (closestShapeIdx != kNoShapeFound) {
      Material m = geomList[closestShapeIdx]->getMaterial();
      clr.x = m.clr.x * 255.0; clr.y = m.clr.y * 255.0; 
      clr.z = m.clr.z * 255.0; clr.w = 255;
   } else {
      clr.x = 0; clr.y = 0; clr.z = 0; clr.w = 255;
   }

   output[index] = clr;
}

extern "C" void launch_kernel(TKSceneData *data, int width, int height, uchar4 *output) {
  Geometry **dGeomList; 
  TKSphere *dSphereTokens = NULL;
  uchar4 *dOutput;

  int geometryCount = 0;

  // Cuda memory allocation
  int sphereCount = data->spheres.size();
  if (sphereCount > 0) {
     HANDLE_ERROR(cudaMalloc(&dSphereTokens, sizeof(TKSphere) * sphereCount));
     HANDLE_ERROR(cudaMemcpy(dSphereTokens, &data->spheres[0], 
              sizeof(TKSphere) * sphereCount, cudaMemcpyHostToDevice));
  }

  geometryCount += sphereCount;
  HANDLE_ERROR(cudaMalloc(&dGeomList, sizeof(Geometry *) * kMaxGeometry));

  HANDLE_ERROR(cudaMalloc(&dOutput, sizeof(uchar4) * width * height));

  // Do the actual kernel
  dim3 dimBlock(kBlockWidth, kBlockWidth);
  dim3 dimGrid((width - 1) / kBlockWidth + 1, (height - 1) / kBlockWidth + 1);
  rayTrace<<<dimGrid, dimBlock>>>(width, height, dGeomList, geometryCount, *data->camera, dSphereTokens, 
        sphereCount, dOutput);

  cudaDeviceSynchronize();
  checkCUDAError("kernel failed");

  if (dSphereTokens) HANDLE_ERROR(cudaFree(dSphereTokens));

  HANDLE_ERROR(cudaFree(dGeomList));
  HANDLE_ERROR(cudaMemcpy(output, dOutput, 
           sizeof(uchar4) * width * height, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaFree(dOutput));
}



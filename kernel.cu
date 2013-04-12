#include <stdio.h>
#include <float.h>

#include "Geometry.h"
#include "Light.h"
#include "PointLight.h"
#include "Sphere.h"
#include "Plane.h"
#include "glm/glm.hpp"
#include "kernel.h"
#include "Shader.h"
#include "cudaError.h"

const int kBlockWidth = 16;
const int kNoShapeFound = -1;
const float kMaxDist = FLT_MAX;

using glm::vec3;

__device__ uchar4 shadeObject(Geometry *geomList[], int geomCount, 
                              Light *lights[], int lightCount, int objIdx, 
                              vec3 intersectPoint, Ray ray) {
      uchar4 clr;
      Material m = geomList[objIdx]->getMaterial();
      vec3 totalLight(0.0f);

      for(int i = 0; i < lightCount; i++) {
         vec3 light = lights[i]->getLightAtPoint(geomList, geomCount, objIdx, intersectPoint);
         vec3 lightDir = lights[i]->getLightDir(intersectPoint);
         vec3 normal = geomList[i]->getNormalAt(ray);
         totalLight += Shader::shade(m.amb, m.dif, m.spec, m.rough, 
               glm::normalize(-ray.d), lightDir, light, normal); 
      }

      clr.x = clamp(m.clr.x * totalLight.x * 255.0, 0.0f, 255.0f); 
      clr.y = clamp(m.clr.y * totalLight.y * 255.0, 0.0f, 255.0f); 
      clr.z = clamp(m.clr.z * totalLight.z * 255.0, 0.0f, 255.0f); 
      clr.w = 255;
      return clr;
}

__global__ void initScene(Geometry *geomList[], Light *lights[], TKSphere *sphereTks, int numSpheres,
      TKPlane *planeTks, int numPlanes, TKPointLight *pLightTks, int numPointLights) {
   int geomIdx = 0;
   // This should really only be run with one thread and block anyways, but this is a safety check
   if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
      for (int i = 0; i < numSpheres; i++) {
         TKSphere s = sphereTks[i];
         TKFinish f = s.mod.fin;
         Material m(s.mod.pig.clr, f.amb, f.dif, f.spec, f.rough, f.refl, f.refr, f.ior);
         geomList[geomIdx++] = new Sphere(s.p, s.r, m);
      }

      for (int i = 0; i < numPlanes; i++) {
         TKPlane p = planeTks[i];
         TKFinish f = p.mod.fin;
         Material m(p.mod.pig.clr, f.amb, f.dif, f.spec, f.rough, f.refl, f.refr, f.ior);
         geomList[geomIdx++] = new Plane(p.d, p.n, m);
      }

      for (int i = 0; i < numPointLights; i++) {
         TKPointLight &p = pLightTks[i];
         lights[i] = new PointLight(p.pos, p.clr);

      }
   }
}

__global__ void rayTrace(int resWidth, int resHeight, TKCamera cam,
      Geometry *geomList[], int geomCount, Light *lights[], int lightCount,  
      uchar4 *output) {

   int x = blockIdx.x * blockDim.x + threadIdx.x;
   int y = blockIdx.y * blockDim.y + threadIdx.y;

   if (x >= resWidth || y >= resHeight)
      return;

   int index = y * resWidth + x;
   uchar4 clr;
   
   // Generate rays
   float halfWidth = resWidth / 2.0;
   float halfHeight = resHeight / 2.0;

   //Image space coordinates 
   float u = 2.0f * (x / (float)resWidth) - 1.0f; 
   float v = 2.0f * (y / (float)resHeight) - 1.0f;

   //TODO currently makes the assumption that cam.up is normalized
   // .5f is because the magnitude of cam.right and cam.up should be equal
   // to the width and height of the image plane in world space
   vec3 rPos = u *.5f * cam.right + v * .5f * cam.up + cam.pos;

   //TODO if the cam.lookAt - cam.pos was already normalized, could lead to 
   // speedups
   vec3 lookAtVec = glm::normalize(cam.lookAt - cam.pos);
   vec3 rDir = rPos - cam.pos + lookAtVec;
   Ray r(rPos, rDir);

   float t = kMaxDist;
   int closestShapeIdx = kNoShapeFound;
   for (int i = 0; i < geomCount; i++) {
      float dist = geomList[i]->getIntersection(r);
      if (dist > 0.0f && dist < t) {
         closestShapeIdx = i;
         t = dist;
      }
   }

   if (closestShapeIdx != kNoShapeFound) {
      clr = shadeObject(geomList, geomCount, lights, lightCount, 
                        closestShapeIdx, r.getPoint(t), r);
   } else {
      clr.x = 0; clr.y = 0; clr.z = 0; clr.w = 255;
   }

   output[index] = clr;
}

extern "C" void launch_kernel(TKSceneData *data, int width, int height, uchar4 *output) {
  Geometry **dGeomList; 
  Light **dLightList;

  TKSphere *dSphereTokens = NULL;
  TKPlane *dPlaneTokens = NULL;
  TKPointLight *dPointLightTokens = NULL;
  uchar4 *dOutput;

  int geometryCount = 0;
  int lightCount = 0;

  // Cuda memory allocation
  int sphereCount = data->spheres.size();
  if (sphereCount > 0) {
     HANDLE_ERROR(cudaMalloc(&dSphereTokens, sizeof(TKSphere) * sphereCount));
     HANDLE_ERROR(cudaMemcpy(dSphereTokens, &data->spheres[0], 
              sizeof(TKSphere) * sphereCount, cudaMemcpyHostToDevice));
     geometryCount += sphereCount;
  }
  
  int planeCount = data->planes.size();
  if (planeCount > 0) {
     HANDLE_ERROR(cudaMalloc(&dPlaneTokens, sizeof(TKPlane) * planeCount));
     HANDLE_ERROR(cudaMemcpy(dPlaneTokens, &data->planes[0],
           sizeof(TKPlane) * planeCount, cudaMemcpyHostToDevice));
     geometryCount += planeCount;
  }

  int pointLightCount = data->pointLights.size();
  if (pointLightCount > 0) {
      HANDLE_ERROR(cudaMalloc(&dPointLightTokens, 
             sizeof(TKPointLight) * pointLightCount));
      HANDLE_ERROR(cudaMemcpy(dPointLightTokens, &data->pointLights[0],
            sizeof(TKPointLight) * pointLightCount, cudaMemcpyHostToDevice));
      lightCount += pointLightCount;
  }

  HANDLE_ERROR(cudaMalloc(&dGeomList, sizeof(Geometry *) * geometryCount));
  HANDLE_ERROR(cudaMalloc(&dLightList, sizeof(Light *) * lightCount));

  HANDLE_ERROR(cudaMalloc(&dOutput, sizeof(uchar4) * width * height));

  // Do the actual kernel
  initScene<<<1, 1>>>(dGeomList, dLightList, dSphereTokens, sphereCount, dPlaneTokens, 
        planeCount, dPointLightTokens, pointLightCount);

  dim3 dimBlock(kBlockWidth, kBlockWidth);
  dim3 dimGrid((width - 1) / kBlockWidth + 1, (height - 1) / kBlockWidth + 1);
  rayTrace<<<dimGrid, dimBlock>>>(width, height, *data->camera, 
        dGeomList, geometryCount, dLightList, lightCount, dOutput);

  cudaDeviceSynchronize();
  checkCUDAError("kernel failed");

  if (dSphereTokens) HANDLE_ERROR(cudaFree(dSphereTokens));
  if (dPlaneTokens) HANDLE_ERROR(cudaFree(dPlaneTokens));

  //HANDLE_ERROR(cudaFree(dGeomList));
  HANDLE_ERROR(cudaMemcpy(output, dOutput, 
           sizeof(uchar4) * width * height, cudaMemcpyDeviceToHost));
  //HANDLE_ERROR(cudaFree(dOutput));
}



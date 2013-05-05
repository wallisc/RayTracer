#include <stdio.h>
#include <float.h>

#include "Geometry.h"
#include "Light.h"
#include "PointLight.h"
#include "Sphere.h"
#include "Plane.h"
#include "Triangle.h"
#include "glm/glm.hpp"
#include "kernel.h"
#include "Shader.h"
#include "PhongShader.h"
#include "SmoothTriangle.h"
#include "CookTorranceShader.h"
#include "cudaError.h"

const int kBlockWidth = 16;
const int kNoShapeFound = -1;
const float kMaxDist = FLT_MAX;

using glm::vec3;

__device__ bool isInShadow(const Ray &shadow, Geometry *geomList[], int geomCount, float intersectParam) {
   return false;
   for (int i = 0; i < geomCount; i++) {
      float dist = geomList[i]->getIntersection(shadow);
      if (isFloatAboveZero(dist) && isFloatLessThan(dist, intersectParam)) { 
         return true;
      }
   }
   return false;
}

// Find the closest shape. The index of the intersecting object is stored in
// retOjIdx and the t-value along the input ray is stored in retParam
//
// If no intersection is found, retObjIdx is set to 'kNoShapeFound'
__device__ void getClosestIntersection(const Ray &ray, Geometry *geomList[], 
      int geomCount, int *retObjIdx, float *retParam) {
   float t = kMaxDist;
   int closestShapeIdx = kNoShapeFound;
   for (int i = 0; i < geomCount; i++) {
      float dist = geomList[i]->getIntersection(ray);


      glm::vec3 newNorm = geomList[i]->getNormalAt(ray, dist);
      glm::vec3 eye = glm::normalize(-ray.d);
      //if (glm::dot(eye, newNorm) < 0.0f)
      //   continue;
      // If two faces are very close, this picks the face that's normal
      // is closer to the incoming ray
      if (isFloatEqual(t, dist)) {
         glm::vec3 oldNorm = geomList[closestShapeIdx]->getNormalAt(ray, t);
         //glm::vec3 newNorm = geomList[i]->getNormalAt(ray, dist);
         //glm::vec3 eye = glm::normalize(-ray.d);
         if (glm::dot(eye, newNorm) > glm::dot(eye, oldNorm)) {
            closestShapeIdx = i;
            t = dist;
         }

      // Otherwise, if one face is front of the current one
      } else if (isFloatAboveZero(dist) && dist < t) {
         //if (t - dist < 0.2f) { printf("Changing t from %f to %f\nIndex change from %d to %d\n", t, dist, closestShapeIdx, i); }
         closestShapeIdx = i;
         t = dist;
      }
   }

   *retObjIdx = closestShapeIdx;
   *retParam = t;
}

//Note: The ray parameter must stay as a copy (not a reference) 
template <int invRecLevel> 
__device__ vec3 shadeObject(Geometry *geomList[], int geomCount, 
      Light *lights[], int lightCount, int objIdx, 
      float intParam, Ray ray, Shader **shader) {
   glm::vec3 intersectPoint = ray.getPoint(intParam);
   Material m = geomList[objIdx]->getMaterial();
   vec3 totalLight(0.0f);

   vec3 normal = geomList[objIdx]->getNormalAt(ray, intParam);
   vec3 eyeVec = glm::normalize(-ray.d);

   for(int lightIdx = 0; lightIdx < lightCount; lightIdx++) {
      vec3 light = lights[lightIdx]->getLightAtPoint(geomList, geomCount, objIdx, intersectPoint);
      vec3 lightDir = lights[lightIdx]->getLightDir(intersectPoint);

      Ray shadow = lights[lightIdx]->getShadowFeeler(intersectPoint);
      // 2 square roots happening here, could probably be optimized
      //float intersectParam = glm::length(intersectPoint - shadow.o) / glm::length(shadow.d);
      float intersectParam = geomList[objIdx]->getIntersection(shadow);
      bool inShadow = isInShadow(shadow, geomList, geomCount, intersectParam); 
      totalLight += (*shader)->shade(m.clr, m.amb, m.dif, m.spec, m.rough, 
            eyeVec, lightDir, light, normal, 
            inShadow);
   }

   vec3 reflectedLight(0.0f);
   if (m.refl > 0.0f && invRecLevel > 0) {
      Ray reflectRay(intersectPoint, 2.0f * glm::dot(normal, eyeVec) * normal - eyeVec);
      int reflObjIdx;
      float reflParam;

      getClosestIntersection(reflectRay, geomList, geomCount, &reflObjIdx, &reflParam);
      if (reflObjIdx != kNoShapeFound) {
         reflectedLight = shadeObject<invRecLevel - 1>(geomList, geomCount, 
               lights, lightCount,
               reflObjIdx, reflParam,
               reflectRay, shader);
      }
   }

   vec3 refractedLight(0.0f);
   //TODO move the recursion break condition to a more efficient position
   if (m.refr > 0.0f && invRecLevel > 0) {
      float n1, n2;
      vec3 refrNorm;
      vec3 d = -eyeVec;

      if (isFloatLessThan(glm::dot(eyeVec, normal), 0.0f)) {
         n1 = m.ior; n2 = kAirIOR;
         refrNorm = -normal;
      } else { 
         n1 = kAirIOR; n2 = m.ior;
         refrNorm = normal;
      }

      float dDotN = glm::dot(d, refrNorm);
      float nr = n1 / n2;
      float discriminant = 1.0f - nr * nr * (1.0f - dDotN * dDotN);
      if (discriminant > 0.0f) {
         vec3 refracDir = nr * (d - refrNorm * dDotN) - refrNorm * sqrt(discriminant);
         Ray refracRay(intersectPoint, refracDir);
         // TODO duplicate code from reflection, DRY
         int refrObjIdx;
         float refrParam;
         getClosestIntersection(refracRay, geomList, geomCount, &refrObjIdx, &refrParam);
         if (refrObjIdx != kNoShapeFound) {
            refractedLight = shadeObject<invRecLevel - 1>(geomList, geomCount, 
                  lights, lightCount,
                  refrObjIdx, refrParam,
                  refracRay, shader);
         }
      }

   }

   totalLight =  totalLight
      + m.refl * reflectedLight+ m.refr * refractedLight;
   return totalLight;
}

template <> 
__device__ vec3 shadeObject<0>(Geometry *geomList[], int geomCount, 
      Light *lights[], int lightCount, int objIdx, 
      float intParam, Ray ray, Shader **shader) { return vec3(0.0f); }

__global__ void initScene(Geometry *geomList[], Light *lights[], TKSphere *sphereTks, int numSpheres,
      TKPlane *planeTks, int numPlanes, TKTriangle *triangleTks, int numTris, 
      TKSmoothTriangle *smthTriTks, int numSmthTris, TKPointLight *pLightTks, int numPointLights, 
      Shader **shader, ShadingType stype) {
   int geomIdx = 0;
   int lightIdx = 0;

   // This should really only be run with one thread and block anyways, but this is a safety check
   if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0) {

      // Setup the shader
      switch(stype) {
      case PHONG:
         *shader = new PhongShader(); 
         break;
      case COOK_TORRANCE:
         *shader = new CookTorranceShader();
         break;
      default:
         printf("Improper shading type specified\n");
         break;
      }

      // Add all the geometry
      for (int i = 0; i < numSpheres; i++) {
         const TKSphere &s = sphereTks[i];
         const TKFinish f = s.mod.fin;
         Material m(s.mod.pig.clr, f.amb, f.dif, f.spec, f.rough, f.refl, f.refr, f.ior);
         geomList[geomIdx++] = new Sphere(s.p, s.r, m, s.mod.trans, s.mod.invTrans);
      }

      for (int i = 0; i < numPlanes; i++) {
         const TKPlane &p = planeTks[i];
         const TKFinish &f = p.mod.fin;
         Material m(p.mod.pig.clr, f.amb, f.dif, f.spec, f.rough, f.refl, f.refr, f.ior);
         geomList[geomIdx++] = new Plane(p.d, p.n, m, p.mod.trans, p.mod.invTrans);
      }

      for (int i = 0; i < numTris; i++) {
         const TKTriangle &t = triangleTks[i];
         const TKFinish f = t.mod.fin;
         Material m(t.mod.pig.clr, f.amb, f.dif, f.spec, f.rough, f.refl, f.refr, f.ior);
         geomList[geomIdx++] = new Triangle(t.p1, t.p2, t.p3, m, t.mod.trans, 
               t.mod.invTrans);
      }

      for (int i = 0; i < numSmthTris; i++) {
         const TKSmoothTriangle &t = smthTriTks[i];
         const TKFinish f = t.mod.fin;
         Material m(t.mod.pig.clr, f.amb, f.dif, f.spec, f.rough, f.refl, f.refr, f.ior);
         geomList[geomIdx++] = new SmoothTriangle(t.p1, t.p2, t.p3, t.n1, t.n2, t.n3, 
               m, t.mod.trans, t.mod.invTrans);

      }

      // Add all the lights
      for (int i = 0; i < numPointLights; i++) {
         TKPointLight &p = pLightTks[i];
         lights[lightIdx++] = new PointLight(p.pos, p.clr);
      }
   }
}

__global__ void deleteScene(Geometry *geomList[], int geomCount, Light *lightList[], int lightCount) {
   // This should really only be run with one thread and block anyways, but this is a safety check
   if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
      for (int i = 0; i < geomCount; i++) {
         delete geomList[i];
      }

      for (int i = 0; i < lightCount; i++) {
         delete lightList[i];
      }
   }
}


__global__ void rayTrace(int resWidth, int resHeight, TKCamera cam,
      Geometry *geomList[], int geomCount, Light *lights[], int lightCount,  
      uchar4 *output, Shader **shader) {

   int x = blockIdx.x * blockDim.x + threadIdx.x;
   int y = blockIdx.y * blockDim.y + threadIdx.y;

   if (x >= resWidth || y >= resHeight)
      return;

   int index = y * resWidth + x;
   uchar4 clr;

   // Generate rays
   //Image space coordinates 
   float u = 2.0f * (x / (float)resWidth) - 1.0f; 
   float v = 2.0f * (y / (float)resHeight) - 1.0f;

   // .5f is because the magnitude of cam.right and cam.up should be equal
   // to the width and height of the image plane in world space
   vec3 rPos = u *.5f * cam.right + v * .5f * cam.up + cam.pos;

   //TODO if the cam.lookAt - cam.pos was already normalized, could lead to 
   // speedups
   vec3 lookAtVec = glm::normalize(cam.lookAt - cam.pos);
   vec3 rDir = rPos - cam.pos + lookAtVec;
   Ray r(rPos, rDir);

   float t;
   int closestShapeIdx;
   getClosestIntersection(r, geomList, geomCount, &closestShapeIdx, &t);

   if (closestShapeIdx != kNoShapeFound) {
      vec3 totalColor = shadeObject<kMaxRecurse>(geomList, geomCount, lights, lightCount, 
            closestShapeIdx, t, r, shader);

      clr.x = clamp(totalColor.x * 255.0, 0.0f, 255.0f); 
      clr.y = clamp(totalColor.y * 255.0, 0.0f, 255.0f); 
      clr.z = clamp(totalColor.z * 255.0, 0.0f, 255.0f); 
      clr.w = 255;
   } else {
      clr.x = 0; clr.y = 0; clr.z = 0; clr.w = 255;
   }

   output[index] = clr;
}

void allocateGPUScene(TKSceneData *data, Geometry ***dGeomList, Light ***dLightList, 
      int *retGeometryCount, int *retLightCount, Shader **dShader, ShadingType stype) {
   int geometryCount = 0;
   int lightCount = 0;

   TKSphere *dSphereTokens = NULL;
   TKPlane *dPlaneTokens = NULL;
   TKPointLight *dPointLightTokens = NULL;
   TKTriangle *dTriangleTokens = NULL;
   TKSmoothTriangle *dSmthTriTokens = NULL;

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

   int triangleCount = data->triangles.size();
   if (triangleCount > 0) {
      HANDLE_ERROR(cudaMalloc(&dTriangleTokens, sizeof(TKTriangle) * triangleCount));
      HANDLE_ERROR(cudaMemcpy(dTriangleTokens, &data->triangles[0], 
               sizeof(TKTriangle) * triangleCount, cudaMemcpyHostToDevice));
      geometryCount += triangleCount;
   }

   int smoothTriangleCount = data->smoothTriangles.size();
   if (smoothTriangleCount > 0) {
      HANDLE_ERROR(cudaMalloc(&dSmthTriTokens, sizeof(TKSmoothTriangle) * smoothTriangleCount));
      HANDLE_ERROR(cudaMemcpy(dSmthTriTokens, &data->smoothTriangles[0],
               sizeof(TKSmoothTriangle) * smoothTriangleCount, cudaMemcpyHostToDevice));
      geometryCount += smoothTriangleCount;
   }

   int pointLightCount = data->pointLights.size();
   if (pointLightCount > 0) {
      HANDLE_ERROR(cudaMalloc(&dPointLightTokens, 
               sizeof(TKPointLight) * pointLightCount));
      HANDLE_ERROR(cudaMemcpy(dPointLightTokens, &data->pointLights[0],
               sizeof(TKPointLight) * pointLightCount, cudaMemcpyHostToDevice));
      lightCount += pointLightCount;
   }

   HANDLE_ERROR(cudaMalloc(dGeomList, sizeof(Geometry *) * geometryCount));
   HANDLE_ERROR(cudaMalloc(dLightList, sizeof(Light *) * lightCount));

   // Fill up GeomList and LightList with actual objects on the GPU
   initScene<<<1, 1>>>(*dGeomList, *dLightList, dSphereTokens, sphereCount, 
         dPlaneTokens, planeCount, dTriangleTokens, triangleCount, 
         dSmthTriTokens, smoothTriangleCount, dPointLightTokens, pointLightCount, 
         dShader, stype);

   if (dSphereTokens) HANDLE_ERROR(cudaFree(dSphereTokens));
   if (dPlaneTokens) HANDLE_ERROR(cudaFree(dPlaneTokens));
   if (dTriangleTokens) HANDLE_ERROR(cudaFree(dTriangleTokens));
   if (dSmthTriTokens) HANDLE_ERROR(cudaFree(dSmthTriTokens));

   *retGeometryCount = geometryCount;
   *retLightCount = lightCount;
}

void freeGPUScene(Geometry **dGeomList, int geomCount, Light **dLightList, 
      int lightCount) {
   deleteScene<<<1, 1>>>(dGeomList, geomCount, dLightList, lightCount);

   HANDLE_ERROR(cudaFree(dGeomList));
   HANDLE_ERROR(cudaFree(dLightList));
}

extern "C" void launch_kernel(TKSceneData *data, ShadingType stype, int width, 
      int height, uchar4 *output) {
   Geometry **dGeomList; 
   Light **dLightList;
   Shader **dShader;

   uchar4 *dOutput;

   int geometryCount;
   int lightCount;


   HANDLE_ERROR(cudaMalloc(&dShader, sizeof(Shader*)));
   HANDLE_ERROR(cudaMalloc(&dOutput, sizeof(uchar4) * width * height));

   allocateGPUScene(data, &dGeomList, &dLightList, &geometryCount, &lightCount, dShader, stype);
   cudaDeviceSynchronize();
   checkCUDAError("AllocateGPUScene failed");

   dim3 dimBlock(kBlockWidth, kBlockWidth);
   dim3 dimGrid((width - 1) / kBlockWidth + 1, (height - 1) / kBlockWidth + 1);
   rayTrace<<<dimGrid, dimBlock>>>(width, height, *data->camera, 
         dGeomList, geometryCount, dLightList, lightCount, dOutput, dShader);

   cudaDeviceSynchronize();
   checkCUDAError("RayTrace kernel failed");

   freeGPUScene(dGeomList, geometryCount, dLightList, lightCount);
   HANDLE_ERROR(cudaMemcpy(output, dOutput, 
            sizeof(uchar4) * width * height, cudaMemcpyDeviceToHost));
   HANDLE_ERROR(cudaFree(dOutput));
}

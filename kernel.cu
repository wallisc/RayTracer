#include <algorithm>
#include <stdio.h>
#include <float.h>

#include "Light.h"
#include "Camera.h"
#include "PointLight.h"
#include "Sphere.h"
#include "Box.h"
#include "Plane.h"
#include "Triangle.h"
#include "glm/glm.hpp"
#include "PhongShader.h"
#include "SmoothTriangle.h"
#include "CookTorranceShader.h"
#include "cudaError.h"
#include "kernel.h"
#include "curand.h"

#define kNoShapeFound NULL

using glm::vec3;

const int kBlockWidth = 16;
const float kMaxDist = FLT_MAX;

__device__ bool isInShadow(const Ray &shadow, BVHTree *tree, float intersectParam) {
   return false;
   //for (int i = 0; i < geomCount; i++) {
   //   float dist = geomList[i]->getIntersection(shadow);
   //   if (isFloatAboveZero(dist) && isFloatLessThan(dist, intersectParam)) { 
   //      return true;
   //   }
   //}
   //return false;
}

// Find the closest shape. The index of the intersecting object is stored in
// retOjIdx and the t-value along the input ray is stored in retParam
//
// If no intersection is found, retObjIdx is set to 'kNoShapeFound'
__device__ void getClosestIntersection(const Ray &ray, BVHTree *tree, 
      Geometry **retObj, float *retParam) {
   float t = kMaxDist;
   Geometry *closestGeom = kNoShapeFound;

   BVHNode *stack[kMaxStackSize];
   int stackSize = 0;
   bool justPoppedStack = false;

   BVHNode *cursor = tree->root;
     
   do {
      if (stackSize >= kMaxStackSize) {
         printf("Stack full, aborting!\n");
         return;
      }
         
      // If at a leaf
      if (cursor->geom) {
         float dist = cursor->geom->getIntersection(ray);
         //If two shapes are overlapping, pick the one with the closest facing normal
         if (isFloatEqual(t, dist)) {
            glm::vec3 oldNorm = closestGeom->getNormalAt(ray, t);
            glm::vec3 newNorm = cursor->geom->getNormalAt(ray, dist);
            glm::vec3 eye = glm::normalize(-ray.d);
            float newDot = glm::dot(eye, newNorm);
            float oldDot = glm::dot(eye, oldNorm);
            if (newDot > oldDot) {
               closestGeom = cursor->geom;
               t = dist;
            }
         // Otherwise, if one face is front of the current one
         } else if (dist < t && isFloatAboveZero(dist)) {
            t = dist;
            closestGeom = cursor->geom;
         }
      } else if (!justPoppedStack && isFloatAboveZero(cursor->left->bb.getIntersection(ray)) && cursor->left->bb.getIntersection(ray) < t) {
         //go left
         stack[stackSize++] = cursor;
         cursor = cursor->left;
         justPoppedStack = false;
         continue;
      } else if (cursor->right && isFloatAboveZero(cursor->right->bb.getIntersection(ray)) && cursor->right->bb.getIntersection(ray) < t) {
         //go right
         cursor = cursor->right;
         justPoppedStack = false;
         continue;
      }

      if(stackSize > 0) {
         // Pop the stack
         cursor = stack[stackSize - 1]; 
         justPoppedStack = true;
      }
      stackSize--;
   } while(stackSize >= 0);

   for (int planeIdx = 0; planeIdx < tree->planeListSize; planeIdx++) {
      float dist = tree->planeList[planeIdx]->getIntersection(ray);
      if (isFloatLessThan(dist, t) && isFloatAboveZero(dist)) {
         closestGeom = tree->planeList[planeIdx];
         t = dist;
      }

   }

   *retObj = closestGeom;
   *retParam = t;
}

template <int invRecLevel>
__device__ glm::vec3 getReflection(glm::vec3 point, glm::vec3 normal, glm::vec3 eyeVec, 
   BVHTree *tree, Light *lights[], int lightCount, Shader **shader) {

   Ray reflectRay(point, 2.0f * glm::dot(normal, eyeVec) * normal - eyeVec);
   reflectRay.o += BIG_EPSILON * reflectRay.d;
   Geometry *closestGeom;
   float t;

   getClosestIntersection(reflectRay, tree, &closestGeom, &t);
   if (closestGeom != kNoShapeFound) {
      return shadeObject<invRecLevel>(tree, 
            lights, lightCount,
            closestGeom, t,
            reflectRay, shader);
   } 
   return vec3(0.0f);
}

template <>
__device__ glm::vec3 getReflection<0>(glm::vec3 point, glm::vec3 normal, glm::vec3 eyeVec, 
   BVHTree *tree, Light *lights[], int lightCount, 
   Shader **shader) { return vec3(0.0f); }

template <int invRecLevel>
__device__ glm::vec3 getRefraction(glm::vec3 point, glm::vec3 normal, float ior, glm::vec3 eyeVec, 
   BVHTree *tree, Light *lights[], int lightCount, Shader **shader) {

   float n1, n2;
   vec3 refrNorm;
   vec3 d = -eyeVec;

   if (isFloatLessThan(glm::dot(eyeVec, normal), 0.0f)) {
      n1 = ior; n2 = kAirIOR;
      refrNorm = -normal;
   } else { 
      n1 = kAirIOR; n2 = ior;
      refrNorm = normal;
   }

   float dDotN = glm::dot(d, refrNorm);
   float nr = n1 / n2;
   float discriminant = 1.0f - nr * nr * (1.0f - dDotN * dDotN);
   if (discriminant > 0.0f) {
      vec3 refracDir = nr * (d - refrNorm * dDotN) - refrNorm * sqrtf(discriminant);
      Ray refracRay(point, refracDir);
      refracRay.o += BIG_EPSILON * refracRay.d;
      Geometry *closestGeom;
      float t;
      getClosestIntersection(refracRay, tree, &closestGeom, &t);
      if (closestGeom != kNoShapeFound) {
         return shadeObject<invRecLevel>(tree,
               lights, lightCount,
               closestGeom, t,
               refracRay, shader);
      }
   } 
   return vec3(0.0f);
}

template <>
__device__ glm::vec3 getRefraction<0>(glm::vec3 point, glm::vec3 normal, float ior, glm::vec3 eyeVec, 
   BVHTree *tree, Light *lights[], int lightCount, 
   Shader **shader) { return vec3(0.0f); }


//Note: The ray parameter must stay as a copy (not a reference) 
template <int invRecLevel> 
__device__ vec3 shadeObject(BVHTree *tree, 
      Light *lights[], int lightCount, Geometry* geom, 
      float intParam, Ray ray, Shader **shader) {

   glm::vec3 intersectPoint = ray.getPoint(intParam);
   Material m = geom->getMaterial();
   vec3 normal = geom->getNormalAt(ray, intParam);
   vec3 eyeVec = glm::normalize(-ray.d);
   vec3 totalLight(0.0f);

   for(int lightIdx = 0; lightIdx < lightCount; lightIdx++) {
      vec3 light = lights[lightIdx]->getLightAtPoint(geom, intersectPoint);
      vec3 lightDir = lights[lightIdx]->getLightDir(intersectPoint);
      Ray shadow = lights[lightIdx]->getShadowFeeler(intersectPoint);
      float intersectParam = geom->getIntersection(shadow);
      bool inShadow = isInShadow(shadow, tree, intersectParam); 

      totalLight += (*shader)->shade(m.clr, m.amb, m.dif, m.spec, m.rough, 
            eyeVec, lightDir, light, normal, 
            inShadow);
   }

   vec3 reflectedLight(0.0f);
   if (m.refl > 0.0f && invRecLevel - 1 > 0) {
      reflectedLight = getReflection<invRecLevel - 1>(intersectPoint, 
         normal, eyeVec, tree, lights, lightCount, shader);
   }

   vec3 refractedLight(0.0f);
   if (m.refr > 0.0f && invRecLevel - 1 > 0) {
      refractedLight = getRefraction<invRecLevel - 1>(intersectPoint, 
         normal, m.ior, eyeVec, tree, lights, lightCount, shader);

   }

   return totalLight * (1.0f - m.refl - m.alpha)
      + m.refl * reflectedLight+ m.alpha * refractedLight;
}

template <> 
__device__ vec3 shadeObject<0>(BVHTree *tree, 
      Light *lights[], int lightCount, int objIdx, 
      float intParam, Ray ray, Shader **shader) { return vec3(0.0f); }

__global__ void initScene(Geometry *geomList[], Plane *planeList[], Light *lights[], TKSphere *sphereTks, int numSpheres,
      TKPlane *planeTks, int numPlanes, TKTriangle *triangleTks, int numTris, TKBox *boxTks, int numBoxes,
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
         Material m(s.mod.pig.clr, s.mod.pig.f, f.amb, f.dif, f.spec, f.rough, f.refl, f.refr, f.ior);
         geomList[geomIdx++] = new Sphere(s.p, s.r, m, s.mod.trans, s.mod.invTrans);
      }

      for (int i = 0; i < numPlanes; i++) {
         const TKPlane &p = planeTks[i];
         const TKFinish &f = p.mod.fin;
         Material m(p.mod.pig.clr, p.mod.pig.f, f.amb, f.dif, f.spec, f.rough, f.refl, f.refr, f.ior);
         planeList[i] = new Plane(p.d, p.n, m, p.mod.trans, p.mod.invTrans);
      }

      for (int i = 0; i < numTris; i++) {
         const TKTriangle &t = triangleTks[i];
         const TKFinish f = t.mod.fin;
         Material m(t.mod.pig.clr, t.mod.pig.f, f.amb, f.dif, f.spec, f.rough, f.refl, f.refr, f.ior);
         geomList[geomIdx++] = new Triangle(t.p1, t.p2, t.p3, m, t.mod.trans, 
               t.mod.invTrans);
      }

      for (int i = 0; i < numBoxes; i++) {
         const TKBox &b = boxTks[i];
         const TKFinish f = b.mod.fin;
         Material m(b.mod.pig.clr, b.mod.pig.f, f.amb, f.dif, f.spec, f.rough, f.refl, f.refr, f.ior);
         geomList[geomIdx++] = new Box(b.p1, b.p2, m, b.mod.trans, b.mod.invTrans);
      }

      for (int i = 0; i < numSmthTris; i++) {
         const TKSmoothTriangle &t = smthTriTks[i];
         const TKFinish f = t.mod.fin;
         Material m(t.mod.pig.clr, t.mod.pig.f, f.amb, f.dif, f.spec, f.rough, f.refl, f.refr, f.ior);
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

typedef struct SortFrame {
   int size;
   Geometry **arr;
   __device__ SortFrame(int nSize = 0, Geometry **nArr = NULL) : size(nSize), arr(nArr) {}
} SortFrame;

__device__ inline void cudaSort(Geometry *list[], int end, int axis) {
   SortFrame stack[kMaxStackSize];
   int stackSize = 0;
   bool stackPopped = false;

   int size = end;
   Geometry **arr = list;
   while (1) {
      if (size == 1) {}
      else if (size == 2) {
         if (arr[0]->getCenter()[axis] < arr[1]->getCenter()[axis]) {
            SWAP(arr[0], arr[1]);
         }
      } else {
         if (!stackPopped) {
            int pivot = size / 2;
            SWAP(arr[pivot], arr[size - 1]);
            int topOfBottom = 0;
            for (int i = 0; i < size - 1; i++) {
               if(arr[i] < arr[size - 1]) {
                  SWAP(arr[i], arr[topOfBottom++]);   
               }             }
            stack[stackSize++] = SortFrame(size, arr); 

         }
      }


      if (stackSize == 0) break;
      arr = stack[stackSize - 1].arr;
      size = stack[stackSize - 1].size;
      stackSize--;
      stackPopped = true;
   }
}

__global__ void sortPieces(Geometry *geomList[], int geomCount, int div, int subDiv, int axis) {
   int idx = blockIdx.x * threadIdx.x;
   int size = subDiv;

   if (idx > div) return;

   if ((subDiv + 1) * idx > geomCount) {
      size = geomCount - subDiv * idx;
      if (size == 0) return;
   }
   cudaSort(geomList + subDiv * idx, subDiv, axis);
}

//crazy stuff
//__global__ void createBVH(Geometry *geomList[], int geomCount, Plane *planeList[], int planeCount, BVHTree *tree) {
//   //Change this back to static memory once I get things working
//   BVHStackEntry stack[kMaxStackSize];
//   tree->root = new BVHNode();
//   tree->planeList = planeList;
//   tree->planeListSize = planeCount;
//
//   
//}
__global__ void createBVH(Geometry *geomList[], int geomCount, Plane *planeList[], int planeCount, BVHTree *tree) {
   //Change this back to static memory once I get things working
   BVHStackEntry stack[kMaxStackSize];
   tree->root = new BVHNode();
   tree->planeList = planeList;
   tree->planeListSize = planeCount;

   BVHNode *cursor = tree->root;
   Geometry **arr = geomList;
   int listSize = geomCount;
   int axis = kXAxis;
   int stackSize = 0;

   // Call the BVHNode constructor

   do {
      if (stackSize >= kMaxStackSize) {
         printf("Stack completely full, aborting");
         return;
      }

      if (listSize == 1) {
         cursor->left = new BVHNode(arr[0]);
         // TODO this is creating a bounding box around a bounding box around 1 item
         cursor->bb = cursor->left->bb;
      } else if (listSize == 2) {
         cursor->left = new BVHNode(arr[0]);
         cursor->right = new BVHNode(arr[1]);
         cursor->bb = combineBoundingBox(cursor->left->bb, cursor->right->bb);
      } else {
         // If the leftside is empty, recursively create that first
         if (!cursor->left) {
            cudaSort(arr, listSize, axis);
            cursor->left = new BVHNode();

            stack[stackSize++] = BVHStackEntry(arr, cursor, listSize, axis);

            cursor = cursor->left;
            listSize = listSize / 2;
            axis = (axis + 1) % kAxisNum;
            continue;
         // Otherwise make the rightside
         } else if (!cursor->right) {
            cursor->right = new BVHNode();

            stack[stackSize++] = BVHStackEntry(arr, cursor, listSize, axis);

            cursor = cursor->right;
            arr = arr + listSize / 2;
            listSize = (listSize - 1) / 2 + 1;
            axis = (axis + 1) % kAxisNum;
            continue;
         } else {
            cursor->bb = combineBoundingBox(cursor->left->bb, cursor->right->bb);
         }

      }

      if (stackSize > 0) {
         // Pop the stack
         cursor = stack[stackSize - 1].cursor;
         listSize = stack[stackSize - 1].listSize;
         arr = stack[stackSize - 1].arr;
         axis = stack[stackSize - 1].axis;
      }
      stackSize--; 
   } while (stackSize >= 0); 
}

__global__ void deleteScene(Geometry *geomList[], int geomCount, Light *lightList[], int lightCount, Shader **shader) {
   // This should really only be run with one thread and block anyways, but this is a safety check
   if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
      delete *shader;

      for (int i = 0; i < geomCount; i++) {
         delete geomList[i];
      }

      for (int i = 0; i < lightCount; i++) {
         delete lightList[i];
      }
   }
}

__global__ void rayTrace(int resWidth, int resHeight, Camera cam,
      BVHTree *tree, Light *lights[], int lightCount,  
      vec3 output[], Shader **shader) {

   int x = blockIdx.x * blockDim.x + threadIdx.x;
   int y = blockIdx.y * blockDim.y + threadIdx.y;

   if (x >= resWidth || y >= resHeight)
      return;

   int index = y * resWidth + x;

   // Generate rays
   //Image space coordinates 
   float u = 2.0f * (x / (float)resWidth) - 1.0f; 
   float v = 2.0f * (y / (float)resHeight) - 1.0f;

   // .5f is because the magnitude of cam.right and cam.up should be equal
   // to the width and height of the image plane in world space
   vec3 rPos = u *.5f * cam.right + v * .5f * cam.up + cam.pos;
   vec3 rDir = rPos - cam.pos + cam.lookAtDir;
   Ray ray(rPos, rDir);

   float t;
   Geometry *closestGeom;
   getClosestIntersection(ray, tree, &closestGeom, &t);

   if (closestGeom != kNoShapeFound) {
      vec3 totalColor = shadeObject<kMaxRecurse>(tree, lights, lightCount, 
            closestGeom, t, ray, shader);

      output[index] = vec3(clamp(totalColor.x, 0, 1), 
                           clamp(totalColor.y, 0, 1), 
                           clamp(totalColor.z, 0, 1)); 
   } else {
      output[index] = vec3(0.0f);
   }
}

__global__ void averageBuffer(int resWidth, int resHeight, int sampleCountSqrRoot, uchar4 *output, vec3 *antiAliasBuffer) {
   int x = blockIdx.x * blockDim.x + threadIdx.x;
   int y = blockIdx.y * blockDim.y + threadIdx.y;
   uchar4 clr;
   
   int outputIndex = y * resWidth + x;

   if (x >= resWidth || y >= resHeight)
      return;

   vec3 endColor(0.0f);
   for (int xOffset = 0; xOffset < sampleCountSqrRoot; xOffset++) {
      for (int yOffset = 0; yOffset < sampleCountSqrRoot; yOffset++) {
         int bufferIndex = x * sampleCountSqrRoot + xOffset + (y * sampleCountSqrRoot + yOffset) * resWidth * sampleCountSqrRoot;
         endColor += antiAliasBuffer[bufferIndex];
      }
   }
   endColor /= sampleCountSqrRoot * sampleCountSqrRoot;
   endColor *= 255;

   clr.x = endColor.x; clr.y = endColor.y; clr.z = endColor.z; clr.w = 255;
   output[outputIndex] = clr; 
}

void allocateGPUScene(TKSceneData *data, Geometry ***dGeomList, Plane ***dPlaneList, 
      Light ***dLightList, int *retGeometryCount, int *retPlaneCount,
      int *retLightCount, Shader **dShader, ShadingType stype) {
   int geometryCount = 0;
   int lightCount = 0;

   TKSphere *dSphereTokens = NULL;
   TKPlane *dPlaneTokens = NULL;
   TKPointLight *dPointLightTokens = NULL;
   TKTriangle *dTriangleTokens = NULL;
   TKSmoothTriangle *dSmthTriTokens = NULL;
   TKBox *dBoxTokens = NULL;

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
      *retPlaneCount = planeCount;
   }

   int triangleCount = data->triangles.size();
   if (triangleCount > 0) {
      HANDLE_ERROR(cudaMalloc(&dTriangleTokens, sizeof(TKTriangle) * triangleCount));
      HANDLE_ERROR(cudaMemcpy(dTriangleTokens, &data->triangles[0], 
               sizeof(TKTriangle) * triangleCount, cudaMemcpyHostToDevice));
      geometryCount += triangleCount;
   }

   int boxCount = data->boxes.size();
   if (boxCount > 0) {
      HANDLE_ERROR(cudaMalloc(&dBoxTokens, sizeof(TKBox) * boxCount));
      HANDLE_ERROR(cudaMemcpy(dBoxTokens, &data->boxes[0],
               sizeof(TKBox) * boxCount, cudaMemcpyHostToDevice));
      geometryCount += boxCount;
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
   HANDLE_ERROR(cudaMalloc(dPlaneList, sizeof(Plane *) * planeCount));
   HANDLE_ERROR(cudaMalloc(dLightList, sizeof(Light *) * lightCount));

   // Fill up GeomList and LightList with actual objects on the GPU
   initScene<<<1, 1>>>(*dGeomList, *dPlaneList, *dLightList, dSphereTokens, sphereCount, 
         dPlaneTokens, planeCount, dTriangleTokens, triangleCount, dBoxTokens, boxCount, 
         dSmthTriTokens, smoothTriangleCount, dPointLightTokens, pointLightCount, 
         dShader, stype);

   if (dSphereTokens) HANDLE_ERROR(cudaFree(dSphereTokens));
   if (dPlaneTokens) HANDLE_ERROR(cudaFree(dPlaneTokens));
   if (dTriangleTokens) HANDLE_ERROR(cudaFree(dTriangleTokens));
   if (dSmthTriTokens) HANDLE_ERROR(cudaFree(dSmthTriTokens));
   if (dBoxTokens) HANDLE_ERROR(cudaFree(dBoxTokens));


   *retGeometryCount = geometryCount;
   *retLightCount = lightCount;
}

void freeGPUScene(Geometry **dGeomList, int geomCount, Light **dLightList, 
      int lightCount, Shader **shader) {
   deleteScene<<<1, 1>>>(dGeomList, geomCount, dLightList, lightCount, shader);

   HANDLE_ERROR(cudaFree(dGeomList));
   HANDLE_ERROR(cudaFree(dLightList));
   HANDLE_ERROR(cudaFree(shader));
}

extern "C" void launch_kernel(TKSceneData *data, ShadingType stype, int width, 
      int height, uchar4 *output, int sampleCount) {
   Geometry **dGeomList; 
   Plane **dPlaneList; 
   Light **dLightList;
   Shader **dShader;

   vec3 *dAntiAliasBuffer;
   uchar4 *dOutput;

   BVHTree *dBvhTree;

   int geometryCount;
   int planeCount;
   int lightCount;

   int sqrSampleCount = sqrt(sampleCount);
   if (sqrSampleCount * sqrSampleCount != sampleCount) {
      printf("Invalid sample count: %d. Sample count for anti aliasing must have an integer square root");
      return;
   }

   TKCamera camTK = *data->camera;
   Camera camera(camTK.pos, camTK.up, camTK.right, 
                 glm::normalize(camTK.lookAt - camTK.pos));

   // Fill the geomList and light list with objects dynamically created on the GPU
   HANDLE_ERROR(cudaMalloc(&dShader, sizeof(Shader*)));
   HANDLE_ERROR(cudaMalloc(&dOutput, sizeof(uchar4) * width * height));
   HANDLE_ERROR(cudaMalloc(&dAntiAliasBuffer, sizeof(vec3) * width * height * sampleCount));
   allocateGPUScene(data, &dGeomList, &dPlaneList, &dLightList, &geometryCount, &planeCount, &lightCount, dShader, stype);
   cudaDeviceSynchronize();
   checkCUDAError("AllocateGPUScene failed");

   HANDLE_ERROR(cudaMalloc(&dBvhTree, sizeof(BVHTree)));
   createBVH<<<1, 1>>>(dGeomList, geometryCount, dPlaneList, planeCount, dBvhTree);
   cudaDeviceSynchronize();
   checkCUDAError("CreateBVH failed");

   // Crazy stuff
   /*int div = 2;
   int subDivs;
   int axis = kXAxis;
   do {
      subDivs = geometryCount / div;
      dim3 dimBlock(kBlockWidth * kBlockWidth);
      dim3 dimGrid((div - 1) / kBlockWidth + 1);
      sortPieces<<<dimBlock, dimGrid>>>(dGeomList, geometryCount, div, subDivs, axis);
      axis = (axis + 1) % kAxisNum;
      
      div *= 2;
      subDivs = geometryCount / div;
   } while ( subDivs > 2);*/

   int antiAliasWidth = width * sqrSampleCount;
   int antiAliasHeight = height * sqrSampleCount;
   dim3 dimBlock(kBlockWidth, kBlockWidth);
   dim3 dimGrid((antiAliasWidth - 1) / kBlockWidth + 1, (antiAliasHeight- 1) / kBlockWidth + 1);
   rayTrace<<<dimGrid, dimBlock>>>(width * sqrSampleCount, height * sqrSampleCount, camera, 
         dBvhTree, dLightList, lightCount, dAntiAliasBuffer, dShader);
   cudaDeviceSynchronize();
   checkCUDAError("RayTrace kernel failed");

   dimGrid = dim3((width - 1) / kBlockWidth + 1, (height - 1) / kBlockWidth + 1);
   averageBuffer<<<dimGrid, dimBlock>>>(width, height, sqrSampleCount, dOutput, dAntiAliasBuffer);
   cudaDeviceSynchronize();
   checkCUDAError("averageBuffer kernel failed");

   // Clean up
   //freeGPUScene(dGeomList, geometryCount, dLightList, lightCount, dShader);
   HANDLE_ERROR(cudaMemcpy(output, dOutput, 
            sizeof(uchar4) * width * height, cudaMemcpyDeviceToHost));
   HANDLE_ERROR(cudaFree(dOutput));
}

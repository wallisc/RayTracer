#include <algorithm>
#include <vector>
#include <map>
#include <stdio.h>
#include <float.h>

#include "glm/gtc/matrix_transform.hpp"

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

#include "bvh.cpp"

#define kNoShapeFound NULL

const float kMaxDist = FLT_MAX;
using glm::vec3;
using std::vector;
using std::pair;

texture<uchar4, 2, cudaReadModeNormalizedFloat> mytex;

// Only works with 24 bit images that are a power of 2
unsigned char* readBMP(char* filename, int *retWidth, int *retHeight)
{
   int i;
   FILE* f = fopen(filename, "rb");
   unsigned char info[54];
   fread(info, sizeof(unsigned char), 54, f); // read the 54-byte header

   // extract image height and width from header
   int width = *(int*)&info[18];
   int height = *(int*)&info[22];

   int size = 3 * width * height;
   unsigned char* data = new unsigned char[size]; // allocate 3 bytes per pixel
   unsigned char* retData = new unsigned char[size + width * height]; // allocate 4 bytes per pixel
   fread(data, sizeof(unsigned char), size, f); // read the rest of the data at once
   fclose(f);

   for(i = 0; i < width * height; i++)
   {
      retData[4 * i] = data[3 * i + 2];
      retData[4 * i + 1] = data[3 * i + 1];
      retData[4 * i + 2] = data[3 * i];
      retData[4 * i + 3] = 0;
   }

   delete data;
   *retWidth = width;
   *retHeight = height;
   return retData;
}


const bool kLeft = 0;
const bool kRight = 1;

typedef struct StackEntry {
   bool nextDir;
   BVHNode *node;
   __device__ StackEntry(BVHNode *stackNode = NULL, char nextDirection = 0) : node(stackNode), nextDir(nextDirection) {}
} StackEntry;

// Find the closest shape. The index of the intersecting object is stored in
// retOjIdx and the t-value along the input ray is stored in retParam
//
// toBeat can be set to a float value if you want to short-circuit as soon
// as you find an object closer than toBeat
//
// If no intersection is found, retObjIdx is set to 'kNoShapeFound'
__device__ void getClosestIntersection(const Ray &ray, BVHTree *tree, 
      Geometry **retObj, float *retParam, float toBeat = -FLT_MAX) {
   float t = kMaxDist;
   Geometry *closestGeom = kNoShapeFound;

   int maxDepth = 0;

   StackEntry stack[kMaxStackSize];
   int stackSize = 0;
   bool justPoppedStack = false;

   BVHNode *cursor = tree->root;
   bool nextDir;
     
   do {
      if (stackSize >= kMaxStackSize) {
         printf("Stack full, aborting!\n");
         return;
      }
         
      // If at a leaf
      if (cursor->geom) {
         maxDepth = max(maxDepth, stackSize);
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
               if (t < toBeat) {
                  *retObj = closestGeom;
                  *retParam = t;
                  return;
               }
            }
         // Otherwise, if one face is front of the current one
         } else {
            if (dist < t && isFloatAboveZero(dist)) {
               t = dist;
               closestGeom = cursor->geom;
               if (t < toBeat) {
                  *retObj = closestGeom;
                  *retParam = t;
                  return;
               }
            }
         }
      // If not on a leaf and neither branch has been explored
      } else if (!justPoppedStack) { 
         float left = cursor->left->bb.getIntersection(ray);

         if (!cursor->right && isFloatAboveZero(left) && left < t) {
            cursor = cursor->left;
            justPoppedStack = false;
            continue;
         }

         // Go down the tree with the closest bounding box
         float right = cursor->right->bb.getIntersection(ray);

         if (isFloatAboveZero(right) && (right <= left || !isFloatAboveZero(left)) && right < t) {
            if (isFloatAboveZero(left)) stack[stackSize++] = StackEntry(cursor, kLeft);
            cursor = cursor->right;
            justPoppedStack = false;
            continue;
         } else if (isFloatAboveZero(left) && (left <= right || !isFloatAboveZero(right)) && left < t) {
            if (isFloatAboveZero(right)) stack[stackSize++] = StackEntry(cursor, kRight);
            cursor = cursor->left;
            justPoppedStack = false;
            continue;
         } 
      // If coming back from a 'recursion' and one of the branches hasn't been explored
      } else {
         if (nextDir == kRight) {
            float right = cursor->right->bb.getIntersection(ray);
            if (right < t) {
               cursor = cursor->right;
               justPoppedStack = false;
               continue;
            }
         } else {
            float left = cursor->left->bb.getIntersection(ray);
            if (left < t) {
               cursor = cursor->left;
               justPoppedStack = false;
               continue;
            }
         }
      }

      if(stackSize == 0) {
         break;
      }

      // Pop the stack
      cursor = stack[stackSize - 1].node; 
      nextDir = stack[stackSize - 1].nextDir;
      justPoppedStack = true;
      stackSize--;
   } while(true);

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

__device__ bool isInShadow(const Ray &shadow, BVHTree *tree, float intersectParam) {
   float closestIntersect;
   Geometry *closestObj;
   getClosestIntersection(shadow, tree, &closestObj, &closestIntersect, intersectParam);
   return isFloatLessThan(closestIntersect, intersectParam);
}

template <int invRecLevel>
__device__ glm::vec3 getReflection(glm::vec3 point, glm::vec3 normal, 
      glm::vec3 eyeVec, BVHTree *tree, Light *lights[], int lightCount, 
      Shader **shader, curandState *randState) {

   Ray reflectRay(point, 2.0f * glm::dot(normal, eyeVec) * normal - eyeVec);
   // Offset the ray so there's no self-intersection
   reflectRay.o += BIG_EPSILON * reflectRay.d;
   Geometry *closestGeom;
   float t;

   getClosestIntersection(reflectRay, tree, &closestGeom, &t);
   if (closestGeom != kNoShapeFound) {
      return shadeObject<invRecLevel>(tree, 
            lights, lightCount,
            closestGeom, t,
            reflectRay, shader, randState);
   } 
   return vec3(0.0f);
}

template <>
__device__ glm::vec3 getReflection<0>(glm::vec3 point, glm::vec3 normal, 
      glm::vec3 eyeVec, BVHTree *tree, Light *lights[], int lightCount, 
      Shader **shader, curandState *randState) { return vec3(0.0f); }

template <int invRecLevel>
__device__ glm::vec3 getRefraction(glm::vec3 point, glm::vec3 normal, float ior, glm::vec3 eyeVec, 
   BVHTree *tree, Light *lights[], int lightCount, Shader **shader, curandState *randState) {

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
               refracRay, shader, randState);
      }
   } 
   return vec3(0.0f);
}



template <>
__device__ glm::vec3 getRefraction<0>(glm::vec3 point, glm::vec3 normal, float ior, glm::vec3 eyeVec, 
      BVHTree *tree, Light *lights[], int lightCount, Shader **shader, curandState *randState) 
   { return vec3(0.0f); }

__device__ vec3 cosineWeightedSample(vec3 normal, float rand1, float rand2) {
   float distFromCenter = 1.0f - sqrt(rand1);
   float theta = 2.0f * M_PI * rand2;
   float phi = M_PI / 2.0f - acos(distFromCenter);

   float phiDeg = phi * 180.0f / M_PI;
   float thetaDeg = theta * 180.0f / M_PI;

   vec3 outV = normal.x < .99f ? glm::cross(normal, vec3(1.0f, 0.0, 0.0)) : vec3(0.0f, 1.0f, 0.0f); 
   glm::mat4 rot1 = glm::rotate(glm::mat4(1.0f), phiDeg, outV);
   glm::mat4 rot2 = glm::rotate(glm::mat4(1.0f), thetaDeg, normal);
   glm::vec4 norm(normal.x, normal.y, normal.z, 0.0f);
   
   return vec3(rot2 * rot1 * norm);
}

template<int invRecLevel>
__device__ glm::vec3 getIndirect(vec3 point, vec3 normal, BVHTree *tree, Light *lights[], int lightCount, 
      Shader **shader, curandState *randState) {
   vec3 totalColor(0.0f);

   float sampleRange = 1.0f / kMonteCarloSamplesRoot;

   for (int xSample = 0; xSample < kMonteCarloSamplesRoot; xSample++) {
      for (int ySample = 0; ySample < kMonteCarloSamplesRoot; ySample++) {
         float rand1 = curand_uniform(randState) * sampleRange + xSample * sampleRange; 
         float rand2 = curand_uniform(randState) * sampleRange + ySample * sampleRange;
         vec3 dir = cosineWeightedSample(normal, rand1, rand2);
         Ray mcRay(point, dir);
         Geometry *geom;
         float t;
         getClosestIntersection(mcRay, tree, &geom, &t);
         if (geom != kNoShapeFound) { 
            vec3 c = shadeObject<1>(tree, lights, lightCount, geom, t, mcRay, shader, randState);
            totalColor += c / ((float)kMonteCarloSamples);
         }
      }
   }
   return totalColor;
}

template<>
__device__ glm::vec3 getIndirect<0>(vec3 point, vec3 normal, BVHTree *tree, Light *lights[], int lightCount, 
      Shader **shader, curandState *randState) { return vec3(0.0f); }

__device__ glm::vec3 getColor(Geometry *geom, Ray ray, float param) {
   Material m = geom->getMaterial();
   if (m.texId == NO_TEXTURE) {
      return m.clr;
   } else {
      glm::vec2 uv = geom->UVAt(ray, param);
      float4 clr = tex2D(mytex, uv.x, uv.y);
      return vec3(clr.x, clr.y, clr.z);
   }
}

//Note: The ray parameter must stay as a copy (not a reference) 
template <int invRecLevel> 
__device__ vec3 shadeObject(BVHTree *tree, 
      Light *lights[], int lightCount, Geometry* geom, 
      float intParam, Ray ray, Shader **shader, curandState *randStates) {

   glm::vec3 intersectPoint = ray.getPoint(intParam);
   Material m = geom->getMaterial();
   vec3 normal = geom->getNormalAt(ray, intParam);
   vec3 matClr = getColor(geom, ray, intParam);
   vec3 eyeVec = glm::normalize(-ray.d);
   vec3 totalLight(0.0f);

   for(int lightIdx = 0; lightIdx < lightCount; lightIdx++) {
      vec3 light = lights[lightIdx]->getLightAtPoint(intersectPoint);
      vec3 lightDir = lights[lightIdx]->getLightDir(intersectPoint);
      Ray shadow = lights[lightIdx]->getShadowFeeler(intersectPoint);
      float intersectParam = geom->getIntersection(shadow);
      bool inShadow = isInShadow(shadow, tree, intersectParam); 

      
      totalLight += (*shader)->shade(matClr, m.amb, m.dif, m.spec, m.rough, 
            eyeVec, lightDir, light, normal, 
            inShadow);
   }

   vec3 reflectedLight(0.0f);
   if (m.refl > 0.0f && invRecLevel - 1 > 0) {
      reflectedLight = getReflection<invRecLevel - 1>(intersectPoint, 
         normal, eyeVec, tree, lights, lightCount, shader, randStates);
   }

   vec3 refractedLight(0.0f);
   if (m.refr > 0.0f && invRecLevel - 1 > 0) {
      refractedLight = getRefraction<invRecLevel - 1>(intersectPoint, 
         normal, m.ior, eyeVec, tree, lights, lightCount, shader, randStates);

   }

   //vec3 indirectLight = getIndirect<invRecLevel - 1>(intersectPoint + normal * BIG_EPSILON, normal, tree, lights, lightCount, shader, randStates);

   return totalLight * (1.0f - m.refl - m.alpha)
      + m.refl * reflectedLight+ m.alpha * refractedLight;// + m.clr * indirectLight;
}

template <> 
__device__ vec3 shadeObject<0>(BVHTree *tree, 
      Light *lights[], int lightCount, Geometry *geom, 
      float intParam, Ray ray, Shader **shader, curandState *randStates) { return vec3(0.0f); }

__global__ void initScene(Geometry *geomList[], Plane *planeList[], 
      Light *lights[], TKSphere *sphereTks, int numSpheres, TKPlane *planeTks, 
      int numPlanes, TKTriangle *triangleTks, int numTris, TKBox *boxTks, 
      int numBoxes, TKSmoothTriangle *smthTriTks, int numSmthTris, 
      TKPointLight *pLightTks, int numPointLights, Shader **shader, 
      ShadingType stype) {

   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   int gridSize = gridDim.x * blockDim.x;
   int geomListSize = 0;
   int lightListSize = 0;

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
   }

   for (int planeIdx = idx; planeIdx < numPlanes; planeIdx += gridSize) {
      const TKPlane &p = planeTks[planeIdx];
      const TKFinish &f = p.mod.fin;
      Material m(p.mod.pig.clr, p.mod.pig.f, f.amb, f.dif, f.spec, f.rough, f.refl, f.refr, f.ior);
      planeList[planeIdx] = new Plane(p.d, p.n, m, p.mod.trans, p.mod.invTrans);
   }

   // Add all the geometry
   for (int sphereIdx = idx; sphereIdx < numSpheres; sphereIdx += gridSize) {
      const TKSphere &s = sphereTks[sphereIdx];
      const TKFinish f = s.mod.fin;
      Material m(s.mod.pig.clr, s.mod.pig.f, f.amb, f.dif, f.spec, f.rough, f.refl, f.refr, f.ior);
      geomList[sphereIdx + geomListSize] = new Sphere(s.p, s.r, m, s.mod.trans, s.mod.invTrans);
      if (!geomList[sphereIdx + geomListSize]) printf("Error at %d\n", sphereIdx + geomListSize);
   }
   geomListSize += numSpheres;

   for (int triIdx = idx; triIdx < numTris; triIdx += gridSize) {
      const TKTriangle &t = triangleTks[triIdx];
      const TKFinish f = t.mod.fin;
      Material m(t.mod.pig.clr, t.mod.pig.f, f.amb, f.dif, f.spec, f.rough, f.refl, f.refr, f.ior, t.mod.pig.texId);
      geomList[triIdx + geomListSize] = new Triangle(t.p1, t.p2, t.p3, m, t.mod.trans, 
            t.mod.invTrans, t.vt1, t.vt2, t.vt3);
      if (!geomList[triIdx + geomListSize]) printf("Error at %d\n", triIdx + geomListSize);
   }
   geomListSize += numTris;

   for (int boxIdx = idx; boxIdx < numBoxes; boxIdx += gridSize) {
      const TKBox &b = boxTks[boxIdx];
      const TKFinish f = b.mod.fin;
      Material m(b.mod.pig.clr, b.mod.pig.f, f.amb, f.dif, f.spec, f.rough, f.refl, f.refr, f.ior);
      geomList[boxIdx + geomListSize] = new Box(b.p1, b.p2, m, b.mod.trans, b.mod.invTrans);
      if (!geomList[boxIdx + geomListSize]) printf("Error at %d\n", boxIdx + geomListSize);
   }
   geomListSize += numBoxes;

   for (int smTriIdx = idx; smTriIdx < numSmthTris; smTriIdx += gridSize) {
      const TKSmoothTriangle &t = smthTriTks[smTriIdx];
      const TKFinish f = t.mod.fin;
      Material m(t.mod.pig.clr, t.mod.pig.f, f.amb, f.dif, f.spec, f.rough, f.refl, f.refr, f.ior, t.mod.pig.texId);
      geomList[smTriIdx + geomListSize] = new SmoothTriangle(t.p1, t.p2, t.p3, t.n1, t.n2, t.n3, 
            m, t.mod.trans, t.mod.invTrans, t.vt1, t.vt2, t.vt3);
      if (!geomList[smTriIdx + geomListSize]) printf("Error at %d\n", smTriIdx + geomListSize);

   }
   geomListSize += numSmthTris;

   for (int pointLightIdx = idx; pointLightIdx < numPointLights; pointLightIdx += gridSize) {
      TKPointLight &p = pLightTks[pointLightIdx];
      lights[pointLightIdx + lightListSize] = new PointLight(p.pos, p.clr);
   }
   lightListSize += numPointLights;

}

__global__ void initCurand(curandState randStates[], int resWidth, int resHeight) {
   int x = blockIdx.x * blockDim.x + threadIdx.x;
   int y = blockIdx.y * blockDim.y + threadIdx.y;

   if (x >= resWidth || y >= resHeight)
      return;

   int index = y * resWidth + x;
   curand_init(index, 0, 0, &randStates[index]);
}

__global__ void generateCameraRays(int resWidth, int resHeight, Camera cam, Ray rayQueue[], curandState randStates[]) {
   int x = blockIdx.x * blockDim.x + threadIdx.x;
   int y = blockIdx.y * blockDim.y + threadIdx.y;

   if (x >= resWidth || y >= resHeight)
      return;

   int index = y * resWidth + x;

   // Generate rays
   //Image space coordinates 
   float uJitter = (curand_uniform(&randStates[index]) - .5f) / (float)resWidth; // Passing in arbitrary seed values
   float vJitter = (curand_uniform(&randStates[index]) - .5f) / (float)resHeight;
   float u = 2.0f * (x / (float)resWidth) - 1.0f + uJitter; 
   float v = 2.0f * (y / (float)resHeight) - 1.0f + vJitter; 

   // .5f is because the magnitude of cam.right and cam.up should be equal
   // to the width and height of the image plane in world space
   vec3 rPos = u *.5f * cam.right + v * .5f * cam.up + cam.pos;
   vec3 rDir = rPos - cam.pos + cam.lookAtDir;
   rayQueue[index] = Ray(rPos, rDir);
}

__device__ vec3 addDirectLight(const Ray &eyeRay, Light *lights[], int lightCount) {
   glm::vec3 totClr(0.0f); 
   for (int lightIdx = 0; lightIdx < lightCount; lightIdx++) {
      float lightPow = glm::dot(lights[lightIdx]->getLightDir(eyeRay.o), eyeRay.d);

      //lightPow = pow(lightPow, 18)
      lightPow = lightPow * lightPow * lightPow * lightPow * lightPow * lightPow;
      lightPow *= lightPow * lightPow * lightPow * lightPow * lightPow * lightPow;
      lightPow *= lightPow * lightPow * lightPow * lightPow * lightPow * lightPow;

      totClr += lightPow * lights[lightIdx]->getLightAtPoint(eyeRay.o); 
   }
   return totClr;
}

__global__ void rayTrace(int resWidth, int resHeight, Ray rayQueue[],
      BVHTree *tree, Light *lights[], int lightCount,  
      vec3 output[], Shader **shader, curandState randStates[]) {

   int x = blockIdx.x * blockDim.x + threadIdx.x;
   int y = blockIdx.y * blockDim.y + threadIdx.y;

   if (x >= resWidth || y >= resHeight)
      return;

   int index = y * resWidth + x;

   Ray ray = rayQueue[index];

   float t;
   Geometry *closestGeom;
   getClosestIntersection(ray, tree, &closestGeom, &t);

   if (closestGeom != kNoShapeFound) {
      vec3 totalColor = shadeObject<kMaxRecurse>(tree, lights, lightCount, 
            closestGeom, t, ray, shader, &randStates[index]);

      output[index] = vec3(clamp(totalColor.x, 0, 1), 
                           clamp(totalColor.y, 0, 1), 
                           clamp(totalColor.z, 0, 1)); 
   } else {
      output[index] = vec3(0.0f);
   }
}

__global__ void averageBufferColors(int resWidth, int resHeight, 
      int sampleCountSqrRoot, uchar4 *output, vec3 *antiAliasBuffer) {

   int x = blockIdx.x * blockDim.x + threadIdx.x;
   int y = blockIdx.y * blockDim.y + threadIdx.y;
   uchar4 clr;
   
   int outputIndex = y * resWidth + x;

   if (x >= resWidth || y >= resHeight)
      return;

   vec3 endColor(0.0f);
   for (int xOffset = 0; xOffset < sampleCountSqrRoot; xOffset++) {
      for (int yOffset = 0; yOffset < sampleCountSqrRoot; yOffset++) {
         int bufferIndex = x * sampleCountSqrRoot + xOffset + 
               (y * sampleCountSqrRoot + yOffset) * resWidth * sampleCountSqrRoot;
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
   int biggestListSize = 0;

   int imgWidth, imgHeight;
   unsigned char *texData = readBMP("blitz.bmp", &imgWidth, &imgHeight);

   int imgSize = sizeof(uchar4) * imgWidth * imgHeight;
   cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);

   cudaArray* cu_array;
   cudaMallocArray(&cu_array, &channelDesc, imgWidth, imgHeight );

   //copy image to device array cu_array – used as texture mytex on device
   HANDLE_ERROR(cudaMemcpyToArray(cu_array, 0, 0, texData, imgSize, cudaMemcpyHostToDevice));
   // set texture parameters
   
   mytex.addressMode[0] = cudaAddressModeWrap;
   mytex.addressMode[1] = cudaAddressModeWrap;
   mytex.filterMode = cudaFilterModeLinear;
   mytex.normalized = true; 

   // Bind the array to the texture
   HANDLE_ERROR(cudaBindTextureToArray(mytex, cu_array, channelDesc));

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
      if (sphereCount > biggestListSize) biggestListSize = sphereCount;
   }

   int planeCount = data->planes.size();
   if (planeCount > 0) {
      HANDLE_ERROR(cudaMalloc(&dPlaneTokens, sizeof(TKPlane) * planeCount));
      HANDLE_ERROR(cudaMemcpy(dPlaneTokens, &data->planes[0],
               sizeof(TKPlane) * planeCount, cudaMemcpyHostToDevice));
      if (planeCount > biggestListSize) biggestListSize = planeCount;
   }

   int triangleCount = data->triangles.size();
   if (triangleCount > 0) {
      HANDLE_ERROR(cudaMalloc(&dTriangleTokens, sizeof(TKTriangle) * triangleCount));
      HANDLE_ERROR(cudaMemcpy(dTriangleTokens, &data->triangles[0], 
               sizeof(TKTriangle) * triangleCount, cudaMemcpyHostToDevice));
      geometryCount += triangleCount;
      if (triangleCount > biggestListSize) biggestListSize = triangleCount;
   }

   int boxCount = data->boxes.size();
   if (boxCount > 0) {
      HANDLE_ERROR(cudaMalloc(&dBoxTokens, sizeof(TKBox) * boxCount));
      HANDLE_ERROR(cudaMemcpy(dBoxTokens, &data->boxes[0],
               sizeof(TKBox) * boxCount, cudaMemcpyHostToDevice));
      geometryCount += boxCount;
      if (boxCount > biggestListSize) biggestListSize = boxCount;
   }

   int smoothTriangleCount = data->smoothTriangles.size();
   if (smoothTriangleCount > 0) {
      HANDLE_ERROR(cudaMalloc(&dSmthTriTokens, 
               sizeof(TKSmoothTriangle) * smoothTriangleCount));
      HANDLE_ERROR(cudaMemcpy(dSmthTriTokens, &data->smoothTriangles[0],
               sizeof(TKSmoothTriangle) * smoothTriangleCount, cudaMemcpyHostToDevice));
      geometryCount += smoothTriangleCount;
      if (smoothTriangleCount > biggestListSize) biggestListSize = smoothTriangleCount;
   }

   int pointLightCount = data->pointLights.size();
   if (pointLightCount > 0) {
      HANDLE_ERROR(cudaMalloc(&dPointLightTokens, 
               sizeof(TKPointLight) * pointLightCount));
      HANDLE_ERROR(cudaMemcpy(dPointLightTokens, &data->pointLights[0],
               sizeof(TKPointLight) * pointLightCount, cudaMemcpyHostToDevice));
      lightCount += pointLightCount;
      if (pointLightCount > biggestListSize) biggestListSize = pointLightCount;
   }

   HANDLE_ERROR(cudaMalloc(dGeomList, sizeof(Geometry *) * geometryCount));
   HANDLE_ERROR(cudaMalloc(dPlaneList, sizeof(Plane *) * planeCount));
   HANDLE_ERROR(cudaMalloc(dLightList, sizeof(Light *) * lightCount));

   int blockSize = kBlockWidth * kBlockWidth;
   int gridSize = (biggestListSize - 1) / blockSize + 1;
   // Fill up GeomList and LightList with actual objects on the GPU
   initScene<<<gridSize, blockSize>>>(*dGeomList, *dPlaneList, *dLightList, 
         dSphereTokens, sphereCount, dPlaneTokens, planeCount, dTriangleTokens, 
         triangleCount, dBoxTokens, boxCount, dSmthTriTokens, smoothTriangleCount, 
         dPointLightTokens, pointLightCount, dShader, stype);

   cudaDeviceSynchronize();
   checkCUDAError("initScene failed");

   if (dSphereTokens) HANDLE_ERROR(cudaFree(dSphereTokens));
   if (dPlaneTokens) HANDLE_ERROR(cudaFree(dPlaneTokens));
   if (dTriangleTokens) HANDLE_ERROR(cudaFree(dTriangleTokens));
   if (dSmthTriTokens) HANDLE_ERROR(cudaFree(dSmthTriTokens));
   if (dBoxTokens) HANDLE_ERROR(cudaFree(dBoxTokens));

   *retGeometryCount = geometryCount;
   *retLightCount = lightCount;
   *retPlaneCount = planeCount;
}

extern "C" void launch_kernel(TKSceneData *data, ShadingType stype, int width, 
      int height, uchar4 *output, int sampleCount) {
   Geometry **dGeomList; 
   Plane **dPlaneList; 
   Light **dLightList;
   Shader **dShader;
   Ray *dRayQueue;
   curandState *dRandStates;

   vec3 *dAntiAliasBuffer;
   uchar4 *dOutput;

   BVHTree *dBvhTree;

   int geometryCount;
   int planeCount;
   int lightCount;

   int sqrSampleCount = sqrt(sampleCount);
   if (sqrSampleCount * sqrSampleCount != sampleCount) {
      printf("Invalid sample count: %d. "
             "Sample count for anti aliasing must have an integer square root");
      return;
   }

   HANDLE_ERROR(cudaDeviceSetLimit(cudaLimitMallocHeapSize, kGimmeLotsOfMemory));

   TKCamera camTK = *data->camera;
   Camera camera(camTK.pos, camTK.up, camTK.right, 
                 glm::normalize(camTK.lookAt - camTK.pos));

   // Fill the geomList and light list with objects dynamically created on the GPU
   HANDLE_ERROR(cudaMalloc(&dShader, sizeof(Shader*)));
   HANDLE_ERROR(cudaMalloc(&dOutput, sizeof(uchar4) * width * height));
   HANDLE_ERROR(cudaMalloc(&dAntiAliasBuffer, 
         sizeof(vec3) * width * height * sampleCount));

   allocateGPUScene(data, &dGeomList, &dPlaneList, &dLightList, &geometryCount, 
         &planeCount, &lightCount, dShader, stype);

   cudaDeviceSynchronize();
   checkCUDAError("AllocateGPUScene failed");

   HANDLE_ERROR(cudaMalloc(&dBvhTree, sizeof(BVHTree)));
   formBVH(dGeomList, geometryCount, dPlaneList, planeCount, dBvhTree);

   int antiAliasWidth = width * sqrSampleCount;
   int antiAliasHeight = height * sqrSampleCount;
   HANDLE_ERROR(cudaMalloc(&dRayQueue, sizeof(Ray) * antiAliasWidth * antiAliasHeight));
   HANDLE_ERROR(cudaMalloc(&dRandStates, sizeof(curandState) * antiAliasWidth * antiAliasHeight));


   dim3 dimBlock(kBlockWidth, kBlockWidth);
   dim3 dimGrid((antiAliasWidth - 1) / kBlockWidth + 1, 
         (antiAliasHeight- 1) / kBlockWidth + 1);
   initCurand<<<dimGrid, dimBlock>>>(dRandStates, width * sqrSampleCount, height * sqrSampleCount);

   generateCameraRays<<<dimGrid, dimBlock>>>(width * sqrSampleCount, height * sqrSampleCount, camera, dRayQueue, dRandStates);

   rayTrace<<<dimGrid, dimBlock>>>(width * sqrSampleCount, height * sqrSampleCount, dRayQueue, 
         dBvhTree, dLightList, lightCount, dAntiAliasBuffer, dShader, dRandStates);
   cudaDeviceSynchronize();
   checkCUDAError("RayTrace kernel failed");

   dimGrid = dim3((width - 1) / kBlockWidth + 1, (height - 1) / kBlockWidth + 1);
   averageBufferColors<<<dimGrid, dimBlock>>>(width, height, sqrSampleCount, 
         dOutput, dAntiAliasBuffer);
   cudaDeviceSynchronize();
   checkCUDAError("averageBufferColors kernel failed");

   HANDLE_ERROR(cudaMemcpy(output, dOutput, 
            sizeof(uchar4) * width * height, cudaMemcpyDeviceToHost));
}

#ifndef UTIL_H
#define UTIL_H

#define EPSILON 0.001f
const float EQUAL_EPSILON = 0.01f;

__device__ inline int isFloatZero(float n) {
   return n < EPSILON && n > -EPSILON;
}

__device__ inline int isFloatEqual(float a, float b) {
   return a + EQUAL_EPSILON > b && b + EQUAL_EPSILON > a;
}

__device__ inline int isFloatAboveZero(float n) {
   return n - EPSILON > 0.0f;
}

__device__ inline int isFloatLessThan(float l, float r) {
   return l + EPSILON < r;
}

__device__ inline float clamp(float x, float lo, float hi) {
   return x > hi ? hi : x < lo ? lo : x;
}

#endif //UTIL_H


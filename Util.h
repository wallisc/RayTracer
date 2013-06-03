#ifndef UTIL_H
#define UTIL_H

#define EPSILON 0.0001f
#define BIG_EPSILON 0.001f
const float EQUAL_EPSILON = 0.01f;

__device__ inline int isFloatZero(float n) {
   return n < EPSILON && n > -EPSILON;
}

__device__ inline int isFloatEqual(float a, float b) {
   return a + EQUAL_EPSILON > b && b + EQUAL_EPSILON > a;
}

__device__ inline int isFloatAboveZero(float n) {
   return n + EPSILON > 0.0f;
}

__device__ inline int isFloatZeroOrAbove(float n) {
   return n - EPSILON > 0.0f;
}

__device__ inline int isFloatLessThan(float l, float r) {
   return l + EPSILON < r;
}

__device__ inline float clamp(float x, float lo, float hi) {
   return x > hi ? hi : x < lo ? lo : x;
}

__device__ inline float isInRange(float x, float lo, float hi) {
   return x >= lo && x <= hi;
}

__device__ inline float mymin(float x, float y) {
   return x < y ? x : y;
}

__device__ inline float mymax(float x, float y) {
   return x > y ? x : y;
}

template <class T>
__host__ __device__ inline void SWAP(T &a, T &b) {
   T temp = a; a = b; b = temp;
}


#endif //UTIL_H

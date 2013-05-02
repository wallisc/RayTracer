#ifndef UTIL_H
#define UTIL_H

#define EPSILON 0.001f

__device__ inline int isFloatZero(float n) {
      return n < EPSILON && n > -EPSILON;
}

__device__ inline int isFloatLessThan(float l, float r) {
      return l + EPSILON < r;
}

__device__ inline float clamp(float x, float lo, float hi) {
      return x > hi ? hi : x < lo ? lo : x;
}

#endif //UTIL_H


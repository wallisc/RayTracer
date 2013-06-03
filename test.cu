#include <cuda.h>
#include <stdio.h>
#include "cudaError.h" 

__device__ float myrand(int seed) {
   unsigned int m_w = 150;
   unsigned int m_z = (unsigned int)seed;

   m_z = 36969 * (m_z & 65535) + (m_z >> 16);
   m_w = 18000 * (m_w & 65535) + (m_w >> 16);

   return (float)((m_z << 16) + m_w) / (float)UINT_MAX;
}

__global__ void test() {
   printf("%f\n", myrand(threadIdx.x + blockIdx.x * blockDim.x));
}

int main() {
   test<<<5, 16>>>();
   cudaDeviceSynchronize();
   checkCUDAError("test failed");
}

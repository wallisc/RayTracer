#include <cuda.h>
#include <stdio.h>
#include "cudaError.h" 

__global__ void test() {
   printf("Hi!\n");
}

int main() {
   cudaStream_t stream1, stream2;
   cudaStreamCreate(&stream1);
   cudaStreamCreate(&stream2);
   test<<<1, 1, 0, stream1>>>();
   test<<<1, 1, 0, stream2>>>();
   cudaDeviceSynchronize();
   checkCUDAError("test failed");
}

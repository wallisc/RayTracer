#include <cuda.h>
#include "stdio.h""

__global__ void test2(){
   printf("moo!\n");
}

__global__ void test(){
   printf("Cows\n");
   test2<<<2, 2>>>();
}

int main() {
   test<<<2, 2>>>();
}

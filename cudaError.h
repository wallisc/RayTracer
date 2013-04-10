#ifndef CUDA_ERROR_H
#define CUDA_ERROR_H

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

inline static void HandleError( cudaError_t err,
    const char *file, int line ) 
{
   if (err != cudaSuccess) {
      printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
         file, line );
      exit( EXIT_FAILURE );
   }
}

inline void checkCUDAError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) {
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) ); 
    exit(EXIT_FAILURE); 
  }
} 

#endif //CUDA_ERROR_H

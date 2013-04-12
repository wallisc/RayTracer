CC=nvcc
LD=nvcc
CUDAFLAGS= -O2 -lineinfo -arch=sm_21 -Xptxas -dlcm=ca 

all: raytrace.cpp POVRayParser.o kernel.o
	$(CC) $(CUDAFLAGS) raytrace.cpp Image.cpp POVRayParser.o kernel.o -o raytrace 

POVRayParser.o: POVRayParser.cpp POVRayParser.h
	$(CC) $(CUDAFLAGS) -g -c POVRayParser.cpp

kernel.o: kernel.cu
	$(CC) $(CUDAFLAGS) -w -c kernel.cu  

polymorphism: polymorphism.cu
	$(CC) $(CUDAFLAGS) polymorphism.cu

clean:
	rm -rf core* *.o *.gch junk* raytrace


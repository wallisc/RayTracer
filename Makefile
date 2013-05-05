CC=nvcc
LD=nvcc
CUDAFLAGS= -O3 -arch=sm_21 -Xptxas -dlcm=ca 
ALL= cudaError.h kernel.h Plane.h Shader.h Geometry.h Light.h PointLight.h Ray.h Sphere.h TokenData.h Material.h Util.h PhongShader.h CookTorranceShader.h Triangle.h SmoothTriangle.h

all: raytrace.cpp POVRayParser.o kernel.o
	$(CC) $(CUDAFLAGS) raytrace.cpp Image.cpp POVRayParser.o kernel.o -o raytrace

POVRayParser.o: POVRayParser.cpp POVRayParser.h
	$(CC) $(CUDAFLAGS) -c POVRayParser.cpp

kernel.o: kernel.cu $(ALL)
	$(CC) $(CUDAFLAGS) -w -c kernel.cu  

clean:
	rm -rf core* *.o *.gch junk* raytrace gmon.out


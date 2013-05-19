CC=nvcc
LD=nvcc
CUDAFLAGS= -O2 -lineinfo -arch=sm_20 -Xptxas -dlcm=ca
#-prec-div=false -prec-sqrt=false -use_fast_math
ALL= cudaError.h kernel.h Plane.h Shader.h Geometry.h Light.h PointLight.h Ray.h 
ALL+= Sphere.h TokenData.h Material.h Util.h PhongShader.h CookTorranceShader.h 
ALL+= Triangle.h SmoothTriangle.h Box.h bvh.h GeometryUtil.h BoundingBox.h

all: raytrace.cpp POVRayParser.o kernel.o
	$(CC) $(CUDAFLAGS) raytrace.cpp Image.cpp POVRayParser.o kernel.o -o raytrace

POVRayParser.o: POVRayParser.cpp POVRayParser.h
	$(CC) $(CUDAFLAGS) -c POVRayParser.cpp

kernel.o: kernel.cu $(ALL)
	$(CC) $(CUDAFLAGS) -c  kernel.cu  

clean:
	rm -rf core* *.o *.gch junk* raytrace gmon.out


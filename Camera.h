#ifndef CAMERA_H 
#define CAMERA_H 
#include "glm/glm.hpp"

typedef struct Camera {
   glm::vec3 pos, up, right, lookAtDir;
   Camera(glm::vec3 position, glm::vec3 upVec, glm::vec3 rightVec,
          glm::vec3 lookAtDirection) : pos(position), up(upVec),
          right(rightVec), lookAtDir(lookAtDirection) {}

} Camera;

#endif //CAMERA_H 

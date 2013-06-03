#ifndef POV_RAY_PARSER_H
#define POV_RAY_PARSER_H

#include <iostream>
#include <fstream>
#include "TokenData.h"

class POVRayParser {
public:
   static int parseFile(const std::string &fileName, 
         TKSceneData *data);

   enum {kSuccess = 0, kFileNotFound, kBadFormat, kUnsupportedObject};

private:
   enum {kBufferSize = 200};

   static int parseLightSource(std::ifstream &in, TKSceneData *data);
   static int parseAreaLight(std::ifstream &in, TKSceneData *data);
   static int parseCamera(std::ifstream &in, TKSceneData *data);

   static int parseBox(std::ifstream &in, TKSceneData *data);
   static int parseSphere(std::ifstream &in, TKSceneData *data);
   static int parseCone(std::ifstream &in, TKSceneData *data);
   static int parsePlane(std::ifstream &in, TKSceneData *data);
   static int parseTriangle(std::ifstream &in, TKSceneData *data);
   static int parseSmoothTriangle(std::ifstream &in, TKSceneData *data);

   static int parseModifiers(std::ifstream &in, TKModifier *m, TKSceneData *data);

   static int parsePigment(std::ifstream &in, TKPigment *pigment, TKSceneData *data);
   static int parseFinish(std::ifstream &in, TKFinish *finish);

   static int parseScale(std::ifstream &in, glm::mat4 *matStack);
   static int parseRotate(std::ifstream &in, glm::mat4 *matStack);
   static int parseTranslate(std::ifstream &in, glm::mat4 *matStack);

   static int parseVector(std::ifstream &in, glm::vec3 *v);
   static int parseUV(std::ifstream &in, glm::vec2 *v);
   static int parseWord(std::ifstream &in, std::string s);
   static int parseCharacter(std::ifstream &in, char c);

   static bool isUV(std::ifstream &in);
   static bool isComment(std::ifstream &in);
   static void checkComments(std::ifstream &in);
};

#endif //POV_RAY_PARSER_H

#include "Box.h"

class BVHNode {
public:
   BVHNode *left, *right;
   BVHNode() : left(NULL), right(NULL) {}
private:
};

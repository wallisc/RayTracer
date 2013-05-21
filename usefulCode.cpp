
typedef struct SortFrame {
   int size;
   Geometry **arr;
   int topOfBottom;
   __device__ SortFrame(int nTopOfBottom = 0, int nSize = 0, Geometry **nArr = NULL) 
      : size(nSize), topOfBottom(nTopOfBottom), arr(nArr) {}
} SortFrame;

__device__ int pickPivot(Geometry *list[], int size, int axis) {
   int first = 0, middle = size / 2, last = size -1;
   float firstVal = list[first]->getCenter()[axis];
   float midVal = list[middle]->getCenter()[axis]; 
   float lastVal = list[last]->getCenter()[axis];

   if (firstVal < lastVal) {
      if (midVal < lastVal) {
         return midVal > firstVal ? middle : first;
      } else {
         return last;
      }
   } else { // if (firstVal > lastVal)
      if (midVal < lastVal) {
         return firstVal < midVal ? middle : first;
      } else {
         return last;
      }
   }
}

__global__ void kernelSort(Geometry *list[], int start, int end, int axis) {
   SortFrame stack[kMaxStackSize];
   int stackSize = 0;
   bool stackPopped = false;

   int size = end - start;
   int topOfBottom;
   Geometry **arr = list + start;
   while (1) {
      if (stackSize == kMaxStackSize) {
         printf("Stack size exceeded, aborting\n");
         return;
      }
      // If small enough size, do insertion sort
      if (size < kInsertionSortCutoff) {
         for (int i = 1; i < size; i++) {
            int j = i;
            Geometry *toInsert = arr[j];
            for (; j > 0 && toInsert->getCenter()[axis] < arr[j - 1]->getCenter()[axis]; j--) {
               arr[j] = arr[j-1];
            }
            arr[j] = toInsert;
         }
      } else {
         if (!stackPopped) {
            int pivot = pickPivot(arr, size, axis);
            SWAP(arr[pivot], arr[size - 1]);
            topOfBottom = 0;
            for (int i = 0; i < size - 1; i++) {
               if(arr[i]->getCenter()[axis] < arr[size - 1]->getCenter()[axis]) {
                  SWAP(arr[i], arr[topOfBottom++]);   
               }             
            }
            SWAP(arr[topOfBottom++], arr[size - 1]);   
            stack[stackSize++] = SortFrame(topOfBottom, size, arr); 
            size = topOfBottom;
            stackPopped = false;
            continue;
         } else {
            arr += topOfBottom;
            size -= topOfBottom;
            stackPopped = false;
            continue;
         }
      }

      if (stackSize == 0) break;
      arr = stack[stackSize - 1].arr;
      size = stack[stackSize - 1].size;
      topOfBottom = stack[stackSize - 1].topOfBottom;
      stackSize--;
      stackPopped = true;
   }
}

void singleThreadSort(Geometry *geomList[], int start, int end, int axis, cudaStream_t stream) {
   kernelSort<<<1, 1, 0, stream>>>(geomList, start, end, axis);
}

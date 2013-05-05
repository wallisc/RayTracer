import sys, re 

DEFAULT_CAMERA = """
camera {
    location <-1, 1, 3>
    up <0, 1, 0>
    right <1.33, 0, 0>
    look_at <0, 0, -1>
}
"""

DEFAULT_LIGHTS = """
light_source { <30, 10, 30> color rgb <1.0, 1.0, 1.0> }
"""

DEFAULT_MATERIAL = """
   pigment { color rgb <0.8, 0.8, 0.2> } 
   finish { ambient 0.2 diffuse 0.6 specular 0.3 roughness 0.05 }
"""

def vecToStr(vec):
   return "<" + ", ".join(["%.6f" % i for i in vec]) + ">"

def convertToPOV(fileName):
   objCheck = re.compile(r".*\.obj")
            
   if objCheck.match(fileName) == None:
      print fileName + " is not a valid .obj file\n"
      return
      
   file = open(fileName, 'r')
   outputName = fileName[:len(fileName) - 4] + ".pov"
   outFile = open(outputName, 'w')

   outFile.write(DEFAULT_CAMERA)
   outFile.write(DEFAULT_LIGHTS)

   normList = []
   vertList = []
   normalCheck = re.compile(r"vn\s.*")
   vertexCheck = re.compile(r"v\s.*")
   faceCheck = re.compile(r"f\s.*")
   
   line = file.readline()
   while line != '':
      if normalCheck.match(line):
         normals = re.split("\s", line)
         # Skip the vt tag (i.e. start at index 1)
         n1 = float(normals[1])
         n2 = float(normals[2])
         n3 = float(normals[3])
         normList.append((n1, n2, n3))

      elif vertexCheck.match(line):
         vertex = re.split("\s", line)
         # Skip the v tag (i.e. start at index 1)
         v1 = float(vertex[1])
         v2 = float(vertex[2])
         v3 = float(vertex[3])
         vertList.append((v1, v2, v3))
      
      elif faceCheck.match(line):
         break

      line = file.readline()

         
   # If the list of normals and vertices have been fully constructed
   faceCount = 0
   while line != '': 
      if faceCheck.match(line):
         faceCount += 1
         outFile.write("smooth_triangle {\n")
         face = re.split("\s", line)
         for idx in range(1, 4):
            point = re.split("/", face[idx])
            # Note: OBJ files are 1-indexed
            vertIdx = int(point[0]) - 1 
            normIdx = int(point[len(point) - 1]) - 1
            outFile.write("   " + vecToStr(vertList[vertIdx]) + ", " 
                                + vecToStr(normList[normIdx]))
            
            if idx < 3: outFile.write(",\n")
            else: outFile.write("\n")

         outFile.write(DEFAULT_MATERIAL)
         outFile.write("}\n\n")

      line = file.readline()

   print "Succesfully wrote out " + outputName
   print "Number of triangles: " + repr(faceCount)

def main():
   args = sys.argv
   for i in range(1, len(args)):
      convertToPOV(args[i])

main()

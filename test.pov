// cs473, assignment 1 recursion test (RIGHT HANDED)
camera {
   location  <0, 0, 14>
      up        <0,  1,  0>
      right     <1.5, 0,  0>
      look_at   <0, 0, 0>
}


light_source {<-5, 3, 0> color rgb <0.6, 1.0, 0.6>}
light_source {<5, 10, 5> color rgb <0.6, 0.8, 1.0>}

// floor
plane {<0, 1, 0>, -5
   pigment {color rgb <0.2, 0.2, 0.8>}
   finish {ambient 0.4 diffuse 0.2 reflection 0.2}
   translate <0, -1, 0>
}

// left wall
plane {<1, 0, 0>, -8
   pigment {color rgb <0.8, 0.2, 0.2>}
   finish {ambient 0.4 diffuse 0.2 reflection 0.2}
   rotate <0, 30, 0>
}

// back wall
plane {<0, 0, -1>, 30
   pigment {color rgb <0.8, 0.4, 0.2>}
   finish {ambient 0.4 diffuse 0.2 reflection 0.2}
   rotate <0, 30, 0>
}

// back wall on the right
plane {<0, 0, -1>, 30
   pigment {color rgb <0.0, 0.2, 0.2>}
   finish {ambient 0.4 diffuse 0.8 reflection 0.2}
   rotate <0, -20, 0>
}

box { <-2, -5, -5>, <2, 5, 5>
   pigment { color rgb <1.0, 0.2, 1.0>}
   finish {ambient 0.2 diffuse 0.8}
   rotate <0, -45, 0>
   translate <3, 0, -5>
}

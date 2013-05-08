camera {
    location <1, 6, 12>
    up <0, 1, 0>
    right <1.33, 0, 0>
    look_at <0, 0, 0>
}

light_source { <30, 10, 30> color rgb <0.3, 0.3, 0.3> }
light_source { <-30, 10, 30> color rgb <0.3, 0.3, 0.3> }
light_source { <0, 10, 30> color rgb <0.3, 0.3, 0.3> }

// left wall
plane {<1, 0, 0>, -8
   pigment {color rgb <0.9, 0.2, 0.2>}
   finish {ambient 0.4 diffuse 0.2 reflection 0.2}
   rotate <0, 30, 0>
}

// back wall
plane {<0, 0, -1>, 30
   pigment {color rgb <0.2, 0.2, 0.9>}
   finish {ambient 0.4 diffuse 0.2 reflection 0.2}
   rotate <0, 30, 0>
}

// back wall on the right
plane {<0, 0, -1>, 30
   pigment {color rgb <0.0, 0.9, 0.2>}
   finish {ambient 0.4 diffuse 0.8 reflection 0.2}
   rotate <0, -20, 0>
}

plane { <0, 1, 0>, 0 
    pigment { color rgb <1.0, 1.0, 1.0> }
    finish { ambient 0.2 diffuse 0.6 specular 0.2 reflection .3 roughness 0.05 }
}

sphere { <0, 0, 0>, 2
   pigment { color rgb <1.0, 0.0, 0.0>}
   finish {ambient 0.2 diffuse 0.4 specular 0.5 reflection .3 roughness 0.5}
   translate <-7, 3, 0>
}

sphere { <0, 0, 0>, 2
   pigment { color rgb <0.0, 1.0, 0.0>}
   finish {ambient 0.2 diffuse 0.4 specular 0.5  reflection .3 roughness 0.3}
   translate <0, 3, -7>
}

sphere { <0, 0, 0>, 2
   pigment { color rgb <0.0, 0.0, 1.0>}
   finish {ambient 0.2 diffuse 0.4 specular 0.5  reflection .3 roughness 0.01}
   translate <7, 3, 0>
}

camera {
   location  <0, 0, 14>
   up        <0,  1,  0>
   right     <1.33333, 0,  0>
   look_at   <0, 0, 0>
}


light_source {<0, 3, 10> color rgb <1.5, 1.5, 1.5>}

sphere { <0, 0, 0>, 2
   pigment { color rgb <1.0, 0.0, 0.0>}
   finish {ambient 0.2 diffuse 0.4 specular 0.5 roughness 0.5}
   translate <-4.5, 0, 0>
}

sphere { <0, 0, 0>, 2
   pigment { color rgb <0.0, 1.0, 0.0>}
   finish {ambient 0.2 diffuse 0.4 specular 0.5 roughness 0.3}
   translate <0, 0, 0>
}

sphere { <0, 0, 0>, 2
   pigment { color rgb <0.0, 0.0, 1.0>}
   finish {ambient 0.2 diffuse 0.4 specular 0.5 roughness 0.01}
   translate <4.5, 0, 0>
}


plane {<0, 1, 0>, -4
   pigment {color rgb <0.2, 0.2, 0.8>}
   finish {ambient 0.4 diffuse 0.8 reflection .5}
}

plane {<0, -1, 0>, -4
   pigment {color rgb <0.8, 0.2, 0.2>}
   finish {ambient 0.4 diffuse 0.8 reflection .5}
}

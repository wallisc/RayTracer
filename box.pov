camera {
   location  <-7, 4, -14>
   up        <0,  1,  0>
   right     <1.33333, 0,  0>
   look_at   <0, 0, 0>
}


area_light {<0, 4, -10> <0, 9, -10> <0, 4, -8> sample 5 color rgb <1.5, 1.5, 1.5>}

box { <-1, -1, -1>, <1, 2, 1>
   pigment { color rgb <1.0, 0.0, 1.0>}
   finish {ambient 0.2 diffuse 0.6 specular 0.5 roughness 0.01}
}

box { <-4, -1, -1>, <-2, 1, 1>
   pigment { color rgb <1.0, 1.0, 0.0>}
   finish {ambient 0.2 diffuse 0.6 specular 0.5 roughness 0.01}
}

box { <2, -1, -1>, <4, 3, 1>
   pigment { color rgb <0.0, 1.0, 1.0>}
   finish {ambient 0.2 diffuse 0.6 specular 0.5 roughness 0.01}
}

sphere { <0, 4, 0>, 2
   pigment { color rgb <1.0, 1.0, 1.0>}
   finish {ambient 0.2 diffuse 0.4 specular 0.5 roughness 0.01}
}

plane {<0, 1, 0>, -1
   pigment {color rgb <0.2, 0.2, 0.8>}
   finish {ambient 0.4 diffuse 0.8 reflection .5}
}

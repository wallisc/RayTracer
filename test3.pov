camera {
   location  <0, 0, 14>
   up        <0,  1,  0>
   right     <1.33333, 0,  0>
   look_at   <0, 0, 0>
}


light_source {<0, 3, 10> color rgb <1.5, 1.5, 1.5>}

sphere { <0, 0, 0>, 2
   pigment { color rgb <1.0, 0.0, 0.0>}
   finish {ambient 0.2 diffuse 0.4 specular 0.5 reflection .9 roughness 0.5}
   translate <-4.5, 0, 0>
}

sphere { <0, 0, 0>, 2
   pigment { color rgb <0.0, 1.0, 0.0>}
   finish {ambient 0.2 diffuse 0.4 specular 0.5  reflection .6 roughness 0.3}
   translate <0, 0, 0>
}

sphere { <0, 0, 0>, 2
   pigment { color rgb <0.0, 0.0, 1.0>}
   finish {ambient 0.2 diffuse 0.4 specular 0.5  reflection .3 roughness 0.01}
   translate <4.5, 0, 0>
}


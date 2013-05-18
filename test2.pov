// cs174, assignment 1 sample file (RIGHT HANDED)

camera {
   location  <0, 0, 7>
      up        <0,  1,  0>
      right     <1.33333, 0,  0>
      look_at   <0, 0, 0>
}

light_source {<-100, 100, 100> color rgb <1.5, 1.5, 1.5>}


triangle {
   <1,1 ,1 >,
      <-.001, -.3,4 >,
      <1,-.3 ,1 >
         pigment {color rgb <1.0, 1.0, 1.0>}
   finish {ambient 0.3 diffuse 0.4}}


triangle {
   <-.001,1 ,0 >,
      <-.001, -.3,3 >,
      <1,-.3 ,0 >
         pigment {color rgb <0.65, 0.4, 0.4>}
   finish {ambient 0.3 diffuse 0.4}}

triangle {
   <0,1 ,0 >,
      <-1, -.3,0 >,
      <0,-.3 ,3 >
         pigment {color rgb <0.65, 0.4, 0.4>}
   finish {ambient 0.3 diffuse 0.4}
}

triangle {
   <-1, -.3,0 >,
      <0,-1.3 ,0 >,
      <0,-.3 ,3 >
         pigment {color rgb <0.65, 0.4, 0.4>}
   finish {ambient 0.3 diffuse 0.4}
}

triangle {
   <0,-.3 ,2 >,
      <0,-1.3 ,0 >,
      <1, -.3,0 >
         pigment {color rgb <0.65, 0.4, 0.4>}
   finish {ambient 0.3 diffuse 0.4}
}



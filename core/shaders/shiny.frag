#version 150

uniform vec3 u_eye;
uniform vec3 u_light;
uniform float u_alpha;

in vec3 position;
in vec3 normal;
in vec3 color;

out vec4 outputF;

float diffuse( vec3 N, vec3 L )
{
   return max( 0., dot( N, L ));
}

float specular( vec3 N, vec3 L, vec3 E )
{
   const float shininess = 4.;
   vec3 R = 2.*dot(L,N)*N - L;
   return pow( max( 0., dot( R, E )), shininess );
}

float fresnel( vec3 N, vec3 E )
{
   const float sharpness = 10.;
   float NE = max( 0., dot( N, E ));
   return pow( sqrt( 1. - NE*NE ), sharpness );
}

void main()
{

   vec3 N = normalize( normal );
   vec3 L = normalize( u_light - position );
   vec3 E = normalize( u_eye - position );
   vec3 R = 2.*dot(L,N)*N - L;
   vec3 one = vec3( 1., 1., 1. );

   // Show normals:
   //outputF.rgb = 0.5 * (N + one);

   outputF.rgb = 0.1*color + 0.8*diffuse(N,L)*color + 0.1*specular(N,L,E)*one + 0.2*fresnel(N,E)*color;
   //outputF.rgb = diffuse(N,L)*color;
   outputF.a = u_alpha;
}

#version 150

uniform vec3 u_eye;
uniform vec3 u_light;

in vec3 position;
in vec3 color;

out vec4 outputF;


void main()
{
   outputF.rgb = color;
   outputF.a = 1.0;
}

#version 150

uniform mat4 u_viewMatrix;
uniform mat4 u_projMatrix;

in vec3 a_position;
in vec3 a_normal;
//in vec3 color;

out vec4 color;

void main()
{
    color = vec4(0.2,0.6,1.0,1.0);
    //color = vec4(a_normal, 1.0);
    gl_Position = u_projMatrix * u_viewMatrix * vec4(a_position, 1.0) ;
}

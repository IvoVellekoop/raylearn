from OpenGL import GL
from OpenGL import GLUT
from OpenGL.GL import shaders
from OpenGL.GL import arrays
import numpy as np

def calculate_elements(nx, ny):
        elements = np.zeros((nx-1)*(ny-1)*6,np.uint32)
        index = 0
        for r in range(0, ny-1):
            for c in range(0,nx-1):
                base = c + r*(nx)
                elements[index] = base + 0
                elements[index + 1] = base + 1
                elements[index + 2] = base + nx

                elements[index + 3] = base + 1
                elements[index + 4] = base + nx
                elements[index + 5] = base + nx + 1
                index += 6
        return elements

class shadertest:

    def load_shaders(self): 
        vertexsource = """
        #version 440 core

        layout(location = 0) in vec2 vertexPosition_modelspace;
        layout(location = 1) in float vertexColor;
        out float fragmentColor;

        void main() { 
            fragmentColor = vertexColor;
            gl_Position = vec4(vertexPosition_modelspace,1.0,1.0); 
        }
        """

        fragmentsource = """
        #version 440 core
        #define PI 3.141593

        in float fragmentColor;
        out vec3 color;
        void main() { 
            color = vec3(mod(fragmentColor*1000000,1),0,0);
        }
        """

        vs = GL.shaders.compileShader(vertexsource, GL.GL_VERTEX_SHADER)
        fs = GL.shaders.compileShader(fragmentsource, GL.GL_FRAGMENT_SHADER)
        self.program = GL.shaders.compileProgram(vs,fs)

    def load_data(self, data):
        self.nx = data.shape[1]
        self.ny = data.shape[0]
        self.vao = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.vao)
        data[:,:,0:2] /= 100e-6
        self.vbo = GL.arrays.vbo.VBO(data)
        elements = calculate_elements(self.nx,self.ny)
        self.ibo = GL.arrays.vbo.VBO(elements, target=GL.GL_ELEMENT_ARRAY_BUFFER)
        
    def draw_frame(self):
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        self.vbo.bind()
        self.ibo.bind()
        GL.glUseProgram(self.program)
        GL.glEnableVertexAttribArray(0)
        GL.glEnableVertexAttribArray(1)
        
        GL.glVertexAttribPointer(0, 2, GL.GL_DOUBLE, GL.GL_FALSE, 24, self.vbo)
        GL.glVertexAttribPointer(1, 1, GL.GL_DOUBLE, GL.GL_FALSE, 24, self.vbo+16)
        
        GL.glDrawElements(GL.GL_TRIANGLES, (self.nx - 1)*(self.ny - 1)*6, GL.GL_UNSIGNED_INT, None)
        GLUT.glutSwapBuffers()

    def __init__(self, data):
        GLUT.glutInit()
        GLUT.glutInitContextVersion(4,4)
        GLUT.glutInitContextProfile(GLUT.GLUT_CORE_PROFILE)
        GLUT.glutInitWindowPosition(100,100)
        GLUT.glutInitWindowSize(600,600)
        window = GLUT.glutCreateWindow("A window!")
        GLUT.glutDisplayFunc(self.draw_frame)
        self.load_shaders()
        self.load_data(data)
        GL.glClearColor(0, 0.5, 0, 1)
        GL.glPointSize(5)
        GLUT.glutMainLoop()

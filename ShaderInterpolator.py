from OpenGL import GL
from OpenGL import GLUT
from OpenGL.GL import shaders
from OpenGL.GL import arrays
import glfw
import numpy as np

def interpolate_shader(data):
    """
    Interpolate data using OpenGL shader. Data should be a numpy array. 
    """
    s = ShaderInterpolator(data)
    return s.get_field()

def calculate_elements(nx, ny):
    """
    Helper function for lazy triangulation of a grid.
    Returns indices of triangle corners
    """
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

class ShaderInterpolator:
    """
    Interpolate path length on the GPU through a shader.
    """
    
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
        uniform float lambda;
        out vec4 color;
        void main() { 
            float phi = fragmentColor * 2*PI/lambda;
            color = vec4(cos(phi),sin(phi),0,0);
        }
        """

        vs = GL.shaders.compileShader(vertexsource, GL.GL_VERTEX_SHADER)
        fs = GL.shaders.compileShader(fragmentsource, GL.GL_FRAGMENT_SHADER)
        self.program = GL.shaders.compileProgram(vs,fs)

    def load_data(self, data):
        """
        Load data into GPU buffers. `data` is Ny*Nx*3, paired as (y,x,pathlength)
        Two buffers are created, `vbo` with the raw data, 
        and `ibo` with the elements (indices) for the triangulation
        """
        self.nx = data.shape[1]
        self.ny = data.shape[0]
        
        # Mandatory for OpenGL to work
        self.vao = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.vao)

        data[:,:,0:2] /= 100e-6 # TODO make display range adjustable
        meanpath = np.mean(data[:,:,2])
        data[:,:,2] -= meanpath
        self.vbo = GL.arrays.vbo.VBO(data)
        
        elements = calculate_elements(self.nx,self.ny)
        self.ibo = GL.arrays.vbo.VBO(elements, target=GL.GL_ELEMENT_ARRAY_BUFFER)

    def get_field(self):
        return self.field_out
        
    def draw_frame(self):
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        self.vbo.bind()
        self.ibo.bind()
        GL.glUseProgram(self.program)
        GL.glEnableVertexAttribArray(0) # input 0: vertexPosition_modelspace 
        GL.glEnableVertexAttribArray(1) # input 1: vertexColor
        
        # Position. Two doubles, stride 3*8 bytes
        GL.glVertexAttribPointer(0, 2, GL.GL_DOUBLE, GL.GL_FALSE, 24, self.vbo)
        # Pathlength/'color'. One double, stride 3*8 bytes, offset 2*8 bytes
        GL.glVertexAttribPointer(1, 1, GL.GL_DOUBLE, GL.GL_FALSE, 24, self.vbo+16)
        # Wavelength
        lambda_loc = GL.glGetUniformLocation(self.program, "lambda")
        GL.glUniform1f(lambda_loc, 1e-6) # TODO Make wavelength configurable
        
        GL.glDrawElements(GL.GL_TRIANGLES, (self.nx - 1)*(self.ny - 1)*6, GL.GL_UNSIGNED_INT, None)

    def __init__(self, data):
        # Create OpenGL context
        glfw.init()
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 4)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        self.window = glfw.create_window(600, 600, "A Window", None, None)
        glfw.make_context_current(self.window)
        
        # Prepare GPU resources
        self.load_shaders()
        self.load_data(data)

        # Enable blending for addition of fields
        GL.glClearColor(0, 0, 0, 0)
        GL.glEnable(GL.GL_BLEND) 
        GL.glBlendFunc(GL.GL_ONE, GL.GL_ONE)
        # Default blendEquation is already GL_ADD
        #GL.glBlendEquation(GL.GL_ADD)

        # Create output framebuffers
        self.rb = GL.glGenRenderbuffers(1)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, self.rb)
        GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_RGBA32F, 600, 600) # 600^2 pixels, 4-channel 32-bit float
        self.fb = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fb) # read/write framebuffer
        GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_RENDERBUFFER, self.rb)

        # Draw to framebuffer
        self.draw_frame()
        image_buffer = GL.glReadPixels(0, 0, 600, 600, GL.GL_RGBA, GL.GL_FLOAT)
        self.field_out = image_buffer[:,:,0] + 1j*image_buffer[:,:,1]

        # Main loop, draw to window
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER,0)
        #GL.glViewport(0, 0, 0, 0)
        while not glfw.window_should_close(self.window):
            self.draw_frame()
            glfw.swap_buffers(self.window)
            glfw.poll_events()
        
        # Clean up resources
        self.vbo.delete()
        self.ibo.delete()
        GL.glDeleteProgram(self.program)
        # GL.glDeleteRenderbuffers(1, self.rb) # This crashes for some reason
        # GL.glDeleteFramebuffers(1, self.fb)
        glfw.terminate()

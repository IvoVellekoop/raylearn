from OpenGL import GL
from OpenGL import GLUT
from OpenGL.GL import shaders
from OpenGL.GL import arrays
import glfw
import numpy as np

"""
Functions to interpolate raytracer output rays/pathlength to complex field. 

Only importing interpolate_shader is necessary, the rest are helper functions
and encapsulation of OpenGL state.

This code uses OpenGL functions for hardware-accelerated interpolation, make 
sure that pyOpenGL and pyGlfw are installed and your video drivers are set up
correctly. You may need to manually override the anaconda-provided libstdc++ to
match your system version.
"""

def interpolate_shader(data, npoints=(600,600), limits=(-100e-6,100e-6, -100e-6,100e-6), wavelength_m = 1e-6):
    """
    Interpolate data using OpenGL shader. Data should be a numpy array. 

    parameters:
    data            Nx-by-Ny-by-3 numpy array of rays to be interpolated,
                    in [x,y,pathlength] format. 
    npoints         (x,y)-resolution of calculated field. Tuple, defaults 
                    to (600,600) pixels
    wavelength_m    Wavelength of the light, used to calculate phase 
    limits          The area of the output plane where field interpolation 
                    should be performed. Tuple, (x_min, x_max, y_min, y_max)

    """
    s = ShaderInterpolator(data, npoints, limits, wavelength_m)
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

def coordinate_matrix(resolution, limits):
    """
    Helper function to transpose the rays and interpolate
    at the correct coordinates.
    """

    range_x = limits[1] - limits[0]
    range_y = limits[3] - limits[2]
    pixelsize_x = range_x / resolution[0]
    pixelsize_y = range_y / resolution[1]

    scale_x = 2/range_x
    scale_y = 2/range_y
    offset_x = scale_x * 0.5 * (pixelsize_x - (limits[0]+limits[1]))
    offset_y = scale_y * 0.5 * (pixelsize_y - (limits[2]+limits[3]))

    M = [ scale_x,       0, 0, offset_x,
                0, scale_y, 0, offset_y,
                0,       0, 1,        0,
                0,       0, 0,        1 ]
    return M

class ShaderInterpolator:
    """
    Interpolate path length on the GPU through a shader.
    """
    
    def opengl_setup(self, resolution):
        """
        Set up OpenGL context and buffers 
        """
        # Create window
        glfw.init()
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 4)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        self.window = glfw.create_window(resolution[0], resolution[1], "A Window", None, None)
        glfw.make_context_current(self.window)

        # Enable blending for addition of fields
        GL.glClearColor(0, 0, 0, 0)
        GL.glEnable(GL.GL_BLEND) 
        GL.glBlendFunc(GL.GL_ONE, GL.GL_ONE)
        # Default blendEquation is already GL_ADD
        #GL.glBlendEquation(GL.GL_ADD)

        # Create output framebuffers
        self.rb = GL.glGenRenderbuffers(1)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, self.rb)
        GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_RGBA32F, resolution[0], resolution[1]) # x-by-y pixels, 4-channel 32-bit float
        self.fb = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fb) # read/write framebuffer
        GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_RENDERBUFFER, self.rb)

    def load_shaders(self): 
        vertexsource = """
        #version 440 core

        layout(location = 0) in vec2 vertexPosition_modelspace;
        layout(location = 1) in float vertexColor;
        uniform mat4 MVP;
        out float fragmentColor;

        void main() { 
            fragmentColor = vertexColor;
            gl_Position = MVP * vec4(vertexPosition_modelspace,1.0,1.0); 
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

        meanpath = np.mean(data[:,:,2])
        data[:,:,2] -= meanpath
        self.vbo = GL.arrays.vbo.VBO(data)
        
        elements = calculate_elements(self.nx,self.ny)
        self.ibo = GL.arrays.vbo.VBO(elements, target=GL.GL_ELEMENT_ARRAY_BUFFER)
        
    def draw_frame(self):
        """
        Draw a single frame using the interpolation shader. 
        """
        
        # Bind buffers
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        self.vbo.bind()
        self.ibo.bind()
        GL.glUseProgram(self.program)
        
        # Set up inputs
        # Position. Two doubles, stride 3*8 bytes
        GL.glEnableVertexAttribArray(0) # input 0: vertexPosition_modelspace 
        GL.glVertexAttribPointer(0, 2, GL.GL_DOUBLE, GL.GL_FALSE, 24, self.vbo)
        
        # Pathlength/'color'. One double, stride 3*8 bytes, offset 2*8 bytes
        GL.glEnableVertexAttribArray(1) # input 1: vertexColor
        GL.glVertexAttribPointer(1, 1, GL.GL_DOUBLE, GL.GL_FALSE, 24, self.vbo+16)
        
        # Wavelength
        lambda_loc = GL.glGetUniformLocation(self.program, "lambda")
        GL.glUniform1f(lambda_loc, self.wavelength_m)
        
        # Coordinate system
        mvp_loc = GL.glGetUniformLocation(self.program, "MVP")
        GL.glUniformMatrix4fv(mvp_loc, 1, GL.GL_TRUE, self.MVP) # GLTrue because matrix needs to be transposed
        
        # Actual draw call
        GL.glDrawElements(GL.GL_TRIANGLES, (self.nx - 1)*(self.ny - 1)*6, GL.GL_UNSIGNED_INT, None)

    def __init__(self, data, npoints, limits, wavelength_m):
        # Set parameters
        self.wavelength_m = wavelength_m
        
        # Calculate coordinate transformation
        self.MVP = coordinate_matrix(npoints, limits)

        # Create OpenGL context
        self.opengl_setup(npoints)
        
        # Prepare GPU resources
        self.load_shaders()
        self.load_data(data)

        # Draw to framebuffer
        self.draw_frame()
        image_buffer = GL.glReadPixels(0, 0, npoints[0], npoints[1], GL.GL_RGBA, GL.GL_FLOAT)
        # Convert array from column-major to row-major
        image_out = np.frombuffer(image_buffer, dtype=np.float32).reshape(npoints[1],npoints[0], 4)
        # Convert to complex field
        self.field_out = image_out[:,:,0] + 1j*image_out[:,:,1]
        
        # # Main loop, draw to window. Not necessary for calculation but may be useful for debugging
        # GL.glBindFramebuffer(GL.GL_FRAMEBUFFER,0)
        # #GL.glViewport(0, 0, 0, 0)
        # while not glfw.window_should_close(self.window):
        #     self.draw_frame()
        #     glfw.swap_buffers(self.window)
        #     glfw.poll_events()
        
        # Clean up resources
        self.vbo.delete()
        self.ibo.delete()
        GL.glDeleteProgram(self.program)
        # GL.glDeleteRenderbuffers(1, self.rb) # This crashes for some reason
        # GL.glDeleteFramebuffers(1, self.fb)
        glfw.terminate()

    def get_field(self):
        return self.field_out

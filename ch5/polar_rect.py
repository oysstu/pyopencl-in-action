'''
Listing 5.4: Rectilinear to polar coordinates
'''

import numpy as np
import pyopencl as cl
import pyopencl.array
import utility

kernel_src = '''
__kernel void polar_rect(__global float4 *r_vals,
                         __global float4 *angles,
                         __global float4 *x_coords,
                         __global float4 *y_coords) {

   *y_coords = sincos(*angles, x_coords);
   *x_coords *= *r_vals;
   *y_coords *= *r_vals;
}
'''

# Get device and context, create command queue and program
dev = utility.get_default_device()
context = cl.Context(devices=[dev], properties=None, dev_type=None, cache_dir=None)
queue = cl.CommandQueue(context, dev, properties=None)

# Build program in the specified context using the kernel source code
prog = cl.Program(context, kernel_src)
try:
    prog.build(options=['-Werror'], devices=[dev], cache_dir=None)
except:
    print('Build log:')
    print(prog.get_build_info(dev, cl.program_build_info.LOG))
    raise

# Data and buffers
r_in = cl.array.vec.make_float4(2, 1, 3, 4)
angles_in = cl.array.vec.make_float4(3*np.pi/8, 3*np.pi/4, 4*np.pi/3, 11*np.pi/6)
x_out = np.empty_like(r_in, dtype=cl.array.vec.float4)
y_out = np.empty_like(r_in, dtype=cl.array.vec.float4)

# Create output buffer
r_arg = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=r_in)
angles_arg = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=angles_in)
x_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, size=x_out.nbytes)
y_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, size=y_out.nbytes)

# Enqueue kernel (with argument specified directly)
global_size = (1,)
local_size = None

# __call__(queue, global_size, local_size, *args, global_offset=None, wait_for=None, g_times_l=False)
prog.polar_rect(queue, global_size, local_size, r_arg, angles_arg, x_buffer, y_buffer)

# Enqueue command to copy from buffer_out to host memory
cl.enqueue_copy(queue, dest=x_out, src=x_buffer, is_blocking=False)
cl.enqueue_copy(queue, dest=y_out, src=y_buffer, is_blocking=True)

print('X output: ' + str(x_out))
print('Y output: ' + str(y_out))


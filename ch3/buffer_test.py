'''
Listing 3.2: Reading rectangular buffer data
'''

import pyopencl as cl
import numpy as np
import utility

kernel_src = '''
__kernel void blank(__global float *a) {
}
'''

# Get device, context and command queue
dev = utility.get_default_device()
context = cl.Context(devices=[dev], properties=None, dev_type=None, cache_dir=None)
queue = cl.CommandQueue(context, dev, properties=None)

# Program signatures
# Program(context, src)
# Program(context, devices, binaries)
prog = cl.Program(context, kernel_src)

try:
    prog.build(options=['-Werror'], devices=[dev], cache_dir=None)
except:
    print('Build log:')
    print(prog.get_build_info(dev, cl.program_build_info.LOG))

full_matrix = np.arange(start=0, stop=80, step=1, dtype=np.float32)
zero_matrix = np.zeros(shape=(80,), dtype=np.float32)

#matrix_buffer = cl_array.to_device(queue, full_matrix)
flags = cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR
matrix_buffer = cl.Buffer(context, flags, hostbuf=full_matrix)

blank_kernel = prog.blank  # Note: Every call like this produces a new object
print('Function name: ' + blank_kernel.function_name)

n_globals = full_matrix.shape
n_locals = None
blank_kernel(queue, n_globals, n_locals, matrix_buffer, global_offset=None, wait_for=None, g_times_l=False)

# Itemsize yields the number of bytes in a float
buffer_origin = (0,)
host_origin = (2*full_matrix.itemsize,)
region = (2*full_matrix.itemsize, 3, 1)  # Copy 5 * 4 * sizeof(float32)

print('\nSource buffer:')
print(full_matrix)

print('\nBefore copy back:')
print(zero_matrix)

# Copy rectangular area of the buffer back to host
# Numpy exposes the buffer interface, and can therefore be specified directly in dest
# Note: A version later than pyopencl v2015.2.4 is necessary for this to work
cl.enqueue_copy(queue, dest=zero_matrix, src=matrix_buffer, is_blocking=True,
                buffer_origin=buffer_origin, host_origin=host_origin, region=region)

print('\nAfter copy back')
print(zero_matrix)


'''
Listing 3.3: Copying and mapping buffer objects
'''

import pyopencl as cl
import numpy as np
import utility

kernel_src = '''
__kernel void blank(__global float *a, __global float *b) {
}
'''

# Get device and context, create command queue and program
dev = utility.get_default_device()
context = cl.Context(devices=[dev])
queue = cl.CommandQueue(context, dev)
prog = cl.Program(context, kernel_src)

try:
    prog.build(options=['-Werror'], devices=[dev])
except:
    print('Build log:')
    print(prog.get_build_info(dev, cl.program_build_info.LOG))

data_one = np.arange(start=0, stop=100, step=1, dtype=np.float32)
data_two = -np.arange(start=0, stop=100, step=1, dtype=np.float32)

# Create buffers
flags = cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR
buffer_one = cl.Buffer(context, flags, hostbuf=data_one)
buffer_two = cl.Buffer(context, flags, hostbuf=data_two)

# Set buffers as arguments to the kernel
# The arguments can also be specified by calling kernel(....) directly instead
kernel = prog.blank  # Note: Every call like this produces a new object
kernel.set_arg(0, buffer_one)
kernel.set_arg(1, buffer_two)

# Enqueue kernel (with arguments)
n_globals = data_one.shape
n_locals = None
cl.enqueue_nd_range_kernel(queue, kernel, n_globals, n_locals)

# Enqueue command to copy from buffer one to buffer two
cl.enqueue_copy(queue, dest=buffer_two, src=buffer_one)

# Enqueue command to map from buffer two to host memory
# enqueue_map_buffer(queue, buf, flags, offset, shape, dtype, order="C", strides=None, wait_for=None, is_blocking=True)
(result_array, _) = cl.enqueue_map_buffer(queue,
                                          buf=buffer_two,
                                          flags=cl.map_flags.READ,
                                          offset=0,
                                          shape=(100,),
                                          dtype=np.float32)

print('\nSource array:')
print(data_two)

print('\nAfter copy back:')
print(result_array)


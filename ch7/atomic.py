'''
Listing 7.8: Testing atomic operations
'''

import numpy as np
import pyopencl as cl
import utility

kernel_src = '''
__kernel void atomic(__global int* x) {

   __local int a, b;

   a = 0;
   b = 0;

   /* Increment without atomic add */
   a++;

   /* Increment with atomic add */
   atomic_inc(&b);

   x[0] = a;
   x[1] = b;
}
'''

# Get device and context, create command queue and program
dev = utility.get_default_device()
context = cl.Context(devices=[dev])
queue = cl.CommandQueue(context, dev)

# Build program in the specified context using the kernel source code
prog = cl.Program(context, kernel_src)
try:
    prog.build(options=['-Werror'], devices=[dev])
except:
    print('Build log:')
    print(prog.get_build_info(dev, cl.program_build_info.LOG))
    raise

# Data
data = np.empty(shape=(2,), dtype=np.int32)

# Create input/output buffer
data_buff = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, size=data.nbytes)

# Enqueue kernel
global_size = (8,)
local_size = (4,)

# __call__(queue, global_size, local_size, *args, global_offset=None, wait_for=None, g_times_l=False)
prog.atomic(queue, global_size, local_size, data_buff)


# Print averaged results
cl.enqueue_copy(queue, dest=data, src=data_buff, is_blocking=True)

print('Increment: ' + str(data[0]))
print('Atomic increment: ' + str(data[1]))

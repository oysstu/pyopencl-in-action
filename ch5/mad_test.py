'''
Listing 5.5: Multiply-and-add large numbers
'''

import numpy as np
import pyopencl as cl
import pyopencl.array
import re
import utility

kernel_src = '''
__kernel void mad_test(__global uint *result) {

   uint a = 0x123456;
   uint b = 0x112233;
   uint c = 0x111111;

   result[0] = mad24(a, b, c);
   result[1] = mad_hi(a, b, c);
}
'''

# Get device and context, create command queue and program
dev = utility.get_default_device()
context = cl.Context(devices=[dev], properties=None, dev_type=None, cache_dir=None)
queue = cl.CommandQueue(context, dev, properties=cl.command_queue_properties.PROFILING_ENABLE)

# Build program in the specified context using the kernel source code
prog = cl.Program(context, kernel_src)
try:
    prog.build(options=['-Werror'], devices=[dev], cache_dir=None)
except:
    print('Build log:')
    print(prog.get_build_info(dev, cl.program_build_info.LOG))
    raise

# Data and buffers
out = np.empty(shape=(2,), dtype=np.uint32)
out_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, size=out.nbytes)

# Enqueue kernel (with argument specified directly)
global_size = (1,)
local_size = None

# __call__(queue, global_size, local_size, *args, global_offset=None, wait_for=None, g_times_l=False)
prog.mad_test(queue, global_size, local_size, out_buffer)

# Enqueue command to copy from buffer_out to host memory
cl.enqueue_copy(queue, dest=out, src=out_buffer, is_blocking=True)

# Merge upper and lower bits and print as (upper case) hex
out_joined = np.left_shift(out[1], 32).astype(np.uint64) + np.uint64(out[0])
print('Output: ' + re.sub('[a-f]+', lambda x: x.group(0).upper(), hex(out_joined)))



'''
Listing 5.7: Selecting component content
'''

import numpy as np
import pyopencl as cl
import pyopencl.array
import utility

kernel_src = '''
__kernel void select_test(__global float4 *s1,
                          __global uchar2 *s2) {

   /* Execute select */
   int4 mask1 = (int4)(-1, 0, -1, 0);
   float4 input1 = (float4)(0.25f, 0.5f, 0.75f, 1.0f);
   float4 input2 = (float4)(1.25f, 1.5f, 1.75f, 2.0f);
   *s1 = select(input1, input2, mask1);

   /* Execute bitselect */
   uchar2 mask2 = (uchar2)(0xAA, 0x55);
   uchar2 input3 = (uchar2)(0x0F, 0x0F);
   uchar2 input4 = (uchar2)(0x33, 0x33);
   *s2 = bitselect(input3, input4, mask2);
}
'''

# Get device and context, create command queue and program
dev = utility.get_default_device()
context = cl.Context(devices=[dev])
queue = cl.CommandQueue(context, dev)

# Build program in the specified context using the kernel source code
prog = cl.Program(context, kernel_src)
try:
    prog.build(options=['-Werror'], devices=[dev], cache_dir=None)
except:
    print('Build log:')
    print(prog.get_build_info(dev, cl.program_build_info.LOG))
    raise

# Data and buffers
s1 = cl.array.vec.zeros_float4()
s2 = cl.array.vec.zeros_uchar2()

# Create output buffer
s1_buff = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, size=s1.nbytes)
s2_buff = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, size=s2.nbytes)

# Enqueue kernel (with argument specified directly)
global_size = (1,)
local_size = None

# __call__(queue, global_size, local_size, *args, global_offset=None, wait_for=None, g_times_l=False)
prog.select_test(queue, global_size, local_size, s1_buff, s2_buff)

# Enqueue command to copy from buffers to host memory
cl.enqueue_copy(queue, dest=s1, src=s1_buff, is_blocking=False)
cl.enqueue_copy(queue, dest=s2, src=s2_buff, is_blocking=True)

print('S1 output: ' + str(s1))
print('S2 output: ' + str(s2))


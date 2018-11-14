'''
Listing 5.6: Shuffling vector components
'''

import numpy as np
import pyopencl as cl
import pyopencl.array
import utility

kernel_src = '''
__kernel void shuffle_test(__global float8 *s1,
                           __global char16 *s2) {

   /* Execute the first example */
   uint8 mask1 = (uint8)(1, 2, 0, 1, 3, 1, 2, 3);
   float4 input = (float4)(0.25f, 0.5f, 0.75f, 1.0f);
   *s1 = shuffle(input, mask1);

   /* Execute the second example */
   uchar16 mask2 = (uchar16)(6, 10, 5, 2, 8, 0, 9, 14,
                             7, 5, 12, 3, 11, 15, 1, 13);
   char8 input1 = (char8)('l', 'o', 'f', 'c', 'a', 'u', 's', 'f');
   char8 input2 = (char8)('f', 'e', 'h', 't', 'n', 'n', '2', 'i');
   *s2 = shuffle2(input1, input2, mask2);
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

# Data and buffers
s1 = cl.array.vec.zeros_float8()
s2 = np.empty(shape=(16,), dtype=np.character)  # zeros_char16 would also work

# Create output buffer
s1_buff = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, size=s1.nbytes)
s2_buff = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, size=s2.nbytes)

# Enqueue kernel (with argument specified directly)
global_size = (1,)
local_size = None

# __call__(queue, global_size, local_size, *args, global_offset=None, wait_for=None, g_times_l=False)
prog.shuffle_test(queue, global_size, local_size, s1_buff, s2_buff)

# Enqueue command to copy from buffers to host memory
cl.enqueue_copy(queue, dest=s1, src=s1_buff, is_blocking=False)
cl.enqueue_copy(queue, dest=s2, src=s2_buff, is_blocking=True)

print('S1 output: ' + str(s1))
print('S2 output: ' + str(s2.tobytes().decode('utf-8')))


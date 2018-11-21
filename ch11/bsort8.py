"""
Listing 11.2: An eight-element bitonic sort
"""

from io import open
import numpy as np
import pyopencl as cl
import utility

SORT_ASCENDING = True

kernel_src = '''
#define UP 0
#define DOWN -1

/* Sort elements in a vector */
#define SORT_VECTOR(input, dir)                                   \
   comp = (input < shuffle(input, mask1)) ^ dir;                    \
   input = shuffle(input, as_uint4(comp + add1));                 \
   comp = (input < shuffle(input, mask2)) ^ dir;                    \
   input = shuffle(input, as_uint4(comp * 2 + add2));             \
   comp = (input < shuffle(input, mask3)) ^ dir;                    \
   input = shuffle(input, as_uint4(comp + add3));                 \

/* Sort elements between two vectors */
#define SWAP_VECTORS(input1, input2, dir)                         \
   temp = input1;                                                 \
   comp = ((input1 < input2) ^ dir) * 4 + add4;                     \
   input1 = shuffle2(input1, input2, as_uint4(comp));             \
   input2 = shuffle2(input2, temp, as_uint4(comp));               \

__kernel void bsort8(__global float4 *data, int dir) {

   float4 input1, input2, temp;
   int4 comp;

   uint4 mask1 = (uint4)(1, 0, 3, 2);
   uint4 mask2 = (uint4)(2, 3, 0, 1);
   uint4 mask3 = (uint4)(3, 2, 1, 0);

   int4 add1 = (int4)(1, 1, 3, 3);
   int4 add2 = (int4)(2, 3, 2, 3);
   int4 add3 = (int4)(1, 2, 2, 3);
   int4 add4 = (int4)(4, 5, 6, 7);

   input1 = data[0];
   input2 = data[1];

   SORT_VECTOR(input1, UP)
   SORT_VECTOR(input2, DOWN)

   SWAP_VECTORS(input1, input2, dir)

   SORT_VECTOR(input1, dir)
   SORT_VECTOR(input2, dir)

   data[0] = input1;
   data[1] = input2;
}
'''

# Get device and context, create command queue and program
dev = utility.get_default_device()
context = cl.Context(devices=[dev])
queue = cl.CommandQueue(context, dev, properties=cl.command_queue_properties.PROFILING_ENABLE)

# Build program in the specified context using the kernel source code
prog = cl.Program(context, kernel_src)
try:
    prog.build(options=['-Werror'], devices=[dev])
except:
    print('Build log:')
    print(prog.get_build_info(dev, cl.program_build_info.LOG))
    raise

# Data and device buffers
data = np.array([3., 5., 4., 6., 0., 7., 2., 1.], dtype=np.float32)
print('Input: ' + str(data))

mf = cl.mem_flags
data_buffer = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=data)

# Execute kernels
local_size = (1,)
global_size = (1,)

start_event = prog.bsort8(queue, global_size, local_size, data_buffer, np.int32(0 if SORT_ASCENDING else -1))

cl.enqueue_copy(queue, dest=data, src=data_buffer, is_blocking=True)

print('Output: ' + str(data))







"""
Listing 11.4: An eight-element radix sort
"""

from io import open
import numpy as np
import pyopencl as cl
import utility

NUM_SHORTS = 8

kernel_src = '''
__kernel void radix_sort8(__global ushort8 *global_data) {

   typedef union {
      ushort8 vec;
      ushort array[8];
   } vec_array;

   uint one_count, zero_count;
   uint cmp_value = 1;
   vec_array mask, ones, data;

   data.vec = global_data[0];

   /* Rearrange elements according to bits */
   for(int i=0; i<3; i++) {
      zero_count = 0;
      one_count = 0;

      /* Iterate through each element in the input vector */
      for(int j = 0; j < 8; j++) {
         if(data.array[j] & cmp_value)

            /* Place element in ones vector */
            ones.array[one_count++] = data.array[j];
         else {

            /* Increment number of elements with zero */
            mask.array[zero_count++] = j;
         }
      }

      /* Create sorted vector */
      for(int j = zero_count; j < 8; j++)
         mask.array[j] = 8 - zero_count + j;
      data.vec = shuffle2(data.vec, ones.vec, mask.vec);
      cmp_value <<= 1;
   }
   global_data[0] = data.vec;
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

# Data and device buffers
data = np.arange(start=0, stop=NUM_SHORTS, dtype=np.uint16)
np.random.shuffle(data)
print('Input: ' + str(data))

mf = cl.mem_flags
data_buffer = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=data)

# Execute kernel
# radix_sort8(__global ushort8 *global_data)
kernel = prog.radix_sort8
kernel.set_arg(0, data_buffer)
cl.enqueue_task(queue, kernel)
cl.enqueue_copy(queue, dest=data, src=data_buffer, is_blocking=True)

print('Output: ' + str(data))







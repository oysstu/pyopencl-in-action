'''
Listing 11.1: Implementing MapReduce in OpenCL (vectorized string searching)
'''

from io import open
import numpy as np
import pyopencl as cl
import utility

TEXT_FILE = 'kafka.txt'

kernel_src = '''
__kernel void string_search(char16 pattern, __global char* text,
     int chars_per_item, __local int* local_result,
     __global int* global_result) {

   char16 text_vector, check_vector;

   /* initialize local data */
   local_result[0] = 0;
   local_result[1] = 0;
   local_result[2] = 0;
   local_result[3] = 0;

   /* Make sure previous processing has completed */
   barrier(CLK_LOCAL_MEM_FENCE);

   int item_offset = get_global_id(0) * chars_per_item;

   /* Iterate through characters in text */
   for(int i=item_offset; i<item_offset + chars_per_item; i++) {

      /* load global text into private buffer */
      text_vector = vload16(0, text + i);

      /* compare text vector and pattern */
      check_vector = text_vector == pattern;

      /* Check for 'that' */
      if(all(check_vector.s0123))
         atomic_inc(local_result);

      /* Check for 'with' */
      if(all(check_vector.s4567))
         atomic_inc(local_result + 1);

      /* Check for 'have' */
      if(all(check_vector.s89AB))
         atomic_inc(local_result + 2);

      /* Check for 'from' */
      if(all(check_vector.sCDEF))
         atomic_inc(local_result + 3);
   }

   /* Make sure local processing has completed */
   barrier(CLK_GLOBAL_MEM_FENCE);

   /* Perform global reduction */
   if(get_local_id(0) == 0) {
      atomic_add(global_result, local_result[0]);
      atomic_add(global_result + 1, local_result[1]);
      atomic_add(global_result + 2, local_result[2]);
      atomic_add(global_result + 3, local_result[3]);
   }
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
with open(TEXT_FILE, 'r') as f:
    text = np.fromstring(f.read(), dtype=np.uint8)

pattern = np.fromstring('thatwithhavefrom', dtype=np.uint8)
result = np.zeros(shape=(4,), dtype=np.int32)

mf = cl.mem_flags
text_buffer = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=text)
result_buffer = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=result)
local_result = cl.LocalMemory(4 * np.dtype(np.int32).itemsize)

# Execute kernels
wg_max_compute_units = dev.max_compute_units
wg_max_size = dev.max_work_group_size
local_size = (wg_max_size,)
global_size = (wg_max_compute_units * wg_max_size,)

chars_per_item = np.int32(text.shape[0] // global_size[0] + 1)

start_event = prog.string_search(queue, global_size, local_size,
                                 pattern,
                                 text_buffer,
                                 chars_per_item,
                                 local_result,
                                 result_buffer)

cl.enqueue_copy(queue, dest=result, src=result_buffer, is_blocking=True)

keywords = ['that', 'with', 'have', 'from']
for k, n in zip(keywords, result):
    print('Number of occurrences of \'{}\': {}'.format(k, n))







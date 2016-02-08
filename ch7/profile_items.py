'''
Listing 7.7: Profiling data partitioning
'''

import numpy as np
import pyopencl as cl
import pyopencl.array
import utility

NUM_INTS= 4096
NUM_ITEMS = 512
NUM_ITERATIONS = 2000

kernel_src = '''
__kernel void profile_items(__global int4 *x, int num_ints) {

   int num_vectors = num_ints/(4 * get_global_size(0));

   x += get_global_id(0) * num_vectors;
   for(int i=0; i<num_vectors; i++) {
      x[i] += 1;
      x[i] *= 2;
      x[i] /= 3;
   }
}
'''

# Get device and context, create command queue and program
dev = utility.get_default_device()
context = cl.Context(devices=[dev], properties=None, dev_type=None, cache_dir=None)

# Create a command queue with the profiling flag enabled
queue = cl.CommandQueue(context, dev, properties=cl.command_queue_properties.PROFILING_ENABLE)

# Build program in the specified context using the kernel source code
prog = cl.Program(context, kernel_src)
try:
    prog.build(options=['-Werror'], devices=[dev], cache_dir=None)
except:
    print('Build log:')
    print(prog.get_build_info(dev, cl.program_build_info.LOG))
    raise

# Data
data = np.arange(start=0, stop=NUM_INTS, dtype=np.int32)

# Create input/output buffer
data_buff = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=data)

# Enqueue kernel (with argument specified directly)
global_size = (NUM_ITEMS,)
local_size = None

# Execute the kernel repeatedly using enqueue_read
total_time = 0.0
for i in range(NUM_ITERATIONS):
    # Enqueue kernel
    # __call__(queue, global_size, local_size, *args, global_offset=None, wait_for=None, g_times_l=False)
    kernel_event = prog.profile_items(queue, global_size, local_size, data_buff, np.int32(NUM_INTS))

    # Finish processing the queue and get profiling information
    queue.finish()
    total_time += kernel_event.profile.end - kernel_event.profile.start


# Print averaged results
print('Average time (ms): {}'.format(total_time / ( NUM_ITERATIONS * 1000)))


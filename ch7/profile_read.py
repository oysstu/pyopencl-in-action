'''
Listing 7.6: Profiling data transfer
'''

import numpy as np
import pyopencl as cl
import pyopencl.array
import utility

NUM_VECTORS = 8192
NUM_ITERATIONS = 2000

kernel_src = '''
__kernel void profile_read(__global char16 *c, int num) {

   for(int i=0; i<num; i++) {
      c[i] = (char16)(5);
   }
}
'''

# Get device and context, create command queue and program
dev = utility.get_default_device()
context = cl.Context(devices=[dev])

# Create a command queue with the profiling flag enabled
queue = cl.CommandQueue(context, dev, properties=cl.command_queue_properties.PROFILING_ENABLE)

# Build program in the specified context using the kernel source code
prog = cl.Program(context, kernel_src)
try:
    prog.build(options=['-Werror'], devices=[dev])
except:
    print('Build log:')
    print(prog.get_build_info(dev, cl.program_build_info.LOG))
    raise

# Data
c = np.empty(shape=(NUM_VECTORS,), dtype=cl.array.vec.char16)

# Create output buffer
c_buff = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, size=c.nbytes)

# Enqueue kernel (with argument specified directly)
global_size = (1,)
local_size = None

# There is some overhead involved with spawning a new kernel (code caching)
# A good rule of thumb is therefore to create the kernel object outside of loops
# Ref: https://lists.tiker.net/pipermail/pyopencl/2016-February/002107.html
kernel = prog.profile_read

# Execute the kernel repeatedly using enqueue_read
read_time = 0.0
for i in range(NUM_ITERATIONS):
    # __call__(queue, global_size, local_size, *args, global_offset=None, wait_for=None, g_times_l=False)
    # Store kernel execution event (return value)
    kernel_event = kernel(queue, global_size, local_size, c_buff, np.int32(NUM_VECTORS))

    # Enqueue command to copy from buffers to host memory
    # Store data transfer event (return value)
    prof_event = cl.enqueue_copy(queue, dest=c, src=c_buff, is_blocking=True)

    read_time += prof_event.profile.end - prof_event.profile.start


# Execute the kernel repeatedly using enqueue_map_buffer
map_time = 0.0
for i in range(NUM_ITERATIONS):
    # __call__(queue, global_size, local_size, *args, global_offset=None, wait_for=None, g_times_l=False)
    # Store kernel execution event (return value)
    kernel_event = kernel(queue, global_size, local_size, c_buff, np.int32(NUM_VECTORS))

    # Enqueue command to map from buffer two to host memory
    (result_array, prof_event) = cl.enqueue_map_buffer(queue,
                                                       buf=c_buff,
                                                       flags=cl.map_flags.READ,
                                                       offset=0,
                                                       shape=(NUM_VECTORS,),
                                                       dtype=cl.array.vec.char16)

    map_time += prof_event.profile.end - prof_event.profile.start

    # Release the mapping (is this necessary?)
    result_array.base.release(queue)

# Print averaged results
print('Average read time (ms): {}'.format(read_time / ( NUM_ITERATIONS * 1000)))
print('Average map time (ms): {}'.format(map_time / ( NUM_ITERATIONS * 1000)))
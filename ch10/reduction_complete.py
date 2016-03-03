'''
Combined example of the following listings
Listing 10.4: Computing the final reduction (multiple passes over the data)
'''

import numpy as np
import pyopencl as cl
import utility
from string import Template

ARRAY_SIZE = 2**20  # Note: The code assumes that the size of the array is divisible by the max work group size
VECTOR_LENGTH = 4   # This changes the relevant memory sizes and substitutes the kernel vector types

kernel_src = Template('''
__kernel void reduction_vector(__global float$N* data, __local float$N* partial_sums) {

   int lid = get_local_id(0);
   int group_size = get_local_size(0);

   partial_sums[lid] = data[get_global_id(0)];
   barrier(CLK_LOCAL_MEM_FENCE);

   for(int i = group_size/2; i>0; i >>= 1) {
      if(lid < i) {
         partial_sums[lid] += partial_sums[lid + i];
      }
      barrier(CLK_LOCAL_MEM_FENCE);
   }

   if(lid == 0) {
      data[get_group_id(0)] = partial_sums[0];
   }
}

__kernel void reduction_complete(__global float$N* data, __local float$N* partial_sums, __global float* sum) {

   int lid = get_local_id(0);
   int group_size = get_local_size(0);

   partial_sums[lid] = data[get_local_id(0)];
   barrier(CLK_LOCAL_MEM_FENCE);

   for(int i = group_size/2; i>0; i >>= 1) {
      if(lid < i) {
         partial_sums[lid] += partial_sums[lid + i];
      }
      barrier(CLK_LOCAL_MEM_FENCE);
   }

   if(lid == 0) {
      *sum = dot(partial_sums[0], (float$N)(1.0f));
   }
}
''').substitute(N=VECTOR_LENGTH)

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

# Determine local size
wg_max_size = dev.max_work_group_size
num_groups = ARRAY_SIZE // wg_max_size

print('WG Max Size: ' + str(wg_max_size))
print('Num groups: ' + str(num_groups))
print('Local mem size: ' + str(dev.local_mem_size))

# Print the preferred/native floatN lengths (which is optimal for the compiler/hardware, respectively)
# Vectorization can still yield higher throughput even if preferred/native is 1, due to better use of memory bandwidth
print('Preferred floatN size: ' + str(dev.preferred_vector_width_float))
print('Preferred floatN size: ' + str(dev.native_vector_width_float))

# Data and device buffers
data = np.arange(start=0, stop=ARRAY_SIZE, dtype=np.float32)
result = np.zeros(shape=(1,), dtype=np.float32)

data_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=data)
sum_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, size=np.dtype(np.float32).itemsize)

partial_sums = cl.LocalMemory(wg_max_size * np.dtype(np.float32).itemsize * VECTOR_LENGTH)

# Execute kernels
local_size = wg_max_size
global_size = ARRAY_SIZE // VECTOR_LENGTH
start_event = prog.reduction_vector(queue, (global_size,), (local_size,), data_buffer, partial_sums)
print('\nGlobal size: ' + str(global_size))

# There is some overhead involved with spawning a new kernel (code caching)
# A good rule of thumb is therefore to create the kernel object outside of loops
# Ref: https://lists.tiker.net/pipermail/pyopencl/2016-February/002107.html
kernel_reduction_vector = prog.reduction_vector

# Perform successive stages of reduction
while global_size // local_size > local_size:
    global_size = global_size // local_size
    kernel_reduction_vector(queue, (global_size,), (local_size,), data_buffer, partial_sums)
    print('Global size: ' + str(global_size))

# Perform final reduction when the workload fits within a single work group
# The local size is then set equal to the global size
global_size = global_size // local_size
print('Global size: ' + str(global_size))
end_event = prog.reduction_complete(queue, (global_size,), (global_size,), data_buffer, partial_sums, sum_buffer)
queue.finish()

print('\nTotal time (ms): ' + str((end_event.profile.end - start_event.profile.start)/1000))

cl.enqueue_copy(queue, dest=result, src=sum_buffer, is_blocking=True)

actual_sum = np.float32(ARRAY_SIZE/2*(ARRAY_SIZE-1))

print('Actual sum: ' + str(actual_sum))
print('Computed sum: ' + str(result[0]))




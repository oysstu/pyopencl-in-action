'''
Listing 10.2: Reduction using scalars
Listing 10.3: Reduction using vectors
'''

import numpy as np
import pyopencl as cl
import utility

# Note: The code assumes that the size of the array is divisible by the max work group size
ARRAY_SIZE = 2**20

kernel_src = '''
__kernel void reduction_scalar(__global float* data,
      __local float* partial_sums, __global float* output) {

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
      output[get_group_id(0)] = partial_sums[0];
   }
}


__kernel void reduction_vector(__global float4* data,
      __local float4* partial_sums, __global float* output) {

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
      output[get_group_id(0)] = dot(partial_sums[0], (float4)(1.0f));
   }
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

# Determine local size
wg_max_size = dev.max_work_group_size
num_groups = ARRAY_SIZE // wg_max_size

print('WG Max Size: ' + str(wg_max_size))
print('Num groups: ' + str(num_groups))
print('Local mem size: ' + str(dev.local_mem_size))

# Data and device buffers
data = np.arange(start=0, stop=ARRAY_SIZE, dtype=np.float32)
scalar_sum = np.zeros(shape=(num_groups,), dtype=np.float32)
vector_sum = np.zeros(shape=(num_groups // 4,), dtype=np.float32)

data_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=data)
scalar_sum_buffer = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=scalar_sum)
vector_sum_buffer = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=vector_sum)

scalar_local_mem = cl.LocalMemory(wg_max_size * np.dtype(np.float32).itemsize)
vector_local_mem = cl.LocalMemory(wg_max_size * np.dtype(np.float32).itemsize * 4)

# Execute kernels and copy results
local_size = (wg_max_size,)

# Scalar kernel
global_size = (ARRAY_SIZE,)
scalar_event = prog.reduction_scalar(queue, global_size, local_size, data_buffer, scalar_local_mem, scalar_sum_buffer)

cl.enqueue_copy(queue, dest=scalar_sum, src=scalar_sum_buffer, is_blocking=True)

queue.finish()
scalar_time = scalar_event.profile.end - scalar_event.profile.start
del scalar_local_mem, scalar_sum_buffer, scalar_event

# Vector kernel
global_size = (ARRAY_SIZE // 4,)
vector_event = prog.reduction_vector(queue, global_size, local_size, data_buffer, vector_local_mem, vector_sum_buffer)
cl.enqueue_copy(queue, dest=vector_sum, src=vector_sum_buffer, is_blocking=True)

queue.finish()
vector_time = vector_event.profile.end - vector_event.profile.start

actual_sum = np.float32(ARRAY_SIZE/2*(ARRAY_SIZE-1))

print('\nActual sum: ' + str(actual_sum))
print('Scalar sum: ' + str(scalar_sum.sum()))
print('Vector sum: ' + str(vector_sum.sum()))

print('\nScalar time (ms): ' + str(scalar_time/1000))
print('Vector time (ms): ' + str(vector_time/1000))


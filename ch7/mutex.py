'''
Listing 7.9: Mutex-based synchronization
Note: This freezes/crashes when executed on gpu (nvidia),
TODO: Fix this for GPUs
'''

import numpy as np
import pyopencl as cl
import utility

kernel_src = '''
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#define LOCK(a) atom_cmpxchg(a, 0, 1)
#define UNLOCK(a) atom_xchg(a, 0)

__kernel void mutex(__global int *mutex, __global int *sum) {
    while(LOCK(mutex));
    *sum += 1;
    UNLOCK(mutex);
    int waiting = 1;
    while(waiting) {
        while(LOCK(mutex));
        if(*sum == get_global_size(0)) {
            waiting = 0;
        }
        UNLOCK(mutex);
    }
}
'''

# Get device and context, create command queue and program
dev = utility.get_default_device(use_gpu=False)
context = cl.Context(devices=[dev], properties=None, dev_type=None, cache_dir=None)
queue = cl.CommandQueue(context, dev, properties=None)

# Check for cl_khr_global_int32_base_atomics availability
if 'cl_khr_global_int32_base_atomics' not in dev.extensions.strip().split(' '):
    raise RuntimeError('Selected device does not support int32 atomic operations.')

# Build program in the specified context using the kernel source code
prog = cl.Program(context, kernel_src)
try:
    prog.build(options=['-Werror'], devices=[dev], cache_dir=None)
except:
    print('Build log:')
    print(prog.get_build_info(dev, cl.program_build_info.LOG))
    raise

# Input
mutex = np.zeros(shape=(1,), dtype=np.int32)
mutex_buff = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=mutex)

sum_out = np.zeros(shape=(1,), dtype=np.int32)
sum_buff = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=sum_out)

# Enqueue kernel
global_size = (4,)
local_size = None

# __call__(queue, global_size, local_size, *args, global_offset=None, wait_for=None, g_times_l=False)
prog.mutex(queue, global_size, local_size, mutex_buff, sum_buff)

# Print averaged results
cl.enqueue_copy(queue, dest=sum_out, src=sum_buff, is_blocking=True)

print('Sum: ' + str(sum_out))

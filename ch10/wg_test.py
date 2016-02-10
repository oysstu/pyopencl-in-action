'''
Listing 10.1: Obtaining kernel/device information
'''

import numpy as np
import pyopencl as cl
import utility

kernel_src = '''
__kernel void blank(__global float *a, __global float *b) {
}
'''

# Get device and context, create command queue and program
dev = utility.get_default_device()
context = cl.Context(devices=[dev], properties=None, dev_type=None, cache_dir=None)
queue = cl.CommandQueue(context, dev, properties=None)

# Build program in the specified context using the kernel source code
prog = cl.Program(context, kernel_src)
try:
    prog.build(options=['-Werror'], devices=[dev], cache_dir=None)
except:
    print('Build log:')
    print(prog.get_build_info(dev, cl.program_build_info.LOG))
    raise

# Get kernel object
kernel = prog.blank
wg_size = kernel.get_work_group_info(cl.kernel_work_group_info.WORK_GROUP_SIZE, dev)
wg_multiple = kernel.get_work_group_info(cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE, dev)
local_usage = kernel.get_work_group_info(cl.kernel_work_group_info.LOCAL_MEM_SIZE, dev)
private_usage = kernel.get_work_group_info(cl.kernel_work_group_info.PRIVATE_MEM_SIZE, dev)

print('For kernel {} running on device {}:'.format(kernel.function_name, dev.name))
print('\t Max work-group size: {}'.format(wg_size))
print('\t Recommended work-group multiple: {}'.format(wg_multiple))
print('\t Local mem used: {} of {}'.format(local_usage, dev.local_mem_size))
print('\t Private mem used: {}'.format(private_usage))


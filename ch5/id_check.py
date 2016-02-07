'''
Listing 5.2: Testing work-item/work-group IDs
'''

import numpy as np
import pyopencl as cl
import utility

kernel_src = '''
__kernel void id_check(__global float *output) {

   /* Access work-item/work-group information */
   size_t global_id_0 = get_global_id(0);
   size_t global_id_1 = get_global_id(1);
   size_t global_size_0 = get_global_size(0);
   size_t offset_0 = get_global_offset(0);
   size_t offset_1 = get_global_offset(1);
   size_t local_id_0 = get_local_id(0);
   size_t local_id_1 = get_local_id(1);

   /* Determine array index */
   int index_0 = global_id_0 - offset_0;
   int index_1 = global_id_1 - offset_1;
   int index = index_1 * global_size_0 + index_0;

   /* Set float data */
   float f = global_id_0 * 10.0f + global_id_1 * 1.0f;
   f += local_id_0 * 0.1f + local_id_1 * 0.01f;

   output[index] = f;
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


# Create output buffer
out = np.zeros(shape=(4, 6), dtype=np.float32)
buffer_out = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, size=out.nbytes)

# Enqueue kernel (with argument specified directly)
global_offset = (3, 5)
global_size = (6, 4)
local_size = (3, 2)

# __call__(queue, global_size, local_size, *args, global_offset=None, wait_for=None, g_times_l=False)
prog.id_check(queue, global_size, local_size, buffer_out, global_offset=global_offset)

# Enqueue command to copy from buffer_out to host memory
cl.enqueue_copy(queue, dest=out, src=buffer_out, is_blocking=True)

# Only print 2 decimals
np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
print('Output:\n' + str(out))


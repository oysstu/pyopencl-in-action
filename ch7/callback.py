'''
Listing 7.2: Host notification
'''

import numpy as np
import pyopencl as cl
import utility
from time import sleep

kernel_src = '''
__kernel void callback(__global float *buffer) {
   float4 five_vector = (float4)(5.0);

   for(int i=0; i<1024; i++) {
      vstore4(five_vector, i, buffer);
   }
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

# Data
# Can either use 1024 float4 or 4096 scalar float
out = np.empty(shape=(4096,), dtype=np.float32)

# Create output buffer
out_buff = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, size=out.nbytes)

# Enqueue kernel (with argument specified directly)
global_size = (1,)
local_size = None


# Define callback functions
def kernel_complete(status):
    print('Kernel complete')

def read_complete(status):
    print('Read complete')

# __call__(queue, global_size, local_size, *args, global_offset=None, wait_for=None, g_times_l=False)
# Store kernel execution event (return value)
kernel_event = prog.callback(queue, global_size, local_size, out_buff)
kernel_event.set_callback(cl.command_execution_status.COMPLETE, kernel_complete)

print('Kernel enqueued')

# Enqueue command to copy from buffers to host memory
# Store data transfer event (return value)
read_event = cl.enqueue_copy(queue, dest=out, src=out_buff, is_blocking=False)
read_event.set_callback(cl.command_execution_status.COMPLETE, read_complete)

print('Read enqueued')

# Allow the callback functions to finish
read_event.wait()

# The callback function might not execute if the script terminates too quickly
sleep(1)

# If this is placed before the sleep function, it is sometimes executed before "read complete" is printed
print('Output: ' + str(out))




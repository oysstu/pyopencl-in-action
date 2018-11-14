'''
Listing 7.3: Stalling commands with user events (host events)
'''

from time import sleep
import numpy as np
import pyopencl as cl
import utility

kernel_src = '''
__kernel void user_event(__global float4 *v) {

   *v *= -1.0f;
}
'''

# Get device and context, create command queue and program
dev = utility.get_default_device()
context = cl.Context(devices=[dev])

# Create command queue with out of order execution enabled
queue = cl.CommandQueue(context, dev, properties=cl.command_queue_properties.OUT_OF_ORDER_EXEC_MODE_ENABLE)

# Build program in the specified context using the kernel source code
prog = cl.Program(context, kernel_src)
try:
    prog.build(options=['-Werror'], devices=[dev])
except:
    print('Build log:')
    print(prog.get_build_info(dev, cl.program_build_info.LOG))
    raise

# Data
v = np.arange(4, dtype=np.float32)
print('Input: ' + str(v))

# Create output buffer
v_buff = cl.Buffer(context, flags=cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=v)

# Create user event
user_event = cl.UserEvent(context)


def read_complete(status, data):
    print('Output: ' + str(data))

# Enqueue kernel that waits for user event before executing
global_size = (1,)
local_size = None

# __call__(queue, global_size, local_size, *args, global_offset=None, wait_for=None, g_times_l=False)
kernel_event = prog.user_event(queue, global_size, local_size, v_buff, wait_for=[user_event])

# Enqueue command to copy from buffers to host memory
read_event = cl.enqueue_copy(queue, dest=v, src=v_buff, is_blocking=False, wait_for=[kernel_event])

# Set the callback event to read_complete, pass data by wrapping in a lambda function
read_event.set_callback(cl.command_execution_status.COMPLETE, lambda s: read_complete(s, data=v))

# The kernel should not execute yet, wait and verify
sleep(1)

print('Set user event (kernel start)')
user_event.set_status(cl.command_execution_status.COMPLETE)

sleep(1)

print('Script finished')




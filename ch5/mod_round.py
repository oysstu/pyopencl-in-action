'''
Listing 5.3: Division and rounding
'''

import numpy as np
import pyopencl as cl
import pyopencl.array
import utility

kernel_src = '''
__kernel void mod_round(__global float *mod_input,
                        __global float *mod_output,
                        __global float4 *round_input,
                        __global float4 *round_output) {

   /* Use fmod and remainder: 317.0, 23.0 */
   mod_output[0] = fmod(mod_input[0], mod_input[1]);
   mod_output[1] = remainder(mod_input[0], mod_input[1]);

   /* Rounds the input values: -6.5, -3.5, 3.5, and 6.5 */
   round_output[0] = rint(*round_input);
   round_output[1] = round(*round_input);
   round_output[2] = ceil(*round_input);
   round_output[3] = floor(*round_input);
   round_output[4] = trunc(*round_input);
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

# Data and buffers
mod_input = np.array([317.0, 23.0], dtype=np.float32)
mod_output = np.empty_like(mod_input, dtype=np.float32)
round_input = cl.array.vec.make_float4(-6.5, -3.5, 3.5, 6.5)
round_output = np.empty(shape=(5,), dtype=cl.array.vec.float4)

# Create output buffer
mod_arg = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=mod_input)
round_arg = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=round_input)
mod_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, size=mod_output.nbytes)
round_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, size=round_output.nbytes)

# Enqueue kernel (with argument specified directly)
global_size = (1,)
local_size = None

# __call__(queue, global_size, local_size, *args, global_offset=None, wait_for=None, g_times_l=False)
prog.mod_round(queue, global_size, local_size, mod_arg, mod_buffer, round_arg, round_buffer)

# Enqueue command to copy from buffer_out to host memory
cl.enqueue_copy(queue, dest=mod_output, src=mod_buffer, is_blocking=False)
cl.enqueue_copy(queue, dest=round_output, src=round_buffer, is_blocking=True)

print('Mod output:\n' + str(mod_output))
print('Round output:\n' + str(round_output))


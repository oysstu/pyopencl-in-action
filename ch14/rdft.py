"""
Listing 14.1: The discrete Fourier transform (for real numbers)
"""

from io import open
import numpy as np
import pyopencl as cl
import utility

NUM_POINTS = 2 ** 8

kernel_src = '''
__kernel void rdft(__global float *x) {

   int N = (get_global_size(0)-1)*2;
   int num_vectors = N/4;

   float X_real = 0.0f;
   float X_imag = 0.0f;

   float4 input, arg, w_real, w_imag;
   float two_pi_k_over_N =
         2*M_PI_F*get_global_id(0)/N;

   for(int i=0; i<num_vectors; i++) {
      arg = (float4) (two_pi_k_over_N*(i*4),
                      two_pi_k_over_N*(i*4+1),
                      two_pi_k_over_N*(i*4+2),
                      two_pi_k_over_N*(i*4+3));
      w_real = cos(arg);
      w_imag = sin(arg);

      input = vload4(i, x);
      X_real += dot(input, w_real);
      X_imag -= dot(input, w_imag);
   }
   barrier(CLK_GLOBAL_MEM_FENCE);

   if(get_global_id(0) == 0) {
      x[0] = X_real;
   }
   else if(get_global_id(0) == get_global_size(0)-1) {
      x[1] = X_real;
   }
   else {
      x[get_global_id(0) * 2] = X_real;
      x[get_global_id(0) * 2 + 1] = X_imag;
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

# Determine maximum work-group size
wg_max_size = dev.max_work_group_size

# Data and device buffers
input_data = np.zeros(shape=(NUM_POINTS,), dtype=np.float32)
output_data = np.empty_like(input_data, dtype=np.float32)

# Initialize data with a rectangle function
input_data[:NUM_POINTS // 4] = 1.0

mf = cl.mem_flags
data_buffer = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=input_data)

# Execute kernel
# rdft(__global float *x)
global_size = (NUM_POINTS // 2 + 1,)
local_size = None

prog.rdft(queue, global_size, local_size, data_buffer)

cl.enqueue_copy(queue, dest=output_data, src=data_buffer, is_blocking=True)

cl_fft = output_data[::2] + 1j * output_data[1::2]
np_fft = np.fft.fft(input_data)

# Print first ten complex values
np.set_printoptions(precision=4, suppress=True)
print('CL FFT:')
print(cl_fft[:10])
print('\nNumpy FFT:')
print(np_fft[:10])

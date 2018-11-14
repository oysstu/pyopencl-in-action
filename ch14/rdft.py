"""
Listing 14.1: The discrete Fourier transform (for real numbers)
"""

import numpy as np
import pyopencl as cl
import utility
import matplotlib.pyplot as plt

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
context = cl.Context(devices=[dev])
queue = cl.CommandQueue(context, dev)

# Build program in the specified context using the kernel source code
prog = cl.Program(context, kernel_src)
try:
    prog.build(options=['-Werror'], devices=[dev])
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

# Execute kernel and copy result
# rdft(__global float *x)
global_size = (NUM_POINTS // 2 + 1,)
local_size = None
prog.rdft(queue, global_size, local_size, data_buffer)

cl.enqueue_copy(queue, dest=output_data, src=data_buffer, is_blocking=True)

# Change to array of complex values
f_first = np.array([output_data[0] + 0j])  # X[0] is the DC signal (no im. component)
f_last = np.array([output_data[1] + 0j])   # X[1] is the N/2 frequency

cl_fft = output_data[2::2] + 1j * output_data[3::2]  # From there, real and im. alternates
# The final result is assembled by concatenating f0, f1 : fN/2-1, fN/2 and the conjugate of f1:fN/2-1
cl_fft = np.concatenate((f_first, cl_fft, f_last, np.conj(cl_fft[::-1])))

np_fft = np.fft.fft(input_data)

# Print first ten complex values
np.set_printoptions(precision=4, suppress=True)
print('CL FFT [0:10]:')
print(cl_fft[:10])
print('\nNumpy FFT [0:10]:')
print(np_fft[:10])

# Visualize result
cl_magnitude = np.absolute(cl_fft)
np_magnitude = np.absolute(np_fft)

# Before calculating the phase, frequencies of low magnitude should be set to zero
# This is due to numerical inaccuracies
cl_fft[cl_magnitude < 0.0001] = 0.0 + 0.0j
cl_phase = np.angle(cl_fft)
np_phase = np.angle(np_fft)
k = np.arange(0, NUM_POINTS)

f, axes = plt.subplots(4, sharex=True)
axes[0].set_title('Re')
axes[0].plot(k, np.real(cl_fft), label='OpenCL')
axes[0].plot(k, np.real(np_fft), label='Numpy')
axes[0].legend()
axes[1].set_title('Im')
axes[1].plot(k, np.imag(cl_fft), label='OpenCL')
axes[1].plot(k, np.imag(np_fft), label='Numpy')
axes[2].set_title('Magnitude')
axes[2].plot(k, cl_magnitude, label='OpenCL')
axes[2].plot(k, np_magnitude, label='Numpy')
axes[3].set_title('Phase')
axes[3].plot(k, cl_phase, label='OpenCL')
axes[3].plot(k, np_phase, label='Numpy')

[ax.locator_params(nbins=2, axis='y') for ax in axes]
plt.xlim([0, NUM_POINTS-1])
plt.show()


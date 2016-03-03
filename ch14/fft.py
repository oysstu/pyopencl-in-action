"""
Listing 14.2: The fast Fourier transform and inverse fourier transform
"""

import numpy as np
import pyopencl as cl
import utility
import matplotlib.pyplot as plt

NUM_POINTS = 2 ** 16
FORWARD_FFT = True  # False if inverse FFT

kernel_src = '''
#define mask_left fft_index
#define mask_right stage
#define shift_pos N2
#define angle size
#define start br.s0
#define cosine x3.s0
#define sine x3.s1
#define wk x2

__kernel void fft_init(__global float2* g_data, __local float2* l_data,
                       uint points_per_group, uint size, int dir) {

   uint4 br, index;
   uint points_per_item, g_addr, l_addr, i, fft_index, stage, N2;
   float2 x1, x2, x3, x4, sum12, diff12, sum34, diff34;

   points_per_item = points_per_group/get_local_size(0);
   l_addr = get_local_id(0) * points_per_item;
   g_addr = get_group_id(0) * points_per_group + l_addr;

   /* Load data from bit-reversed addresses and perform 4-point FFTs */
   for(i=0; i<points_per_item; i+=4) {
      index = (uint4)(g_addr, g_addr+1, g_addr+2, g_addr+3);
      mask_left = size/2;
      mask_right = 1;
      shift_pos = log2((float)size) - 1.0f;
      br = (index << shift_pos) & mask_left;
      br |= (index >> shift_pos) & mask_right;

      /* Bit-reverse addresses */
      while(shift_pos > 1) {
         shift_pos -= 2;
         mask_left >>= 1;
         mask_right <<= 1;
         br |= (index << shift_pos) & mask_left;
         br |= (index >> shift_pos) & mask_right;
      }

      /* Load global data */
      x1 = g_data[br.s0];
      x2 = g_data[br.s1];
      x3 = g_data[br.s2];
      x4 = g_data[br.s3];

      sum12 = x1 + x2;
      diff12 = x1 - x2;
      sum34 = x3 + x4;
      diff34 = (float2)(x3.s1 - x4.s1, x4.s0 - x3.s0) * dir;
      l_data[l_addr] = sum12 + sum34;
      l_data[l_addr+1] = diff12 + diff34;
      l_data[l_addr+2] = sum12 - sum34;
      l_data[l_addr+3] = diff12 - diff34;
      l_addr += 4;
      g_addr += 4;
   }

   /* Perform initial stages of the FFT - each of length N2*2 */
   for(N2 = 4; N2 < points_per_item; N2 <<= 1) {
      l_addr = get_local_id(0) * points_per_item;
      for(fft_index = 0; fft_index < points_per_item; fft_index += 2*N2) {
         x1 = l_data[l_addr];
         l_data[l_addr] += l_data[l_addr + N2];
         l_data[l_addr + N2] = x1 - l_data[l_addr + N2];
         for(i=1; i<N2; i++) {
            cosine = cos(M_PI_F*i/N2);
            sine = dir * sin(M_PI_F*i/N2);
            wk = (float2)(l_data[l_addr+N2+i].s0*cosine + l_data[l_addr+N2+i].s1*sine,
                          l_data[l_addr+N2+i].s1*cosine - l_data[l_addr+N2+i].s0*sine);
            l_data[l_addr+N2+i] = l_data[l_addr+i] - wk;
            l_data[l_addr+i] += wk;
         }
         l_addr += 2*N2;
      }
   }
   barrier(CLK_LOCAL_MEM_FENCE);

   /* Perform FFT with other items in group - each of length N2*2 */
   stage = 2;
   for(N2 = points_per_item; N2 < points_per_group; N2 <<= 1) {
      start = (get_local_id(0) + (get_local_id(0)/stage)*stage) * (points_per_item/2);
      angle = start % (N2*2);
      for(i=start; i<start + points_per_item/2; i++) {
         cosine = cos(M_PI_F*angle/N2);
         sine = dir * sin(M_PI_F*angle/N2);
         wk = (float2)(l_data[N2+i].s0*cosine + l_data[N2+i].s1*sine,
                       l_data[N2+i].s1*cosine - l_data[N2+i].s0*sine);
         l_data[N2+i] = l_data[i] - wk;
         l_data[i] += wk;
         angle++;
      }
      stage <<= 1;
      barrier(CLK_LOCAL_MEM_FENCE);
   }

   /* Store results in global memory */
   l_addr = get_local_id(0) * points_per_item;
   g_addr = get_group_id(0) * points_per_group + l_addr;
   for(i=0; i<points_per_item; i+=4) {
      g_data[g_addr] = l_data[l_addr];
      g_data[g_addr+1] = l_data[l_addr+1];
      g_data[g_addr+2] = l_data[l_addr+2];
      g_data[g_addr+3] = l_data[l_addr+3];
      g_addr += 4;
      l_addr += 4;
   }
}

__kernel void fft_stage(__global float2* g_data, uint stage, uint points_per_group, int dir) {

   uint points_per_item, addr, N, ang, i;
   float c, s;
   float2 input1, input2, w;

   points_per_item = points_per_group/get_local_size(0);
   addr = (get_group_id(0) + (get_group_id(0)/stage)*stage) * (points_per_group/2) +
            get_local_id(0) * (points_per_item/2);
   N = points_per_group*(stage/2);
   ang = addr % (N*2);

   for(i=addr; i<addr + points_per_item/2; i++) {
      c = cos(M_PI_F*ang/N);
      s = dir * sin(M_PI_F*ang/N);
      input1 = g_data[i];
      input2 = g_data[i+N];
      w = (float2)(input2.s0*c + input2.s1*s, input2.s1*c - input2.s0*s);
      g_data[i] = input1 + w;
      g_data[i+N] = input1 - w;
      ang++;
   }
}

__kernel void fft_scale(__global float2* g_data, uint points_per_group, uint scale) {

   uint points_per_item, addr, i;

   points_per_item = points_per_group/get_local_size(0);
   addr = get_group_id(0) * points_per_group + get_local_id(0) * points_per_item;

   for(i=addr; i<addr + points_per_item; i++) {
      g_data[i] /= scale;
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

# Determine device local memory size
# Truncate to nearest power of two (to stick to the method the book uses)
local_mem_size = 2 ** np.trunc(np.log2(dev.local_mem_size)).astype(np.int32)

# Generate input data by superimposing sines/cosines of different frequencies
input_source = np.zeros((NUM_POINTS,), dtype=np.float32)
sign = 1
for i in range(0, np.int32(np.log2(NUM_POINTS)) - 1):
    input_source += sign*np.sin(np.linspace(start=0, stop=np.pi*(2**i), num=NUM_POINTS, dtype=np.float32))
    input_source -= sign*np.cos(np.linspace(start=0, stop=np.pi*(2**i), num=NUM_POINTS, dtype=np.float32))
    sign *= -1

# Data and device buffers
input_data = np.zeros(shape=(NUM_POINTS*2,), dtype=np.float32)
if FORWARD_FFT:
    input_data[::2] = input_source
else:
    np_fft = np.fft.fft(input_source)
    input_data[::2] = np.real(np_fft)
    input_data[1::2] = np.imag(np_fft)

output_data = np.empty_like(input_data, dtype=np.float32)

mf = cl.mem_flags
data_buffer = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=input_data)
local_buffer = cl.LocalMemory(local_mem_size)

# Initial kernel
# fft_init(__global float2* g_data, __local float2* l_data, uint points_per_group, uint size, int dir)
kernel_init = prog.fft_init

# Determine maximum work-group size
kernel_wg_size = kernel_init.get_work_group_info(cl.kernel_work_group_info.WORK_GROUP_SIZE, dev)
local_size = 2 ** np.trunc(np.log2(kernel_wg_size)).astype(np.int32)

# Kernel parameters
direction = np.int32(1 if FORWARD_FFT else -1)
points_per_group = np.uint32(local_mem_size // (2*np.dtype(np.float32).itemsize))
print('Points per group: ' + str(points_per_group))
if points_per_group > NUM_POINTS:
    points_per_group = NUM_POINTS

global_size = (NUM_POINTS // points_per_group)*local_size
print('Global size: ' + str(global_size))
print('Local size\n: ' + str(local_size))

kernel_init(queue, (global_size,), (local_size,),
            data_buffer,
            local_buffer,
            points_per_group,
            np.uint32(NUM_POINTS),
            direction)


# There is some overhead involved with spawning a new kernel (code caching)
# A good rule of thumb is therefore to create the kernel object outside of loops
# Ref: https://lists.tiker.net/pipermail/pyopencl/2016-February/002107.html
kernel_stage = prog.fft_stage

# Enqueue further stages of the FFT
if NUM_POINTS > points_per_group:
    # for(stage = 2; stage <= num_points/points_per_group; stage <<= 1)
    for stage in utility.range_bitwise_shift(low=2, high=NUM_POINTS//points_per_group + 1, n=1):
        print('Stage: ' + str(stage))
        # fft_stage(__global float2* g_data, uint stage, uint points_per_group, int dir)
        kernel_stage(queue, (global_size,), (local_size,),
                     data_buffer,
                     np.uint32(stage),
                     points_per_group,
                     direction)

# Scale values if performing the inverse FFT
if not FORWARD_FFT:
    # fft_scale(__global float2* g_data, uint points_per_group, uint scale)
    prog.fft_scale(queue, (global_size,), (local_size,), data_buffer, points_per_group, np.uint32(NUM_POINTS))

# Read results
cl.enqueue_copy(queue, dest=output_data, src=data_buffer, is_blocking=True)

# Change to array of complex values
cl_fft = output_data[::2] + 1j * output_data[1::2]  # From there, real and im. alternates

# Visualize results to compare with numpy
if FORWARD_FFT:
    # Compute numpy fft for comparison
    np_fft = np.fft.fft(input_source)

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
else:
    # Result only has a real component
    cl_real = np.real(cl_fft)
    np_real = np.real(np.fft.ifft(np_fft))

    # Print first ten inverse values
    np.set_printoptions(precision=4, suppress=True)
    print('CL iFFT [0:10]:')
    print(cl_real[:10])
    print('\nNumpy iFFT [0:10]:')
    print(np_real[:10])

    k = np.arange(0, NUM_POINTS)

    plt.figure()
    plt.title('Inverse FFT results')
    plt.plot(k, cl_real, label='OpenCL')
    plt.plot(k, np_real, label='Numpy')
    plt.legend()

plt.xlim([0, NUM_POINTS-1])
plt.show()


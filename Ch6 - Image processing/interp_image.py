'''
Listing 6.2: Image interpolation
'''

import numpy as np
import pyopencl as cl
import pyopencl.array
import matplotlib.pyplot as plt
from skimage.io import imread
import utility

from os import environ
environ['PYOPENCL_COMPILER_OUTPUT'] = '0'

scale_factor = 5

kernel_src = '''
__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void interp(read_only image2d_t src_image,
                     write_only image2d_t dst_image) {

   float4 pixel;

   /* Determine input coordinate */
   float2 input_coord = (float2)
      (get_global_id(0) + (1.0f/(SCALE*2)),
       get_global_id(1) + (1.0f/(SCALE*2)));

   /* Determine output coordinate */
   int2 output_coord = (int2)
      (SCALE*get_global_id(0),
       SCALE*get_global_id(1));

   /* Compute interpolation */
   for(int i=0; i<SCALE; i++) {
      for(int j=0; j<SCALE; j++) {
         pixel = read_imagef(src_image, sampler,
           (float2)(input_coord +
           (float2)(1.0f*i/SCALE, 1.0f*j/SCALE)));

         write_imagef(dst_image, output_coord + (int2)(i, j), pixel);
      }
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
    #'-Werror'
    prog.build(options=['-DSCALE={}'.format(scale_factor)], devices=[dev], cache_dir=None)
except:
    print('Build log:')
    print(prog.get_build_info(dev, cl.program_build_info.LOG))
    raise

# Data and buffers
im_src = imread('input_car.png').astype(dtype=np.uint16)
shape_dst = (im_src.shape[0]*scale_factor, im_src.shape[1]*scale_factor)
im_dst = np.empty(shape=shape_dst, dtype=np.uint16)

src_buff = cl.image_from_array(context, im_src, mode='r')
dst_buff = cl.image_from_array(context, im_dst, mode='w')

# Enqueue kernel
# Note: Global indices is reversed due to OpenCL using column-major order when reading images
global_size = im_src.shape[::-1]
local_size = None

# __call__(queue, global_size, local_size, *args, global_offset=None, wait_for=None, g_times_l=False)
prog.interp(queue, global_size, local_size, src_buff, dst_buff)

# Enqueue command to copy from buffers to host memory
# Note: Region indices is reversed due to OpenCL using column-major order when reading images
cl.enqueue_copy(queue, dest=im_dst, src=dst_buff, is_blocking=True, origin=(0, 0), region=im_dst.shape[::-1])

# Plot images with built-in scaling disabled
plt.figure()
plt.figimage(im_src, cmap='gray', vmin=0, vmax=np.iinfo(np.uint16).max)
plt.figure()
plt.figimage(im_dst, cmap='gray', vmin=0, vmax=np.iinfo(np.uint16).max)
plt.show()

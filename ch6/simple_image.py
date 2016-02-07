'''
Listing 6.1: Simple image processing
'''

import numpy as np
import pyopencl as cl
import matplotlib.pyplot as plt
import utility

kernel_src = '''
__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
      CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void simple_image(read_only image2d_t src_image,
                        write_only image2d_t dst_image) {

   /* Compute value to be subtracted from each pixel */
   uint offset = get_global_id(1) * 0x4000 + get_global_id(0) * 0x1000;

   /* Read pixel value */
   int2 coord = (int2)(get_global_id(0), get_global_id(1));
   uint4 pixel = read_imageui(src_image, sampler, coord);

   /* Subtract offset from pixel */
   pixel.x -= offset;

   /* Write new pixel value to output */
   write_imageui(dst_image, coord, pixel);
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
im_src = np.full(shape=(4, 4), fill_value=np.iinfo(np.uint16).max, dtype=np.uint16)
im_dst = np.empty_like(im_src, dtype=np.uint16)

src_buff = cl.image_from_array(context, im_src, mode='r')
dst_buff = cl.image_from_array(context, im_dst, mode='w')

# Enqueue kernel (with argument specified directly)
# Note: Global indices is reversed due to OpenCL using column-major order when reading images
global_size = im_src.shape[::-1]
local_size = None

# __call__(queue, global_size, local_size, *args, global_offset=None, wait_for=None, g_times_l=False)
prog.simple_image(queue, global_size, local_size, src_buff, dst_buff)

# Enqueue command to copy from buffers to host memory
# Note: Region indices is reversed due to OpenCL using column-major order when reading images
cl.enqueue_copy(queue, dest=im_dst, src=dst_buff, is_blocking=True, origin=(0, 0), region=im_src.shape[::-1])

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(im_src, cmap='gray', vmin=0, vmax=np.iinfo(np.uint16).max)
ax2.imshow(im_dst, cmap='gray', vmin=0, vmax=np.iinfo(np.uint16).max, interpolation='nearest')
plt.show()

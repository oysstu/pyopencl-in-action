'''
Listing 5.1: Operator usage (and vector usage)
'''

import pyopencl as cl
import pyopencl.array
import utility

kernel_src = '''
__kernel void op_test(__global int4 *output) {
   int4 vec = (int4)(1, 2, 3, 4);

   /* Adds 4 to every element of vec */
   vec += 4;

   /* Sets the third element to 0
      Doesn't change the other elements
      (-1 in hexadecimal = 0xFFFFFFFF */
   if(vec.s2 == 7){
      vec &= (int4)(-1, -1, 0, -1);
    }

   /* Sets the first element to -1, the second to 0 */
   vec.s01 = vec.s23 < 7;

   /* Divides the last element by 2 until it is less than or equal to 7 */
   while(vec.s3 > 7 && (vec.s0 < 16 || vec.s1 < 16)){
      vec.s3 >>= 1;
    }

   *output = vec;
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
out = cl.array.vec.zeros_int4()
buffer_out = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, size=out.itemsize)

# Enqueue kernel (with argument specified directly)
n_globals = (1,)
n_locals = None
prog.op_test(queue, n_globals, n_locals, buffer_out)

# Enqueue command to copy from buffer_out to host memory
cl.enqueue_copy(queue, dest=out, src=buffer_out, is_blocking=True)

print('Output: ' + str(out))


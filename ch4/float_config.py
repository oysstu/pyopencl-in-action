'''
Listing 4.3: Testing a device’s floating-point features
'''

import pyopencl as cl
import utility

# Get device and context, create command queue and program
dev = utility.get_default_device()

# Check for double floating point features
fp_flag = dev.single_fp_config

fp_masks = [('Denorm',              cl.device_fp_config.DENORM),
            ('Fused multiply-add',  cl.device_fp_config.FMA),
            ('INF & NAN',           cl.device_fp_config.INF_NAN),
            ('Round to INF',        cl.device_fp_config.ROUND_TO_INF),
            ('Round to nearest',    cl.device_fp_config.ROUND_TO_NEAREST),
            ('Round to zero',       cl.device_fp_config.ROUND_TO_ZERO)]

version_number = float(dev.version.split(' ')[1])

if version_number >= 1.1:
    fp_masks.append(('Soft float', cl.device_fp_config.SOFT_FLOAT))

if version_number >= 1.2:
    fp_masks.append(('Correctly rounded div sqrt', cl.device_fp_config.CORRECTLY_ROUNDED_DIVIDE_SQRT))

print('Floating point features:')
[print('\t{0:<30}{1:<5}'.format(name, str(bool(fp_flag & mask)))) for name, mask in fp_masks]


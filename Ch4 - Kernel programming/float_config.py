'''
Listing 4.3: Testing a device’s floating-point features
'''

import pyopencl as cl
import utility

# Get device and context, create command queue and program
dev = utility.get_default_device()

# Check for double floating point features
fp_flag = dev.single_fp_config


print('Floating point features:')
print('\tDenorm: ' + str(bool(fp_flag & cl.device_fp_config.DENORM)))
print('\tFused multiply-add: ' + str(bool(fp_flag & cl.device_fp_config.FMA)))
print('\tINF & NAN: ' + str(bool(fp_flag & cl.device_fp_config.INF_NAN)))
print('\tRound to INF: ' + str(bool(fp_flag & cl.device_fp_config.ROUND_TO_INF)))
print('\tRound to nearest: ' + str(bool(fp_flag & cl.device_fp_config.ROUND_TO_NEAREST)))
print('\tRound to zero: ' + str(bool(fp_flag & cl.device_fp_config.ROUND_TO_ZERO)))

version_number = float(dev.version.split(' ')[1])

if version_number >= 1.1:
    print('\tSoft float: ' + str(bool(fp_flag & cl.device_fp_config.SOFT_FLOAT)))

if version_number >= 1.2:
    print('\tCorrectly rounded div sqrt: ' + str(bool(fp_flag & cl.device_fp_config.CORRECTLY_ROUNDED_DIVIDE_SQRT)))


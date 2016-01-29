import pyopencl as cl
from operator import attrgetter

'''
Retrieves the GPU device with the most global memory
'''
def get_default_device():
    platforms = cl.get_platforms()
    gpu_devices = [plat.get_devices(cl.device_type.GPU) for plat in platforms]
    gpu_devices = [dev for devices in gpu_devices for dev in devices]  # Flatten to 1d if multiple GPU devices exists

    if len(gpu_devices) == 0:
        print('No GPU devices found')

    dev = max(gpu_devices, key=attrgetter('global_mem_size'))
    print('Using GPU: ' + dev.name)
    return dev


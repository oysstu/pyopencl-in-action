import pyopencl as cl
from operator import attrgetter


def get_default_device() -> cl.Device:
    '''
    Retrieves the GPU device with the most global memory if available, otherwise returns the CPU.
    '''
    platforms = cl.get_platforms()
    gpu_devices = [plat.get_devices(cl.device_type.GPU) for plat in platforms]
    gpu_devices = [dev for devices in gpu_devices for dev in devices]  # Flatten to 1d if multiple GPU devices exists

    if gpu_devices:
        dev = max(gpu_devices, key=attrgetter('global_mem_size'))
        print('Using GPU: {}\n'.format(dev.name))
        return dev
    else:
        cpu_devices = [plat.get_devices(cl.device_type.CPU) for plat in platforms]
        cpu_devices = [dev for devices in cpu_devices for dev in devices]
        if cpu_devices:
            dev = max(cpu_devices, key=attrgetter('global_mem_size'))
            print('Using CPU: {}\n'.format(dev.name))
            return dev
        else:
            raise RuntimeError('No suitable OpenCL GPU/CPU devices found')



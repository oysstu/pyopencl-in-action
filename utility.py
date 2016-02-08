import pyopencl as cl
from operator import attrgetter
from typing import List


def get_default_device(use_gpu=True) -> cl.Device:
    '''
    Retrieves the GPU device with the most global memory if available, otherwise returns the CPU.
    '''
    platforms = cl.get_platforms()
    gpu_devices = [plat.get_devices(cl.device_type.GPU) for plat in platforms]
    gpu_devices = [dev for devices in gpu_devices for dev in devices]  # Flatten to 1d if multiple GPU devices exists

    if gpu_devices and use_gpu:
        dev = max(gpu_devices, key=attrgetter('global_mem_size'))
        print('Using GPU: {}'.format(dev.name))
        print('On platform: {} ({})\n'.format(dev.platform.name, dev.platform.version.strip()))
        return dev
    else:
        cpu_devices = [plat.get_devices(cl.device_type.CPU) for plat in platforms]
        cpu_devices = [dev for devices in cpu_devices for dev in devices]
        if cpu_devices:
            dev = max(cpu_devices, key=attrgetter('global_mem_size'))
            print('Using CPU: {}'.format(dev.name))
            print('On platform: {} ({})\n'.format(dev.platform.name, dev.platform.version.strip()))
            return dev
        else:
            raise RuntimeError('No suitable OpenCL GPU/CPU devices found')


def get_devices_by_name(name) -> List[cl.Device]:
    if not name:
        raise RuntimeError('Device name must be specified')

    platforms = cl.get_platforms()
    devices = [plat.get_devices(cl.device_type.ALL) for plat in platforms]
    devices = [dev for devices in devices for dev in devices]
    name_matches = [dev for dev in devices if dev.name == name]

    return name_matches


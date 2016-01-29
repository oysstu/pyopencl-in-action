'''
Prints relevant information regarding the capabilities of the current OpenCL runtime and devices
'''

import pyopencl as cl

print('PyOpenCL version: ' + cl.VERSION_TEXT)
print('OpenCL header version: ' + '.'.join(map(str, cl.get_cl_header_version())) + '\n')

# Get installed platforms (SDKs)
print('- Installed platforms (SDKs) and available devices:')
platforms = cl.get_platforms()

for plat in platforms:
    indent = ''

    # Get and print platform info
    print(indent + '{} ({})'.format(plat.name, plat.vendor))
    indent = '\t'
    print(indent + 'Version: ' + plat.version)
    print(indent + 'Profile: ' + plat.profile)
    print(indent + 'Extensions: ' + str(plat.extensions.strip().split(' ')))

    # Get and print device info
    devices = plat.get_devices(cl.device_type.ALL)

    print(indent + 'Available devices: ')
    if len(devices) == 0:
        print(indent + '\tNone')

    for dev in devices:
        indent = '\t\t'
        print(indent + '{} ({})'.format(dev.name, dev.vendor))

        indent = '\t\t\t'
        print(indent + 'Version: ' + dev.version)
        print(indent + 'Type: ' + cl.device_type.to_string(dev.type))
        print(indent + 'Extensions: ' + str(dev.extensions.strip().split(' ')))
        print(indent + 'Memory (global): ' + str(dev.global_mem_size))
        print(indent + 'Memory (local): ' + str(dev.local_mem_size))
        print(indent + 'Address bits: ' + str(dev.address_bits))
        print(indent + 'Max work item dims: ' + str(dev.max_work_item_dimensions))
        print(indent + 'Max work group size: ' + str(dev.max_work_group_size))
        print(indent + 'Max compute units: ' + str(dev.max_compute_units))
        print(indent + 'Driver version: ' + dev.driver_version)
        print(indent + 'Little endian: ' + str(bool(dev.endian_little)))
        print(indent + 'Device available: ' + str(bool(dev.available)))
        print(indent + 'Compiler available: ' + str(bool(dev.compiler_available)))

        # Device version string has the following syntax, extract the number like this
        # OpenCL<space><major_version.minor_version><space><vendor-specific information>
        version_number = float(dev.version.split(' ')[1])

    print('')


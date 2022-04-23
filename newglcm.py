import pyopencl as cl
import numpy as np
from skimage import io as si
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte
import math
import os
import time

# Enable more elaborate OpenCL compiler output
os.environ["PYOPENCL_COMPILER_OUTPUT"] = "1"

def platformselect(index):
    """
    Returns the platform at the specified index in the list of
    available platforms found by PyOpenCL

    :param index: the index of the platform
    :type index: int
    :return: a PyOpenCL Platform object
    :rtype: pyopencl.Platform
    """
    return [platform for platform in cl.get_platforms()][index]

# Define parameters for the GLCM computation
windowsz = 13
hws = windowsz // 2
dx = -1
dy = 3

# Get the devices from platform 0 (usually computers have only 1 platform, hence the index 0)
platform_devices = platformselect(0).get_devices()
context = cl.Context(devices=platform_devices)
src = open("glcm_compute.cl", "r").read()
prgs_src = cl.Program(context, src)
prgs = prgs_src.build(["-DSYMMETRIC", f"-DWINDOW_SIZE={windowsz}"])
print(platform_devices)

# Open the image
img = img_as_ubyte(rgb2gray(si.imread("./ischdown20x.png")))

# Print the dimensions of the image
print(img.shape)

# Initialize buffers for computation
res = np.zeros((7, img.shape[0] - 2 * hws, img.shape[1] - 2 * hws), dtype=np.float32)
img_buff = cl.Buffer(context, flags=cl.mem_flags.READ_ONLY, size=img.nbytes)
res_buff = cl.Buffer(context, flags=cl.mem_flags.WRITE_ONLY, size=res.nbytes)
# Add other buffers

queue = cl.CommandQueue(context)

# Copy buffers to device
inp = [(img, img_buff), (res, res_buff)]
out = [(res, res_buff)]
for (arr, buff) in inp:
    cl.enqueue_copy(queue, src=arr, dest=buff)

# Kernel arguments
krn_args = [img_buff, res_buff, np.int32(dx), np.int32(dy), np.int32(img.shape[0]),
            np.int32(img.shape[1])]
print((math.ceil(img.shape[0] / windowsz), math.ceil(img.shape[1] / windowsz)))

# Execute kernel
begin = time.perf_counter()
completedEvent = prgs.glcmgen(queue, (math.ceil(img.shape[0] / windowsz), math.ceil(img.shape[1] / windowsz)), (1, 1),
                              *krn_args)
completedEvent.wait()  # (math.ceil(img.shape[0]/windowsz),math.ceil(img.shape[1]/windowsz))
print(time.perf_counter() - begin)

# Copy buffers from device
for (arr, buff) in out:
    cl.enqueue_copy(queue, src=buff, dest=arr)
queue.finish()

# Save all images to files
for i in range(7):
    si.imsave(f"result{i}.tif", res[i], check_contrast=False)

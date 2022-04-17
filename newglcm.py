import pyopencl as cl
import numpy as np
from skimage import io as si
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte
import math
import os

print(os.getcwd())
os.environ["PYOPENCL_COMPILER_OUTPUT"] = "1"


def platformslist():
    return [platform.name for platform in cl.get_platforms()]


def platformselect(ind):
    return [platform for platform in cl.get_platforms()][ind]


platform_devices = platformselect(0).get_devices()
context = cl.Context(devices=platform_devices)
src = open("glcm_compute.cl").read()
prgs_src = cl.Program(context, src)
prgs = prgs_src.build()
print(platform_devices)

windowsz = 13
hws = windowsz // 2
dx = -1
dy = 3

img = img_as_ubyte(rgb2gray(si.imread("./ischdown20x.png")))
# img=img_as_ubyte(rgb2gray(si.imread("/home/simo1427/Documents/jBioMed Data/MkyISH_SVZi1/ischdown8x.png")))
print(img.shape)
res = np.zeros((7, img.shape[0] - 2 * hws, img.shape[1] - 2 * hws), dtype=np.float32)
graypair_t = np.dtype([("ref", np.uint8), ("val", np.uint8), ("weight", np.float32)])
graypair_t, graypair_t_c_decl = cl.tools.match_dtype_to_c_struct(context.devices[0], "graypair_t", graypair_t)
graypair_t = cl.tools.get_or_register_dtype("graypair_t", graypair_t)
pair = np.zeros(((img.shape[0]), (img.shape[1])), dtype=graypair_t)
print(pair.shape)
img_buff = cl.Buffer(context, flags=cl.mem_flags.READ_ONLY, size=img.nbytes)
res_buff = cl.Buffer(context, flags=cl.mem_flags.WRITE_ONLY, size=res.nbytes)
pair_buff = cl.Buffer(context, flags=cl.mem_flags.READ_WRITE, size=pair.nbytes)
# add other buffers

queue = cl.CommandQueue(context)

inp = [(img, img_buff), (res, res_buff), (pair, pair_buff)]
out = [(res, res_buff), (pair, pair_buff)]
for (arr, buff) in inp:
    cl.enqueue_copy(queue, src=arr, dest=buff)

krn_args = [img_buff, res_buff, pair_buff, np.int32(dx), np.int32(dy), np.int32(windowsz), np.int32(img.shape[0]),
            np.int32(img.shape[1])]
print((math.ceil(img.shape[0] / windowsz), math.ceil(img.shape[1] / windowsz)))
import time

begin = time.perf_counter()
completedEvent = prgs.glcmgen(queue, (math.ceil(img.shape[0] / windowsz), math.ceil(img.shape[1] / windowsz)), (1, 1),
                              *krn_args)
completedEvent.wait()  # (math.ceil(img.shape[0]/windowsz),math.ceil(img.shape[1]/windowsz))
print(time.perf_counter() - begin)
for (arr, buff) in out:
    cl.enqueue_copy(queue, src=buff, dest=arr)
queue.finish()
"""calculatedpairs=np.zeros((26,),dtype=graypair_t)
k=0
for i in range(2):
    for j in range(13):
        calculatedpairs[k]=pair[i,j]
        k+=1
calculatedpairs.sort(order="ref")
print(calculatedpairs)
#print(pair[:13,:13])"""
# si.imsave(f"memoryref.tif", pair["ref"], check_contrast=False)
# si.imsave(f"memoryval.tif", pair["val"], check_contrast=False)
# si.imsave(f"memorywei.tif", pair["weight"], check_contrast=False)
for i in range(7):
    si.imsave(f"result{i}.tif", res[i], check_contrast=False)
